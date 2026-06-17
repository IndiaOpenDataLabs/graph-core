"""Tenant-scoped FalkorDB proxy for the Browser UI.

The Browser only understands a single Redis/FalkorDB endpoint. This proxy
accepts Browser connections, authenticates them against the stored namespace
credential, then constrains graph visibility to the namespace's graph prefix.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from graph_core.config import settings
from graph_core.database import AsyncSessionLocal
from graph_core.models.credential import Credential
from graph_core.models.namespace import Namespace
from graph_core.services.crypto import CredentialCrypto

try:
    from redis.asyncio import Redis
    from redis.exceptions import RedisError
except ImportError as exc:  # pragma: no cover - only if optional deps missing
    raise RuntimeError("redis is required to run the FalkorDB proxy") from exc

logger = logging.getLogger(__name__)


class ProxyError(Exception):
    """Protocol-level proxy error."""


@dataclass(frozen=True)
class NamespaceAuth:
    namespace_id: str
    namespace_name: str
    username: str
    password: str
    graph_prefix: str
    upstream_url: str
    db: int


@dataclass(frozen=True)
class SimpleString:
    value: str


@dataclass(frozen=True)
class ErrorReply:
    message: str


def _is_permission_error_message(message: str) -> bool:
    normalized = message.lower()
    return (
        "no permissions to access a key" in normalized
        or "noperm" in normalized
        or "permission denied" in normalized
    )


class FalkorDBTenantProxy:
    """RESP proxy that scopes a Browser connection to one namespace."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        upstream_default_url: str | None = None,
    ) -> None:
        self._host = host or os.getenv("FALKORDB_PROXY_HOST", "0.0.0.0")
        self._port = port or int(os.getenv("FALKORDB_PROXY_PORT", "6381"))
        self._upstream_default_url = (
            upstream_default_url
            or os.getenv("FALKORDB_PROXY_UPSTREAM_URL")
            or settings.falkordb_url
            or "redis://localhost:6379"
        )
        self._crypto = CredentialCrypto()

    async def serve(self) -> None:
        server = await asyncio.start_server(
            self._handle_client,
            self._host,
            self._port,
        )
        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        logger.info(
            "FalkorDB proxy listening on %s",
            addrs or f"{self._host}:{self._port}",
        )
        async with server:
            await server.serve_forever()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session = _ProxySession(
            proxy=self,
            reader=reader,
            writer=writer,
        )
        try:
            await session.run()
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _resolve_namespace_auth(
        self, username: str, password: str
    ) -> NamespaceAuth | None:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Credential)
                .where(
                    Credential.provider == "falkordb",
                    Credential.label == username,
                )
                .order_by(Credential.created_at.desc())
            )
            credential = result.scalars().first()
            if credential is None:
                return None
            if self._crypto.decrypt(credential.encrypted_secret) != password:
                return None

            namespace = await session.get(Namespace, credential.namespace_id)
            if namespace is None:
                return None
            db = int(namespace.falkordb_db or 0)
            return NamespaceAuth(
                namespace_id=str(namespace.id),
                namespace_name=namespace.name,
                username=username,
                password=password,
                graph_prefix=f"tenant:{namespace.id}:",
                upstream_url=(
                    credential.base_url or self._upstream_default_url
                ).strip(),
                db=db,
            )

    @staticmethod
    def _normalize_upstream_url(url: str) -> str:
        normalized = (url or "").strip()
        if normalized.startswith("falkordb://"):
            normalized = "redis://" + normalized[len("falkordb://") :]
        return normalized


class _ProxySession:
    """Per-connection proxy state."""

    def __init__(
        self,
        *,
        proxy: FalkorDBTenantProxy,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._proxy = proxy
        self._reader = reader
        self._writer = writer
        self._auth: NamespaceAuth | None = None
        self._upstream: Redis | None = None

    async def run(self) -> None:
        while True:
            try:
                command = await _read_command(self._reader)
            except EOFError:
                return
            if command is None:
                return
            reply = await self.handle_command(command)
            if reply is None:
                continue
            self._writer.write(_encode_reply(reply))
            await self._writer.drain()
            if command and command[0].upper() == "QUIT":
                return

    async def handle_command(self, command: list[str]) -> Any | None:
        try:
            if not command:
                return ErrorReply("ERR empty command")

            name = command[0].upper()
            args = command[1:]
            logger.debug(
                "Proxy command name=%s arg0=%s arg_count=%d",
                name,
                args[0] if args else "",
                len(args),
            )

            if name == "QUIT":
                return SimpleString("OK")

            if name == "AUTH":
                return await self._handle_auth(args)

            if self._auth is None:
                if name in {"PING", "HELLO", "CLIENT"}:
                    return await self._handle_pre_auth(name, args)
                return ErrorReply("NOAUTH Authentication required.")

            if name == "SELECT":
                return SimpleString("OK")

            if name == "PING":
                return SimpleString("PONG")

            if name == "CLIENT" and args:
                subcommand = args[0].upper()
                if subcommand in {"SETINFO", "SETNAME"}:
                    return SimpleString("OK")

            if name == "GRAPH.LIST":
                try:
                    result = await self._execute_upstream(name, args)
                except ProxyError as exc:
                    if _is_permission_error_message(str(exc)):
                        return []
                    raise
                return self._filter_graph_list(result)

            if name == "GRAPH.UDF" and args and args[0].upper() == "LIST":
                try:
                    result = await self._execute_upstream(name, args)
                except ProxyError as exc:
                    if _is_permission_error_message(str(exc)):
                        return []
                    raise
                return result if result is not None else []

            if name.startswith("GRAPH."):
                self._ensure_allowed_graph_args(command)
                try:
                    return await self._execute_upstream(name, args)
                except ProxyError as exc:
                    if not _is_permission_error_message(str(exc)):
                        raise
                    if name in {"GRAPH.QUERY", "GRAPH.RO_QUERY"}:
                        return []
                    if name == "GRAPH.MEMORY":
                        return 0
                    if name == "GRAPH.DELETE":
                        return SimpleString("OK")
                    return []

            if name == "MODULE" and args and args[0].upper() == "LIST":
                return await self._execute_upstream(name, args)

            if name == "INFO":
                result = await self._execute_upstream(name, args)
                return self._format_info_reply(result)

            if name == "COMMAND":
                return await self._execute_upstream(name, args)

            if name == "ACL" and args and args[0].upper() in {"WHOAMI", "GETUSER"}:
                return await self._execute_upstream(name, args)

            if name in {"TIME", "ECHO"}:
                return await self._execute_upstream(name, args)

            return ErrorReply(
                f"NOPERM command '{name}' is not allowed through the proxy"
            )
        except ProxyError as exc:
            return ErrorReply(str(exc))

    async def _handle_pre_auth(self, name: str, args: list[str]) -> Any:
        if name == "PING":
            return SimpleString("PONG")
        if name == "CLIENT":
            return SimpleString("OK")
        if name == "HELLO":
            return ErrorReply("NOAUTH Authentication required.")
        return ErrorReply("NOAUTH Authentication required.")

    async def _handle_auth(self, args: list[str]) -> Any:
        if len(args) != 2:
            return ErrorReply("ERR wrong number of arguments for 'AUTH'")
        username, password = args
        logger.info("Proxy AUTH attempt for username=%s", username)
        auth = await self._proxy._resolve_namespace_auth(username, password)
        if auth is None:
            logger.info("Proxy AUTH rejected for username=%s", username)
            return ErrorReply(
                "WRONGPASS invalid username-password pair or user is disabled."
            )
        await self._close_upstream()
        self._auth = auth
        logger.info(
            "Proxy AUTH accepted for namespace_id=%s graph_prefix=%s",
            auth.namespace_id,
            auth.graph_prefix,
        )
        return SimpleString("OK")

    async def _open_upstream(self) -> None:
        if self._auth is None:
            return
        upstream_url = self._proxy._normalize_upstream_url(self._auth.upstream_url)
        self._upstream = Redis.from_url(
            upstream_url,
            username=self._auth.username,
            password=self._auth.password,
            db=self._auth.db,
            decode_responses=True,
        )
        await self._upstream.ping()

    async def _close_upstream(self) -> None:
        if self._upstream is not None:
            await self._upstream.aclose()
            self._upstream = None

    async def _execute_upstream(self, command: str, args: list[str]) -> Any:
        if self._upstream is None:
            await self._open_upstream()
        if self._upstream is None:
            raise ProxyError("upstream not connected")
        try:
            return await self._upstream.execute_command(command, *args)
        except RedisError as exc:
            # Preserve a RESP error reply so the Browser sees a clean failure
            # instead of a socket reset when FalkorDB rejects a probe command.
            raise ProxyError(str(exc)) from exc

    def _ensure_allowed_graph_args(self, command: list[str]) -> None:
        if self._auth is None or len(command) < 2:
            raise ProxyError("graph command requires a graph name")
        graph_name = command[1]
        if not graph_name.startswith(self._auth.graph_prefix):
            raise ProxyError("No permissions to access a key")

    def _filter_graph_list(self, result: Any) -> Any:
        if self._auth is None:
            return result
        if isinstance(result, list):
            filtered: list[Any] = []
            for item in result:
                if isinstance(item, str) and item.startswith(self._auth.graph_prefix):
                    filtered.append(item)
                elif isinstance(item, (list, tuple)) and item:
                    first = item[0]
                    if isinstance(first, str) and first.startswith(
                        self._auth.graph_prefix
                    ):
                        filtered.append(item)
            return filtered
        return result

    @staticmethod
    def _format_info_reply(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            sections = [
                "\n".join(f"{key}:{value}" for key, value in result.items())
            ]
            payload = "\n".join(section for section in sections if section)
            return payload + ("\n" if payload else "")
        if isinstance(result, (list, tuple)):
            lines = [
                str(item).rstrip("\r\n")
                for item in result
                if str(item).strip()
            ]
            payload = "\n".join(lines)
            return payload + ("\n" if payload else "")
        payload = str(result).rstrip("\r\n")
        return payload + ("\n" if payload else "")


def _parse_inline_command(line: bytes) -> list[str]:
    parts = line.decode("utf-8", errors="replace").strip().split()
    return [part for part in parts if part]


async def _read_command(reader: asyncio.StreamReader) -> list[str] | None:
    line = await reader.readline()
    if not line:
        raise EOFError
    if line == b"\r\n":
        return None
    prefix = line[:1]
    if prefix != b"*":
        return _parse_inline_command(line)
    try:
        count = int(line[1:].strip())
    except ValueError as exc:
        raise ProxyError("ERR invalid multibulk length") from exc
    parts: list[str] = []
    for _ in range(count):
        bulk_len_line = await reader.readline()
        if not bulk_len_line or bulk_len_line[:1] != b"$":
            raise ProxyError("ERR expected bulk string")
        try:
            bulk_len = int(bulk_len_line[1:].strip())
        except ValueError as exc:
            raise ProxyError("ERR invalid bulk length") from exc
        if bulk_len < 0:
            parts.append("")
            continue
        data = await reader.readexactly(bulk_len)
        terminator = await reader.readexactly(2)
        if terminator != b"\r\n":
            raise ProxyError("ERR invalid bulk string terminator")
        parts.append(data.decode("utf-8", errors="replace"))
    return parts


def _encode_reply(reply: Any) -> bytes:
    if isinstance(reply, ProxyError):
        return _encode_error(str(reply))
    if isinstance(reply, ErrorReply):
        return _encode_error(reply.message)
    if isinstance(reply, SimpleString):
        return _encode_simple_string(reply.value)
    if reply is None:
        return b"$-1\r\n"
    if isinstance(reply, bool):
        return f":{1 if reply else 0}\r\n".encode()
    if isinstance(reply, int):
        return f":{reply}\r\n".encode()
    if isinstance(reply, (bytes, bytearray, memoryview)):
        payload = bytes(reply)
        return b"$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n"
    if isinstance(reply, str):
        if reply in {"OK", "PONG", "QUEUED"}:
            return _encode_simple_string(reply)
        payload = reply.encode()
        return b"$" + str(len(payload)).encode() + b"\r\n" + payload + b"\r\n"
    if isinstance(reply, dict):
        items: list[Any] = []
        for key, value in reply.items():
            items.extend([key, value])
        return b"*" + str(len(items)).encode() + b"\r\n" + b"".join(
            _encode_reply(item) for item in items
        )
    if isinstance(reply, (list, tuple)):
        return b"*" + str(len(reply)).encode() + b"\r\n" + b"".join(
            _encode_reply(item) for item in reply
        )
    return _encode_simple_string(str(reply))


def _encode_simple_string(value: str) -> bytes:
    return b"+" + value.encode() + b"\r\n"


def _encode_error(message: str) -> bytes:
    return b"-" + message.encode() + b"\r\n"


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    proxy = FalkorDBTenantProxy()
    asyncio.run(proxy.serve())
