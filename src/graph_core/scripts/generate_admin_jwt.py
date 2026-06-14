"""Generate an admin JWT for Graph Core MCP/API clients.

The token is signed with `JWT_SECRET` and carries:
- `token_type=admin`
- `scope=graph-core:admin`

Usage:
    uv run python -m graph_core.scripts.generate_admin_jwt
    uv run python -m graph_core.scripts.generate_admin_jwt --subject my-client
    uv run graph-core-admin-jwt --subject my-client
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import jwt

from graph_core.config import settings

ADMIN_SCOPE = "graph-core:admin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Graph Core admin JWT")
    parser.add_argument(
        "--secret",
        default=os.getenv("JWT_SECRET") or settings.jwt_secret,
        help="JWT signing secret (defaults to JWT_SECRET)",
    )
    parser.add_argument(
        "--issuer",
        default=os.getenv("JWT_ISSUER") or settings.jwt_issuer,
        help="Optional JWT issuer claim",
    )
    parser.add_argument(
        "--audience",
        default=os.getenv("JWT_AUDIENCE") or settings.jwt_audience,
        help="Optional JWT audience claim",
    )
    parser.add_argument(
        "--subject",
        default="graph-core-admin",
        help="JWT subject claim",
    )
    parser.add_argument(
        "--expires-in-days",
        type=int,
        default=365,
        help="Token lifetime in days (default: 365)",
    )
    parser.add_argument(
        "--expires-in-minutes",
        type=int,
        default=None,
        help="Override token lifetime in minutes",
    )
    return parser.parse_args()


def build_token(args: argparse.Namespace) -> str:
    if not args.secret:
        raise SystemExit("JWT secret is required. Set JWT_SECRET or pass --secret.")

    now = datetime.now(timezone.utc)
    expires = (
        timedelta(minutes=args.expires_in_minutes)
        if args.expires_in_minutes is not None
        else timedelta(days=args.expires_in_days)
    )

    payload: dict[str, object] = {
        "sub": args.subject,
        "token_type": "admin",
        "scope": ADMIN_SCOPE,
        "iat": int(now.timestamp()),
        "exp": int((now + expires).timestamp()),
    }
    if args.issuer:
        payload["iss"] = args.issuer
    if args.audience:
        payload["aud"] = args.audience

    return jwt.encode(payload, args.secret, algorithm="HS256")


def main() -> None:
    args = parse_args()
    token = build_token(args)
    print(token)
    print()
    print(f"Authorization: Bearer {token}")


if __name__ == "__main__":
    main()
