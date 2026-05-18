"""Credential encryption helpers."""

import base64
import hashlib

from cryptography.fernet import Fernet

from graph_core.config import settings


class CredentialCrypto:
    def __init__(self, seed: str | None = None):
        seed = seed or settings.credential_encryption_key or "graph-core-dev-key"
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(digest))

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def decrypt(self, value: str) -> str:
        return self._fernet.decrypt(value.encode("utf-8")).decode("utf-8")
