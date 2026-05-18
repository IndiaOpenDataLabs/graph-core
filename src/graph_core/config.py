"""Platform configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://graphcore:graphcore@localhost:5432/graphcore"
    redis_url: str = "redis://localhost:6379/0"

    # Encryption key for credential storage (32 bytes, base64 or hex encoded)
    credential_encryption_key: str = ""

    # Defaults for new collections
    default_embedding_provider: str = "openai"
    default_embedding_model: str = "text-embedding-3-large"
    default_embedding_dimensions: int = 3072

    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
