"""Platform configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://graphcore:graphcore@localhost:5432/graphcore"
    redis_url: str = "redis://localhost:6379/0"
    falkordb_url: str = "falkordb://localhost:6379"

    # Encryption key for credential storage (32 bytes, base64 or hex encoded)
    credential_encryption_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # Defaults for new collections
    default_embedding_provider: str = "local_hash"
    default_embedding_model: str = "hash-256"
    default_embedding_dimensions: int = 256
    default_distance_metric: str = "cosine"

    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 40
    vector_query_top_k: int = 5

    default_llm_provider: str = "local_echo"
    default_llm_model: str = "echo-v1"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
