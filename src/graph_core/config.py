"""Platform configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://graphcore:graphcore@localhost:5432/graphcore"
    redis_url: str = "redis://localhost:6379/0"
    falkordb_url: str = "falkordb://localhost:6379"
    redis_semaphore_url: str = "redis://localhost:6380/0"
    llm_max_concurrent_calls: int = 1
    embedding_max_concurrent_calls: int = 10
    provider_semaphore_lease_seconds: int = 1800
    provider_semaphore_poll_interval_ms: int = 100
    provider_semaphore_acquire_timeout_seconds: float = 180
    ingest_chunk_time_limit_ms: int = 7200000
    ingest_chunk_max_age_ms: int = 28800000

    # Encryption key for credential storage (32 bytes, base64 or hex encoded)
    credential_encryption_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str | None = None

    # Deployment mode: "self_hosted" (default) or "multi_tenant"
    platform_mode: str = "self_hosted"

    # Admin key for platform management (create namespaces, register apps)
    # Supports dual-key rotation via platform_admin_key_secondary
    platform_admin_key: str = ""
    platform_admin_key_secondary: str = ""

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

    # Graph RAG settings
    falkordb_graph_name: str = "knowledge_graph"
    graph_rag_max_concurrent_workers: int = 5
    graph_rag_query_embedding_concurrency: int = 4
    sqlalchemy_pool_size: int = 10
    sqlalchemy_max_overflow: int = 20
    sqlalchemy_pool_timeout: float = 30
    graph_rag_high_confidence_threshold: float = 0.3
    graph_rag_medium_confidence_threshold: float = 0.7
    graph_rag_fuzzy_name_threshold: float = 0.8
    graph_rag_description_similarity_threshold: float = 0.90
    graph_rag_max_relationship_weight: int = 100
    graph_rag_min_edge_similarity: float = 0.3
    graph_rag_edge_weight_score_ratio: float = 0.3
    graph_rag_keyword_score_ratio: float = 0.2
    graph_rag_active_dimensions: list[str] = []  # empty = all
    graph_rag_dimension_weights: dict[str, float] = {}  # rel_type -> weight

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
