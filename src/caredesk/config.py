"""Application configuration.

All runtime configuration flows through the `Settings` object defined here,
loaded from environment variables / a local `.env` file via
pydantic-settings. No module outside this file should call `os.getenv`
directly — import and depend on `Settings` (or the `get_settings` accessor)
instead, so configuration stays centralized and testable.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the CareDesk service.

    No default is provided for a setting that would be unsafe to run with
    a placeholder value (credentials, API keys). Everything else has a
    sensible local-development default.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data stores
    database_url: str = Field(
        default="postgresql+psycopg://caredesk:caredesk@localhost:5432/caredesk",
        description="Postgres (pgvector-enabled) connection string.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string, used for the semantic cache.",
    )

    # Corpus
    corpus_root: Path = Field(
        default=Path("data/corpus"),
        description="Root directory containing the document corpus and its manifest.json.",
    )

    # OpenAI
    openai_api_key: str = Field(
        description="OpenAI API key. Required, no default.",
    )

    # Langfuse
    # Optional, unlike openai_api_key above: commit 8 requires the app to
    # start and serve requests with tracing in no-op mode when these are
    # absent, rather than failing Settings() construction outright.
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key. Tracing runs in no-op mode when absent.",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Langfuse secret key. Tracing runs in no-op mode when absent.",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse ingestion host.",
    )
    langfuse_enabled: bool = Field(
        default=True,
        description="Master switch for tracing. False forces no-op mode even with credentials set.",
    )
    trace_sample_rate: float = Field(
        default=1.0,
        description="Fraction of traces recorded in full detail, 0.0-1.0. Errors and refusals "
        "are always recorded regardless -- see observability/tracing.py.",
    )
    trace_flush_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for flushing buffered spans to Langfuse on application shutdown.",
    )
    environment: str = Field(
        default="dev",
        description="Deployment environment tag attached to every trace (dev/staging/prod).",
    )

    # Models
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model used for ingestion and retrieval.",
    )
    classifier_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for query classification / routing decisions.",
    )
    generator_model: str = Field(
        default="gpt-4o",
        description="Model used for grounded answer generation.",
    )

    # Chunking
    chunk_size: int = Field(
        default=512,
        description="Target chunk size in tokens for the fixed-size chunker.",
    )
    chunk_overlap: int = Field(
        default=50,
        description="Token overlap between consecutive chunks in the sliding window.",
    )
    chunk_min_tokens: int = Field(
        default=20,
        description="Minimum token count for a trailing chunk; shorter ones are dropped.",
    )

    # Embedding
    embedding_dimension: int = Field(
        default=1536,
        description="Expected embedding vector dimension; must match embedding_model.",
    )
    embedding_batch_size: int = Field(
        default=100,
        description="Chunks per embedding API call.",
    )
    embedding_max_concurrency: int = Field(
        default=4,
        description="Maximum concurrent in-flight embedding batch calls.",
    )
    embedding_max_retries: int = Field(
        default=5,
        description="Max retry attempts for a rate-limited or transient embedding call.",
    )
    embedding_retry_base_delay_seconds: float = Field(
        default=1.0,
        description="Base delay for exponential backoff between embedding call retries.",
    )
    embedding_cost_ceiling_usd: float = Field(
        default=5.0,
        description="Refuse to run indexing outright if the estimated cost exceeds this.",
    )
    embedding_price_per_million_tokens_usd: float = Field(
        default=0.02,
        description="Embedding model price per 1M tokens, used only for cost estimates.",
    )

    # Indexing
    hnsw_m: int = Field(
        default=16,
        description="pgvector HNSW index 'm' parameter (max connections per layer).",
    )
    hnsw_ef_construction: int = Field(
        default=64,
        description="pgvector HNSW index 'ef_construction' parameter (build-time search width).",
    )
    test_database_url: str | None = Field(
        default=None,
        description="Postgres URL for the indexer's DB-backed test suite. Never the dev database.",
    )

    # Retrieval
    retrieval_top_k: int = Field(
        default=20,
        description="Number of candidates pulled from vector + keyword search prior to reranking.",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of reranked candidates passed to generation.",
    )
    vector_retrieval_k: int = Field(
        default=5,
        description="Default number of results returned by the vector-only baseline retriever.",
    )
    query_embedding_cache_size: int = Field(
        default=1000,
        description="Max entries in the in-process exact-match query embedding cache.",
    )

    # Generation
    min_relevance_score: float = Field(
        default=0.3,
        description="Refuse without calling the model if the top retrieval score is below this.",
    )
    max_answer_tokens: int = Field(
        default=500,
        description="Max tokens the generator model may produce for one answer.",
    )
    generation_temperature: float = Field(
        default=0.0,
        description="Generator sampling temperature. 0 for deterministic, eval-stable output.",
    )
    generator_input_price_per_million_tokens_usd: float = Field(
        default=2.50,
        description="Generator model input token price, used only for cost estimates.",
    )
    generator_output_price_per_million_tokens_usd: float = Field(
        default=10.00,
        description="Generator model output token price, used only for cost estimates.",
    )

    # API
    api_request_timeout_seconds: float = Field(
        default=30.0,
        description=(
            "Outer bound for a /query request. Model calls have their own "
            "timeouts already; this catches everything else that could hang."
        ),
    )
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description=(
            "Permissive by default for local development. Must be tightened to "
            "an explicit allowlist before any deployment outside a developer's "
            "own machine."
        ),
    )

    # Decision thresholds
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence required to resolve a query without escalation.",
    )
    max_cost_per_conversation_usd: float = Field(
        default=0.50,
        description="Soft budget ceiling per conversation, used to trigger cost-aware routing.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance."""
    return Settings()  # type: ignore[call-arg]
