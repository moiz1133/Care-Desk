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
    langfuse_public_key: str = Field(
        description="Langfuse public key. Required, no default.",
    )
    langfuse_secret_key: str = Field(
        description="Langfuse secret key. Required, no default.",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse ingestion host.",
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

    # Retrieval
    retrieval_top_k: int = Field(
        default=20,
        description="Number of candidates pulled from vector + keyword search prior to reranking.",
    )
    rerank_top_k: int = Field(
        default=5,
        description="Number of reranked candidates passed to generation.",
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
