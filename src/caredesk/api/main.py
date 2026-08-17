"""FastAPI application factory.

Exposes `create_app()` which builds and returns the FastAPI instance.
Currently wires up only a `/health` endpoint and a stubbed lifespan
handler; routers for ingestion, retrieval, and conversation endpoints
are added in later commits.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook.

    Will eventually manage DB connection pools, Redis clients, and the
    Langfuse client. No-op for now.
    """
    yield


def create_app() -> FastAPI:
    """Build and return the CareDesk FastAPI application."""
    app = FastAPI(title="CareDesk", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
