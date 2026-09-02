"""FastAPI application factory.

Builds the CareDesk FastAPI app: the `/health` endpoint (commit 1) and the
`/query` endpoint (commit 7, wired through `services.pipeline`), plus the
cross-cutting concerns every request goes through regardless of route --
CORS, exception-to-status-code mapping, and the per-request context
middleware (request_id/RequestContext establishment, request logging)
that actually lives in `api.middleware`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from caredesk.api.middleware import install_request_id_log_factory, request_context_middleware
from caredesk.api.routes.query import router as query_router
from caredesk.config import get_settings
from caredesk.generation.generator import GeneratorError
from caredesk.ingestion.embedder import EmbedderError
from caredesk.observability import tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook.

    Will eventually manage DB connection pools and Redis clients too. On
    shutdown, flushes any buffered Langfuse spans -- bounded by
    `Settings.trace_flush_timeout_seconds` so a slow/unreachable Langfuse
    backend can't hang process shutdown; see observability/tracing.py.
    """
    install_request_id_log_factory()
    yield
    await tracing.shutdown(get_settings())


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "-"))


def _error_response(
    request: Request, status_code: int, message: str, *, retry_after: int | None = None
) -> JSONResponse:
    """Build a client-facing error body that never carries provider/stack-trace detail.

    Full detail belongs in the server-side log line for the same
    request_id, not in the response.
    """
    response = JSONResponse(
        status_code=status_code,
        content={"detail": message, "request_id": _request_id(request)},
    )
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


def create_app() -> FastAPI:
    """Build and return the CareDesk FastAPI application."""
    settings = get_settings()
    app = FastAPI(title="CareDesk", lifespan=lifespan)

    # Permissive for local development only. settings.cors_allow_origins
    # must be tightened to an explicit allowlist before any deployment
    # outside a developer's own machine.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(request_context_middleware)

    @app.exception_handler(TimeoutError)
    async def _timeout_handler(request: Request, exc: TimeoutError) -> JSONResponse:
        logger.error("request_timeout", extra={"request_id": _request_id(request)})
        return _error_response(
            request, 503, "The request took too long to complete. Please retry.", retry_after=5
        )

    @app.exception_handler(GeneratorError)
    async def _generator_error_handler(request: Request, exc: GeneratorError) -> JSONResponse:
        logger.error(
            "generator_failure", extra={"request_id": _request_id(request), "error": str(exc)}
        )
        return _error_response(
            request,
            503,
            "The answer generation service is temporarily unavailable. Please retry.",
            retry_after=5,
        )

    @app.exception_handler(EmbedderError)
    async def _embedder_error_handler(request: Request, exc: EmbedderError) -> JSONResponse:
        logger.error(
            "embedder_failure", extra={"request_id": _request_id(request), "error": str(exc)}
        )
        return _error_response(
            request,
            503,
            "The embedding service is temporarily unavailable. Please retry.",
            retry_after=5,
        )

    @app.exception_handler(OperationalError)
    async def _db_unavailable_handler(request: Request, exc: OperationalError) -> JSONResponse:
        logger.error(
            "database_unavailable", extra={"request_id": _request_id(request), "error": str(exc)}
        )
        return _error_response(
            request, 503, "The database is temporarily unavailable. Please retry.", retry_after=5
        )

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"request_id": _request_id(request)})
        return _error_response(request, 500, "An unexpected error occurred.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(query_router)

    return app


app = create_app()
