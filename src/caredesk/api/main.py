"""FastAPI application factory.

Builds the CareDesk FastAPI app: the `/health` endpoint (commit 1) and the
`/query` endpoint (commit 7, wired through `services.pipeline`), plus the
cross-cutting concerns every request goes through regardless of route --
CORS, request-id/logging middleware, and exception-to-status-code mapping.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from caredesk.api.routes.query import router as query_router
from caredesk.config import get_settings
from caredesk.generation.generator import GeneratorError
from caredesk.ingestion.embedder import EmbedderError

logger = logging.getLogger(__name__)

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_log_record_factory_installed = False


def _install_request_id_log_factory() -> None:
    """Make every LogRecord, from any module's logger, carry the active request_id.

    A `logging.Filter` attached to one logger only applies to records that
    logger itself originates, not to records from other loggers that
    happen to propagate through the same handler -- which would miss
    almost every log call in this codebase (each module logs through its
    own `logging.getLogger(__name__)`). Wrapping the global LogRecord
    factory once, here, is what actually reaches all of them.
    """
    global _log_record_factory_installed
    if _log_record_factory_installed:
        return
    previous_factory = logging.getLogRecordFactory()

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        record.request_id = _request_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
    _log_record_factory_installed = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook.

    Will eventually manage DB connection pools, Redis clients, and the
    Langfuse client. No-op for now besides installing log correlation.
    """
    _install_request_id_log_factory()
    yield


async def _request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Assign a request_id, bind it for logging, and log the outcome.

    Runs for every request regardless of route. Exceptions raised inside a
    route are converted into responses by the exception handlers registered
    in `create_app` *before* `call_next` returns here, so this always sees
    a real `Response` and always gets to attach the header and log.
    """
    request_id = str(uuid4())
    request.state.request_id = request_id
    token = _request_id_var.set(request_id)
    start = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        _request_id_var.reset(token)
    duration_ms = (time.monotonic() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response


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
    app.middleware("http")(_request_context_middleware)

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
