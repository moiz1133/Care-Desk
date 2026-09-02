"""Per-request context establishment: request_id, RequestContext, request logging.

This is the one place `request_id` is generated and the one place
`RequestContext` gets created -- everything downstream (tracing,
`retrieve()`, `generate_answer()`) reads it; nothing else constructs its
own. Persona and conversation_id aren't known yet when this middleware
runs -- they live in the request body, which FastAPI only parses once
routing/dependency resolution starts inside `call_next` -- so
`set_query_identity` lets the route enrich the *same* bound
RequestContext once the body validates, rather than this middleware
guessing or the route creating a second one.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request, Response

from caredesk.config import get_settings
from caredesk.observability.context import (
    RequestContext,
    bind_request_context,
    get_current_request_context,
)
from caredesk.observability.vocabulary import ClientType

logger = logging.getLogger(__name__)

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_log_record_factory_installed = False


def install_request_id_log_factory() -> None:
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


def _resolve_client(request: Request) -> str:
    """`X-Client` header, defaulting to "api".

    An unrecognised value falls back to "api" rather than being passed
    through raw: `client` is a closed-set trace tag (see
    `observability.vocabulary.ClientType`), and a typo'd header value
    must not silently become a new, unfilterable tag.
    """
    raw = request.headers.get("X-Client", ClientType.API.value).strip().lower()
    try:
        return str(ClientType(raw))
    except ValueError:
        logger.warning("unknown_client_header", extra={"value": raw})
        return str(ClientType.API)


def set_query_identity(*, persona: str, conversation_id: str) -> None:
    """Enrich the already-bound RequestContext with persona/conversation_id.

    Called once, from the /query route, immediately after the request
    body validates -- the earliest point those two values exist. Mutates
    the RequestContext this module's middleware already bound; does not
    create a new one. If no RequestContext is bound (shouldn't happen for
    a real HTTP request, but this also runs the same code path scripts
    could import directly), logs a warning and does nothing --
    `tracing.start_query_trace` falls back to placeholder identity rather
    than raising when that happens.
    """
    context = get_current_request_context()
    if context is None:
        logger.warning("set_query_identity_without_bound_context")
        return
    context.persona = persona
    context.conversation_id = conversation_id


async def request_context_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Establish request_id and RequestContext, bind them, and log the outcome.

    Runs for every request regardless of route. Exceptions raised inside a
    route are converted into responses by the exception handlers
    registered in `api.main`'s `create_app` *before* `call_next` returns
    here, so this always sees a real `Response` and always gets to attach
    the header and log.
    """
    settings = get_settings()
    request_id = str(uuid4())
    request.state.request_id = request_id
    token = _request_id_var.set(request_id)

    context = RequestContext(
        request_id=request_id,
        environment=settings.environment,
        client=_resolve_client(request),
    )

    start = time.monotonic()
    try:
        with bind_request_context(context):
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
