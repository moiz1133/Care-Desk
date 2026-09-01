"""FastAPI dependency providers.

Settings are wired through `Depends()` rather than imported as a
module-level global, so routes (and tests, via
`app.dependency_overrides`) get them injected per-request instead of
reaching for a hardcoded singleton.

There is deliberately no separate DB-session or OpenAI-client dependency
here. `retrieve()` (caredesk.retrieval.vector) and `generate_answer()`
(caredesk.generation.generator) already own their own per-call resource
lifecycle -- a scoped `session_scope` session and an `AsyncOpenAI` client
respectively -- constructed from the `Settings` passed into them. Standing
up a second, route-level session/client that nothing consumes would just
be two places managing the same lifecycle; the `Settings` dependency below
is the actual seam.
"""

from __future__ import annotations

from caredesk.config import Settings, get_settings


def get_settings_dependency() -> Settings:
    """FastAPI dependency wrapper around the cached `Settings` singleton."""
    return get_settings()
