"""Database session management.

Provides the SQLAlchemy engine and session factory, configured from
`Settings.database_url`. Engines are cached per URL so repeated calls
within a process reuse the same connection pool.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from caredesk.config import Settings


@lru_cache(maxsize=8)
def get_engine(database_url: str) -> Engine:
    """Return a cached SQLAlchemy engine for `database_url`.

    A short connect_timeout ensures an unreachable Postgres (e.g. not yet
    started via `make db-up`) fails fast with a clear error instead of
    hanging — important since `--dry-run` is meant to be safe to run first.
    """
    return create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def get_session_factory(settings: Settings) -> sessionmaker[Session]:
    """Return a session factory bound to `settings.database_url`."""
    return sessionmaker(bind=get_engine(settings.database_url), expire_on_commit=False)


@contextmanager
def session_scope(settings: Settings) -> Iterator[Session]:
    """Yield a `Session` that commits on success and rolls back on error.

    Callers are expected to use one `session_scope` per unit of work (the
    indexer uses one per document) so a failure part-way through leaves the
    database consistent rather than half-written.
    """
    session = get_session_factory(settings)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
