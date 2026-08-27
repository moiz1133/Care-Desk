.PHONY: install dev ui test lint typecheck db-up db-down db-migrate index

install:
	uv sync

dev:
	uv run uvicorn caredesk.api.main:app --reload

ui:
	uv run streamlit run ui/ask.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

db-up:
	docker compose up -d

db-down:
	docker compose down

db-migrate:
	uv run alembic upgrade head

index:
	uv run python scripts/index_corpus.py --dry-run
