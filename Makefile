.PHONY: sync lint format test demo ui

sync:
	uv sync --all-extras --dev --locked

lint:
	uv run --locked ruff check app.py src tests

format:
	uv run --locked ruff check --fix app.py src tests
	uv run --locked ruff format app.py src tests

test:
	uv run --locked pytest -q

demo:
	uv run --locked genome-firewall demo

ui:
	uv run --locked streamlit run app.py
