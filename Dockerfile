FROM ncbi/amr:4.2.7-2026-05-15.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    GENOME_FIREWALL_AMRFINDER_EXECUTION=direct \
    GENOME_FIREWALL_EPHEMERAL_INFERENCE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app /opt/venv /opt/uv/python \
    && chown -R user:user /app /home/user

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --python 3.12

COPY --chown=user:user . .
RUN uv sync --frozen --no-dev --python 3.12 \
    && mkdir -p data/amrfinder/inference \
    && chown -R user:user data/amrfinder

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail http://127.0.0.1:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true", "--browser.gatherUsageStats=false"]
