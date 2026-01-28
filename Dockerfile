FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen

COPY src ./src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api_transformer.server:app", "--host", "0.0.0.0", "--port", "8000"]
