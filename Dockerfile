FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy

# The lockfile includes the API dependencies, spaCy model, and CPU PyTorch.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY app ./app
COPY models ./models
RUN test -s models/cifar10_cnn.pt && test -s models/mnist_gan_generator.pt

EXPOSE 80

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
