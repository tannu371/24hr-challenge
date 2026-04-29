FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /repo

# System deps: matplotlib backends + cvxpy / numpy build helpers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential libfreetype6-dev pkg-config git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /repo/requirements.txt
RUN pip install -r /repo/requirements.txt

# The backend imports the repo-level `portfolio/` package, so we copy the
# whole repo. The .dockerignore excludes node_modules / venv / etc.
COPY . /repo

WORKDIR /repo/backend

EXPOSE 8765

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
