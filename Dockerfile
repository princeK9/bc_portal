# Single image for both the API and the worker. They run different commands but
# must share one extraction code path -- separate images drift, and the whole
# point of extraction.py is that both paths produce identical records.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# psycopg2-binary ships wheels, but torch/sentence-transformers still want a
# compiler on some platforms; curl is here for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so a code edit does not invalidate the (very large) torch layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the MiniLM weights into the image. Without this every worker downloads
# ~90MB on first task, which turns a cold start into a timeout and makes the
# first extraction in each container mysteriously slow.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/spool \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
