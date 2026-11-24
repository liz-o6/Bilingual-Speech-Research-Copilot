FROM python:3.13-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOME=/home/app \
    ANONYMIZED_TELEMETRY=False
RUN groupadd --gid 10001 app && \
    useradd --uid 10001 --gid app --create-home app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ src/
COPY --chown=app:app frontend/ frontend/
COPY --chown=app:app data/corpus/manifest.jsonl data/corpus/manifest.jsonl
COPY --chown=app:app data/corpus/catalog.json data/corpus/catalog.json
COPY --chown=app:app data/corpus/raw/ data/corpus/raw/
# Chroma may write SQLite metadata; each instance has its own disposable copy.
COPY --chown=app:app storage/chroma/ storage/chroma/
COPY --chown=app:app deploy/start.sh deploy/start.sh
COPY --chown=app:app deploy/check_image.py deploy/check_image.py
USER app
RUN python deploy/check_image.py
EXPOSE 8080
CMD ["sh", "/app/deploy/start.sh"]
