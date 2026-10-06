FROM node:24-bookworm-slim AS frontend
WORKDIR /web
COPY web/package*.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY prompts ./prompts
COPY data/annotations ./data/annotations
RUN pip install --no-cache-dir --no-deps -e .
COPY --from=frontend /web/dist ./web/dist
RUN useradd --uid 10001 --create-home reader && chown -R reader:reader /app
USER reader
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
