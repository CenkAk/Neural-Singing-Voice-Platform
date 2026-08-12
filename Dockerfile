FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY apps/api ./apps/api
RUN pip install --no-cache-dir ".[audio,api,training,deployment]"
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "nsvp.api:app", "--host", "0.0.0.0", "--port", "8000"]
