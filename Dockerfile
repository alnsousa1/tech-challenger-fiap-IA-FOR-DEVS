FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app/src

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .
RUN python scripts/train.py

EXPOSE 8000

CMD ["uvicorn", "womens_health_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
