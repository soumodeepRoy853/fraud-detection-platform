FROM python:3.12-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements-api.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements-api.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]