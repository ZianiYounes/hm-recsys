# Image unique pour l'API (defaut) et la demo Streamlit (voir docker-compose.yml).
# Les donnees et artefacts sont montes en volume, l'image ne contient que le code.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RECSYS_ROOT=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-serving.txt

COPY pyproject.toml .
COPY src ./src
COPY configs ./configs
COPY app ./app
RUN pip install --no-deps -e .

EXPOSE 8000 8501
CMD ["uvicorn", "recsys.serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
