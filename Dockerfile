FROM python:3.11-slim

# libgomp1 is required by LightGBM
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# exact pins from the universal lock (same versions as CI and local dev)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ src/
COPY models/ models/
COPY app/ app/
COPY deploy/seed.py deploy/seed.py

# SQLite and the NYISO fetch cache live on a mounted volume so they survive
# container restarts. data/ itself is dockerignored (no NYISO bytes in images).
ENV APP_DB_PATH=/data/app.db
ENV NYISO_CACHE_DIR=/data/cache
ENV WEATHER_PATH=/data/weather.csv
ENV WEATHER_HOURLY_PATH=/data/weather_hourly.csv
# the ONNX TFT bundle (tft.onnx + tft.json) is uploaded to the volume monthly
ENV TFT_BUNDLE_DIR=/data/models/tft
VOLUME /data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
