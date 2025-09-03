FROM python:3.12-slim

# System deps (just in case pandas/numpy need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY daily_scan_rsi15_sma.py ./app.py

# Default envs (override in Railway)
ENV DATA_FEED=iex \
    UNIVERSE=us_equity \
    STATUS_FILTER=active \
    INCLUDE_FRACTIONAL=true \
    BATCH_SIZE=50 \
    HISTORY_DAYS_15M=14 \
    RUN_AT_UTC=09:00

# Run
CMD ["python", "app.py"]
