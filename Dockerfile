# Use Python 3.11 to avoid alpaca-trade-api/aiohttp build issues on 3.12
FROM python:3.11-slim

# System deps (helpful for scientific wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Speed up pip a bit
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app
# If your file is named daily_scan_rsi15_sma.py, we run it as app.py
COPY daily_scan_rsi15_sma.py ./app.py

# Default envs (override in Railway)
ENV DATA_FEED=iex \
    UNIVERSE=us_equity \
    STATUS_FILTER=active \
    INCLUDE_FRACTIONAL=true \
    BATCH_SIZE=50 \
    HISTORY_DAYS_15M=14 \
    RUN_AT_UTC=09:00

# Run the scanner (self-schedules once per day)
CMD ["python", "app.py"]
