#!/usr/bin/env python3
"""
Daily RSI(14) + Trend scan (Alpaca) — logs symbols that match:
    RSI14(15m) < 30  AND  SMA60(15m) < SMA240(15m)

- Runs once per day (UTC) at RUN_AT_UTC (default 09:00).
- Scans a configurable asset universe from Alpaca.
- No trading, just logs results for Railway stdout.

Env vars:
  ALPACA_API_KEY / APCA_API_KEY_ID
  ALPACA_SECRET_KEY / APCA_API_SECRET_KEY
  APCA_API_BASE_URL (default https://api.alpaca.markets)
  DATA_FEED=iex|sip (default iex)
  UNIVERSE=us_equity|crypto|both (default us_equity)
  STATUS_FILTER=active|inactive|all (default active)
  INCLUDE_FRACTIONAL=true|false (default true)  [us_equity only]
  EXCH_LIST=comma,separated,exchanges  (optional; e.g. NYSE,NASDAQ,ARCA)
  BATCH_SIZE=int (default 50)
  HISTORY_DAYS_15M=int (default 14)
  RUN_AT_UTC=HH:MM (default 09:00)
  DRY_RUN=true|false (default true)  # no effect here (read-only), kept for symmetry
"""

import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

from alpaca_trade_api.rest import REST, TimeFrame

# ========= Config =========
DATA_FEED = os.getenv("DATA_FEED", "iex").lower()
UNIVERSE = os.getenv("UNIVERSE", "us_equity").lower()  # us_equity | crypto | both
STATUS_FILTER = os.getenv("STATUS_FILTER", "active").lower()  # active|inactive|all
INCLUDE_FRACTIONAL = os.getenv("INCLUDE_FRACTIONAL", "true").lower() == "true"
EXCH_LIST = [e.strip().upper() for e in os.getenv("EXCH_LIST", "").split(",") if e.strip()]

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
HISTORY_DAYS_15M = int(os.getenv("HISTORY_DAYS_15M", "14"))  # enough for 240+RSI warmup
RUN_AT_UTC = os.getenv("RUN_AT_UTC", "09:00")  # daily schedule (container stays alive)
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
APCA_API_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://api.alpaca.markets")

if not (ALPACA_API_KEY and ALPACA_SECRET_KEY):
    raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY (or APCA_* equivalents).")

api = REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, base_url=APCA_API_BASE_URL)

def log(msg: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    print(f"[daily-rsi15m-scan] {now} | {msg}", flush=True)

# ========= Indicators =========
def rsi(closes: List[float], length: int = 14) -> float:
    if len(closes) < length + 1:
        return float("nan")
    gains = losses = 0.0
    for i in range(1, length + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / length
    avg_loss = losses / length
    for i in range(length + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        gain = max(d, 0.0)
        loss = max(-d, 0.0)
        avg_gain = (avg_gain * (length - 1) + gain) / length
        avg_loss = (avg_loss * (length - 1) + loss) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def sma(closes: List[float], length: int) -> float:
    if len(closes) < length:
        return float("nan")
    return sum(closes[-length:]) / float(length)

# ========= Universe =========
def load_symbols() -> List[str]:
    status = None if STATUS_FILTER == "all" else STATUS_FILTER
    symbols: List[str] = []

    def want(asset) -> bool:
        if status and asset.status != status:
            return False
        if UNIVERSE in ("us_equity", "both") and asset.asset_class == "us_equity":
            if not asset.tradable:
                return False
            if not INCLUDE_FRACTIONAL and getattr(asset, "fractionable", False):
                return False
            if EXCH_LIST and (asset.exchange not in EXCH_LIST):
                return False
            return True
        if UNIVERSE in ("crypto", "both") and asset.asset_class == "crypto":
            if not asset.tradable:
                return False
            return True
        return False

    # get_assets returns all types; filter per settings
    assets = api.list_assets()
    for a in assets:
        if want(a):
            symbols.append(a.symbol)

    symbols = sorted(list(set(symbols)))
    log(f"Universe loaded: {len(symbols)} symbols (UNIVERSE={UNIVERSE}, STATUS={STATUS_FILTER}, FEED={DATA_FEED})")
    return symbols

# ========= Bars & Scan =========
def fetch_15m_bars(symbols: List[str], start: datetime, end: datetime):
    """
    Fetch 15m bars for a batch of symbols using the multi-symbol endpoint.
    Returns a pandas MultiIndex DataFrame in .df form (may be empty).
    """
    bars = api.get_bars(
        symbols,
        TimeFrame.Minute,  # we’ll request 15m via the '15Min' alias below
        start=start.isoformat(),
        end=end.isoformat(),
        adjustment="raw",
        feed=DATA_FEED,
        timeframe="15Min",  # explicit alias; alpaca_trade_api accepts this kw
        limit=None,
    )
    return getattr(bars, "df", None)

def last_closed_only(sym_df, bar_minutes: int = 15):
    """
    Given a single-symbol DataFrame (index=bar start time), drop the last row
    if it is a forming (not fully closed) bar.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=bar_minutes)
    if not sym_df.empty and sym_df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) > cutoff:
        return sym_df.iloc[:-1]
    return sym_df

def scan_batch(symbols: List[str], start: datetime, end: datetime) -> List[Tuple[str, float, float, float]]:
    """
    Returns list of (symbol, rsi14, sma60, sma240) that meet the condition.
    """
    df = fetch_15m_bars(symbols, start, end)
    if df is None or df.empty:
        log(f"Batch fetch: no data for {len(symbols)} symbols (holiday/closed/latency?)")
        return []

    matches = []
    # If MultiIndex, level 0 is symbol; else single symbol
    multi = isinstance(df.index, type(df.index)) and getattr(df.index, "nlevels", 1) > 1
    sym_groups = df.groupby(level=0) if multi else [(symbols[0], df)]

    for sym, sym_df in sym_groups:
        try:
            # Normalize to one symbol DataFrame
            if multi:
                s_df = sym_df.sort_index()
            else:
                s_df = sym_df.sort_index()

            s_df = last_closed_only(s_df, 15)
            if s_df.empty:
                continue

            closes = s_df["close"].astype(float).tolist()
            if len(closes) < (240 + 14):  # warmup: need ≥ 254 closes
                continue

            r = rsi(closes, 14)
            s60 = sma(closes, 60)
            s240 = sma(closes, 240)

            if (not math.isnan(r)) and (not math.isnan(s60)) and (not math.isnan(s240)):
                if (r < 30.0) and (s60 < s240):
                    matches.append((sym, r, s60, s240))
        except Exception as e:
            log(f"{sym} | scan error: {type(e).__name__}: {e}")

    return matches

def chunked(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def run_scan_once():
    start = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS_15M)
    end = datetime.now(timezone.utc)

    symbols = load_symbols()
    total = len(symbols)
    if total == 0:
        log("No symbols in universe; done.")
        return

    log(f"Scanning {total} symbols over ~{HISTORY_DAYS_15M} days of 15m bars…")
    all_matches: List[Tuple[str, float, float, float]] = []

    processed = 0
    for batch in chunked(symbols, BATCH_SIZE):
        matches = scan_batch(batch, start, end)
        all_matches.extend(matches)
        processed += len(batch)
        if processed % (BATCH_SIZE*4) == 0 or processed == total:
            log(f"Progress: {processed}/{total} symbols")

        # Small pacing delay to be gentle on API
        time.sleep(0.5)

    # Sort matches by RSI ascending (most oversold first)
    all_matches.sort(key=lambda t: t[1])

    log("===== DAILY SCAN RESULTS =====")
    log(f"Matches: {len(all_matches)}")
    for sym, r, s60, s240 in all_matches:
        log(f"{sym} | RSI14={r:.2f} | SMA60={s60:.4f} < SMA240={s240:.4f}")

    if not all_matches:
        log("No symbols met RSI14<30 and SMA60(15m)<SMA240(15m) today.")

def seconds_until_next_daily_run(target_hhmm: str) -> float:
    """Compute seconds until next daily target (UTC 'HH:MM')."""
    try:
        hh, mm = [int(x) for x in target_hhmm.split(":")]
    except Exception:
        hh, mm = 9, 0  # fallback 09:00
    now = datetime.now(timezone.utc)
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(1.0, (target - now).total_seconds())

def main():
    log(f"Starting daily RSI+trend scanner | base={APCA_API_BASE_URL} | feed={DATA_FEED} | universe={UNIVERSE}")
    # run immediately on boot, then every day at RUN_AT_UTC
    run_scan_once()
    while True:
        delay = seconds_until_next_daily_run(RUN_AT_UTC)
        log(f"Sleeping {int(delay)}s until next daily run at {RUN_AT_UTC} UTC…")
        time.sleep(delay)
        run_scan_once()

if __name__ == "__main__":
    main()
