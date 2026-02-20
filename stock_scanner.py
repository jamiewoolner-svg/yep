#!/usr/bin/env python3
"""Simple stock scanner CLI.

Fetches historical daily candles from Stooq and applies technical/fundamental-like
filters to find candidate stocks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence, Tuple


STOOQ_URL = "https://stooq.com/q/d/l/"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
POLYGON_AGGS_URL = "https://api.polygon.io/v2/aggs/ticker"
SP500_CONSTITUENTS_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
NASDAQ100_COMPANIES_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
WIKIPEDIA_NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
NASDAQ100_FALLBACK_CSV_URL = "https://raw.githubusercontent.com/Gary-Strauss/NASDAQ100_Constituents/master/data/nasdaq100_constituents.csv"
POWS_BB_LENGTH = 21
POWS_BB_STDDEV = 2.0
POWS_SMA_FAST = 50
POWS_SMA_MID = 89
POWS_SMA_SLOW = 200
POWS_RSI_LENGTH = 13
POWS_STOCH_PERIOD = 21
POWS_STOCH_SMOOTH_K = 3
POWS_STOCH_SMOOTH_D = 5
POWS_MACD_FAST = 8
POWS_MACD_SLOW = 21
POWS_MACD_SIGNAL = 5
POWS_DMI_LENGTH = 5
POWS_ADX_SMOOTHING = 13
POWS_STOCH_OB = 80.0
POWS_STOCH_OS = 20.0
HISTORICAL_LOOKBACK_YEARS = 5

DATA_SOURCE = os.getenv("SCANNER_DATA_SOURCE", "auto")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_MIN_GAP_SEC = float(os.getenv("POLYGON_MIN_GAP_SEC", "0.25"))
_POLYGON_LOCK = threading.Lock()
_POLYGON_NEXT_ALLOWED_TS = 0.0


@dataclass
class Candle:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ScanResult:
    symbol: str
    close: float
    sma20: float
    sma50: float
    sma89: float
    sma200: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    rsi14: float
    stoch_rsi_k: float
    stoch_rsi_d: float
    macd: float
    macd_signal: float
    macd_gap_now: float
    macd_gap_prev: float
    stoch_gap_now: float
    stoch_gap_prev: float
    adx14: float
    plus_di14: float
    minus_di14: float
    avg_volume20: float
    dollar_volume20: float
    breakout_20d: bool
    price_cross_age: int
    price_bear_cross_age: int
    macd_cross_age: int
    stoch_cross_age: int
    dual_cross_gap: int
    macd_bear_cross_age: int
    stoch_bear_cross_age: int
    dual_bear_cross_gap: int
    triple_cross_gap: int
    triple_bear_cross_gap: int
    pre3x_bull_score: float
    pre3x_bear_score: float
    hist_setups_5y: int
    hist_win_rate_5y: float
    hist_avg_return_5y: float
    setup_direction: str
    setup_type: str
    target_mid_pct: float
    target_band_pct: float
    risk_pct: float
    rr_mid: float
    options_setup_score: float
    course_pattern_score: float
    bb_compression_score: float
    touched_outer_band_recent: bool
    outer_touch_age: int
    band_width_expansion: float
    band_widen_start_age: int
    band_widen_window_ok: bool
    liftoff_from_band: bool
    rejection_from_band: bool
    timeframe: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan US stocks using simple rules.")
    parser.add_argument(
        "--symbols",
        help="Comma-separated symbols (e.g. AAPL,MSFT,TSLA).",
    )
    parser.add_argument(
        "--symbols-file",
        default="data/default_symbols.txt",
        help="Path to newline-delimited symbols list.",
    )
    parser.add_argument(
        "--sp500",
        action="store_true",
        help="Scan the full S&P 500 ticker universe.",
    )
    parser.add_argument(
        "--qqq",
        action="store_true",
        help="Scan the Nasdaq-100 (QQQ) ticker universe.",
    )
    parser.add_argument(
        "--data-source",
        choices=["auto", "polygon", "stooq", "yahoo"],
        default=os.getenv("SCANNER_DATA_SOURCE", "auto"),
        help="Market data source.",
    )
    parser.add_argument(
        "--polygon-api-key",
        default=os.getenv("POLYGON_API_KEY", ""),
        help="Polygon API key. Can also be set via POLYGON_API_KEY env var.",
    )
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--max-price", type=float, default=1_000.0)
    parser.add_argument(
        "--min-dollar-volume",
        type=float,
        default=2_000_000.0,
        help="Minimum 20-day average dollar volume (price * volume).",
    )
    parser.add_argument(
        "--max-rsi",
        type=float,
        default=80.0,
        help="Upper RSI(14) threshold. Lower values bias toward pullbacks.",
    )
    parser.add_argument(
        "--max-stoch-rsi-k",
        type=float,
        default=95.0,
        help="Upper bound for StochRSI %%K.",
    )
    parser.add_argument(
        "--min-adx",
        type=float,
        default=10.0,
        help="Minimum ADX(14) for trend strength.",
    )
    parser.add_argument(
        "--require-uptrend",
        action="store_true",
        help="Require close > SMA20 > SMA50.",
    )
    parser.add_argument(
        "--require-breakout",
        action="store_true",
        help="Require close >= highest high of previous 20 sessions.",
    )
    parser.add_argument(
        "--require-macd-bull",
        action="store_true",
        help="Require MACD line above signal line.",
    )
    parser.add_argument(
        "--require-di-bull",
        action="store_true",
        help="Require +DI above -DI.",
    )
    parser.add_argument(
        "--require-macd-stoch-cross",
        action="store_true",
        help="Require directional MACD and StochRSI crosses within lookback.",
    )
    parser.add_argument(
        "--require-simultaneous-cross",
        action="store_true",
        help="Require directional MACD and StochRSI crosses on near-same bar.",
    )
    parser.add_argument(
        "--require-band-liftoff",
        action="store_true",
        help="Require outer Bollinger touch + widening bands + directional rejection/lift-off.",
    )
    parser.add_argument(
        "--bb-spread-watchlist",
        action="store_true",
        help="Allow pre-liftoff Bollinger expansion candidates likely to mature into directional crosses.",
    )
    parser.add_argument(
        "--signal-direction",
        choices=["bull", "bear", "both"],
        default="both",
        help="Scan for bullish, bearish, or either directional setups.",
    )
    parser.add_argument(
        "--cross-lookback",
        type=int,
        default=4,
        help="Bars back allowed for directional MACD/Stoch crosses.",
    )
    parser.add_argument(
        "--max-macd-cross-age",
        type=int,
        default=3,
        help="Require MACD cross within this many bars, or imminently approaching within this window.",
    )
    parser.add_argument(
        "--max-stoch-cross-age",
        type=int,
        default=3,
        help="Require StochRSI cross within this many bars, or imminently approaching within this window.",
    )
    parser.add_argument(
        "--band-touch-lookback",
        type=int,
        default=6,
        help="Bars back to detect outer Bollinger touch before lift-off.",
    )
    parser.add_argument(
        "--min-band-expansion",
        type=float,
        default=0.05,
        help="Minimum fractional BB width expansion vs recent baseline.",
    )
    parser.add_argument(
        "--min-course-pattern-score",
        type=float,
        default=55.0,
        help="Minimum class-rules pattern score (0-100). Higher = stricter.",
    )
    parser.add_argument(
        "--max-setup-age",
        type=int,
        default=3,
        help="Maximum age in daily bars for surfaced setups (0 disables age cap).",
    )
    parser.add_argument(
        "--require-daily-and-233",
        action="store_true",
        help="Require setup confirmation on both Daily and 233-minute charts.",
    )
    parser.add_argument(
        "--intraday-interval-min",
        type=int,
        default=5,
        help="Intraday source interval in minutes for building 233-minute candles.",
    )
    parser.add_argument(
        "--pows",
        action="store_true",
        help="Enable bundled POWS-style scan preset using BB/SMA/StochRSI/MACD/ADX.",
    )
    parser.add_argument(
        "--plot-symbol",
        help="Plot one symbol with indicator crossing markers (e.g. AAPL).",
    )
    parser.add_argument(
        "--plot-days",
        type=int,
        default=252,
        help="Number of most recent candles to render in plot mode.",
    )
    parser.add_argument(
        "--plot-output",
        help="Optional image output path (e.g. chart.png). If omitted, opens interactive window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show scan progress and fetch warnings.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent symbol workers for scans.",
    )
    parser.add_argument("--top", type=int, default=20, help="Max rows to print.")
    return parser.parse_args()


def load_symbols(symbols_arg: str | None, symbols_file: str) -> List[str]:
    if symbols_arg:
        symbols = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    else:
        with open(symbols_file, "r", encoding="utf-8") as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    deduped = []
    seen = set()
    for sym in symbols:
        if sym not in seen:
            deduped.append(sym)
            seen.add(sym)
    return deduped


def configure_data_source(data_source: str = "auto", polygon_api_key: str = "") -> None:
    global DATA_SOURCE, POLYGON_API_KEY
    DATA_SOURCE = data_source
    POLYGON_API_KEY = polygon_api_key or os.getenv("POLYGON_API_KEY", "")


def _dedupe_symbols(symbols: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for symbol in symbols:
        if symbol not in seen:
            deduped.append(symbol)
            seen.add(symbol)
    return deduped


def _parse_symbols_from_html_table(raw: str) -> List[str]:
    symbols: List[str] = []
    tables = re.findall(r"<table[^>]*>(.*?)</table>", raw, flags=re.IGNORECASE | re.DOTALL)
    for table in tables:
        if "Ticker" not in table and "Symbol" not in table:
            continue
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.IGNORECASE | re.DOTALL)
        if not rows:
            continue
        header_cells = re.findall(r"<th[^>]*>(.*?)</th>", rows[0], flags=re.IGNORECASE | re.DOTALL)
        header_text = [re.sub(r"<[^>]+>", "", html.unescape(x)).strip().lower() for x in header_cells]
        ticker_idx = next((i for i, h in enumerate(header_text) if ("ticker" in h or "symbol" in h)), None)
        if ticker_idx is None:
            continue

        for row in rows[1:]:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
            if ticker_idx >= len(cells):
                continue
            raw_symbol = re.sub(r"<[^>]+>", "", html.unescape(cells[ticker_idx]))
            raw_symbol = re.sub(r"\[[^\]]+\]", "", raw_symbol).strip().upper()
            raw_symbol = raw_symbol.replace("\xa0", "").replace(" ", "")
            if re.fullmatch(r"[A-Z]{1,5}", raw_symbol):
                symbols.append(raw_symbol)
    return _dedupe_symbols(symbols)


def _parse_symbols_from_csv(raw: str, symbol_column: str = "Ticker") -> List[str]:
    symbols: List[str] = []
    for row in csv.DictReader(raw.splitlines()):
        symbol = (row.get(symbol_column) or row.get("Symbol") or "").strip().upper()
        if re.fullmatch(r"[A-Z]{1,5}", symbol):
            symbols.append(symbol)
    return _dedupe_symbols(symbols)


def _polygon_key() -> str:
    key = POLYGON_API_KEY or os.getenv("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("Polygon API key is required. Set POLYGON_API_KEY or pass --polygon-api-key.")
    return key


def _fetch_polygon_aggs(symbol: str, multiplier: int, timespan: str, date_from: str, date_to: str) -> List[Candle]:
    key = _polygon_key()
    path = f"{POLYGON_AGGS_URL}/{urllib.parse.quote(symbol.upper())}/range/{multiplier}/{timespan}/{date_from}/{date_to}"
    query = urllib.parse.urlencode(
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": "50000",
            "apiKey": key,
        }
    )
    url = f"{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"})

    payload = None
    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            # Global pacing so multi-thread scans don't blast the API.
            with _POLYGON_LOCK:
                now = time.time()
                wait = max(0.0, _POLYGON_NEXT_ALLOWED_TS - now)
                if wait > 0:
                    time.sleep(wait)
                # Reserve the next slot immediately.
                globals()["_POLYGON_NEXT_ALLOWED_TS"] = time.time() + POLYGON_MIN_GAP_SEC

            with urllib.request.urlopen(req, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            break
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    backoff = float(retry_after) if retry_after else 1.0
                except ValueError:
                    backoff = 1.0
                # Exponential component, capped to avoid runaway waits.
                backoff = min(60.0, max(backoff, 1.0) * (1.6 ** attempt))
                with _POLYGON_LOCK:
                    globals()["_POLYGON_NEXT_ALLOWED_TS"] = max(_POLYGON_NEXT_ALLOWED_TS, time.time() + backoff)
                continue
            raise RuntimeError(f"failed to fetch {symbol} from Polygon: {exc}") from exc
        except urllib.error.URLError as exc:
            last_exc = exc
            # transient network issue: small retry
            time.sleep(min(5.0, 0.5 * (attempt + 1)))
            continue
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Polygon response for {symbol}: {exc}") from exc

    if payload is None:
        raise RuntimeError(f"failed to fetch {symbol} from Polygon: {last_exc}")

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError(f"unexpected Polygon payload for {symbol}")

    candles: List[Candle] = []
    for row in results:
        try:
            ts = float(row["t"]) / 1000.0
            candles.append(
                Candle(
                    date=datetime.utcfromtimestamp(ts),
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                    volume=float(row.get("v", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    candles.sort(key=lambda c: c.date)
    return candles


def fetch_polygon_history(symbol: str, lookback_days: int = 2200) -> List[Candle]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=max(lookback_days, 365))
    return _fetch_polygon_aggs(symbol, 1, "day", start.isoformat(), end.isoformat())


def fetch_polygon_intraday(symbol: str, interval_min: int = 5, lookback_days: int = 90) -> List[Candle]:
    end = datetime.utcnow().date()
    start = end - timedelta(days=max(lookback_days, 10))
    return _fetch_polygon_aggs(symbol, max(1, interval_min), "minute", start.isoformat(), end.isoformat())


def fetch_sp500_symbols() -> List[str]:
    req = urllib.request.Request(
        SP500_CONSTITUENTS_URL,
        headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch S&P 500 constituents: {exc}") from exc

    symbols: List[str] = []
    for row in csv.DictReader(raw.splitlines()):
        symbol = (row.get("Symbol") or "").strip().upper()
        if symbol:
            symbols.append(symbol)

    if not symbols:
        raise RuntimeError("S&P 500 constituents source returned no symbols")

    return _dedupe_symbols(symbols)


def fetch_nasdaq100_symbols() -> List[str]:
    sources = [
        ("nasdaq", NASDAQ100_COMPANIES_URL, "html"),
        ("wikipedia", WIKIPEDIA_NASDAQ100_URL, "html"),
        ("fallback_csv", NASDAQ100_FALLBACK_CSV_URL, "csv"),
    ]

    last_error = "unknown"
    for source_name, url, fmt in sources:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                raw = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            last_error = f"{source_name} fetch failed: {exc}"
            continue

        if fmt == "csv":
            symbols = _parse_symbols_from_csv(raw)
        else:
            symbols = _parse_symbols_from_html_table(raw)

        # Nasdaq-100 is ~100 names; accept if parsing yields a strong majority.
        if len(symbols) >= 80:
            return symbols
        last_error = f"{source_name} parsing returned too few symbols ({len(symbols)})"

    raise RuntimeError(f"Nasdaq-100 source parsing returned too few symbols. Last error: {last_error}")


def fetch_stooq_history(symbol: str) -> List[Candle]:
    # Stooq US tickers are lowercase with .us suffix (e.g. aapl.us).
    stooq_symbol = f"{symbol.lower()}.us"
    query = urllib.parse.urlencode({"s": stooq_symbol, "i": "d"})
    url = f"{STOOQ_URL}?{query}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {symbol}: {exc}") from exc

    if "Exceeded the daily hits limit" in raw:
        raise RuntimeError(f"failed to fetch {symbol}: stooq daily hit limit reached")

    rows = list(csv.DictReader(raw.splitlines()))
    candles: List[Candle] = []
    for row in rows:
        if row.get("Close") in (None, "", "0"):
            continue
        try:
            candles.append(
                Candle(
                    date=datetime.strptime(row["Date"], "%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )
        except (ValueError, KeyError):
            continue

    candles.sort(key=lambda c: c.date)
    if not candles:
        raise RuntimeError(f"failed to fetch {symbol}: empty stooq dataset")
    return candles


def fetch_yahoo_history(symbol: str) -> List[Candle]:
    yahoo_symbol = symbol.replace(".", "-")
    query = urllib.parse.urlencode(
        {
            "range": "10y",
            "interval": "1d",
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    url = f"{YAHOO_CHART_URL}/{urllib.parse.quote(yahoo_symbol)}?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {symbol} from yahoo: {exc}") from exc

    try:
        payload = json.loads(raw)
        result = (payload.get("chart", {}).get("result") or [None])[0] or {}
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"failed to parse yahoo history for {symbol}: {exc}") from exc

    candles: List[Candle] = []
    for i, ts in enumerate(timestamps):
        if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
            continue
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i] if i < len(volumes) else 0
        if any(x is None for x in (o, h, l, c)) or c in (0, "0"):
            continue
        try:
            candles.append(
                Candle(
                    date=datetime.utcfromtimestamp(int(ts)),
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v or 0),
                )
            )
        except (TypeError, ValueError, OSError):
            continue

    candles.sort(key=lambda c: c.date)
    if not candles:
        raise RuntimeError(f"failed to fetch {symbol} from yahoo: empty dataset")
    return candles


def fetch_history(symbol: str) -> List[Candle]:
    source = (DATA_SOURCE or "auto").lower()
    if source == "polygon":
        return fetch_polygon_history(symbol)
    if source == "stooq":
        try:
            return fetch_stooq_history(symbol)
        except RuntimeError:
            return fetch_yahoo_history(symbol)
    if source == "yahoo":
        return fetch_yahoo_history(symbol)

    # auto: prefer Polygon if key exists, else fallback to Stooq
    if POLYGON_API_KEY or os.getenv("POLYGON_API_KEY", ""):
        try:
            return fetch_polygon_history(symbol)
        except RuntimeError:
            pass
    try:
        return fetch_stooq_history(symbol)
    except RuntimeError:
        return fetch_yahoo_history(symbol)


def fetch_stooq_intraday(symbol: str, interval_min: int = 5) -> List[Candle]:
    stooq_symbol = f"{symbol.lower()}.us"
    query = urllib.parse.urlencode({"s": stooq_symbol, "i": str(interval_min)})
    url = f"{STOOQ_URL}?{query}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch intraday {symbol}: {exc}") from exc

    if "Exceeded the daily hits limit" in raw:
        raise RuntimeError(f"failed to fetch intraday {symbol}: stooq daily hit limit reached")

    rows = list(csv.DictReader(raw.splitlines()))
    candles: List[Candle] = []
    for row in rows:
        if row.get("Close") in (None, "", "0"):
            continue
        try:
            dt_raw = row.get("Date", "")
            try:
                dt = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = datetime.strptime(dt_raw, "%Y-%m-%d")

            candles.append(
                Candle(
                    date=dt,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )
        except (ValueError, KeyError):
            continue

    candles.sort(key=lambda c: c.date)
    if not candles:
        raise RuntimeError(f"failed to fetch intraday {symbol}: empty stooq dataset")
    return candles


def fetch_yahoo_intraday(symbol: str, interval_min: int = 1) -> List[Candle]:
    yahoo_symbol = symbol.replace(".", "-")
    interval = f"{max(1, int(interval_min))}m"
    supported = {"1m", "2m", "5m", "15m", "30m", "60m"}
    if interval not in supported:
        interval = "1m"
    range_value = "7d" if interval == "1m" else "60d"
    query = urllib.parse.urlencode(
        {
            "range": range_value,
            "interval": interval,
            "includePrePost": "false",
            "events": "div,splits",
        }
    )
    url = f"{YAHOO_CHART_URL}/{urllib.parse.quote(yahoo_symbol)}?{query}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (StockScannerCLI/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch intraday {symbol} from yahoo: {exc}") from exc

    try:
        payload = json.loads(raw)
        result = (payload.get("chart", {}).get("result") or [None])[0] or {}
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [None])[0] or {}
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
    except (ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"failed to parse yahoo intraday for {symbol}: {exc}") from exc

    candles: List[Candle] = []
    for i, ts in enumerate(timestamps):
        if i >= len(opens) or i >= len(highs) or i >= len(lows) or i >= len(closes):
            continue
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i] if i < len(volumes) else 0
        if any(x is None for x in (o, h, l, c)) or c in (0, "0"):
            continue
        try:
            candles.append(
                Candle(
                    date=datetime.utcfromtimestamp(int(ts)),
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(v or 0),
                )
            )
        except (TypeError, ValueError, OSError):
            continue

    candles.sort(key=lambda c: c.date)
    if not candles:
        raise RuntimeError(f"failed to fetch intraday {symbol} from yahoo: empty dataset")
    return candles


def fetch_intraday(symbol: str, interval_min: int = 5) -> List[Candle]:
    source = (DATA_SOURCE or "auto").lower()
    if source == "polygon":
        return fetch_polygon_intraday(symbol, interval_min=interval_min)
    if source == "stooq":
        try:
            return fetch_stooq_intraday(symbol, interval_min=interval_min)
        except RuntimeError:
            return fetch_yahoo_intraday(symbol, interval_min=interval_min)
    if source == "yahoo":
        return fetch_yahoo_intraday(symbol, interval_min=interval_min)

    if POLYGON_API_KEY or os.getenv("POLYGON_API_KEY", ""):
        try:
            return fetch_polygon_intraday(symbol, interval_min=interval_min)
        except RuntimeError:
            pass
    try:
        return fetch_stooq_intraday(symbol, interval_min=interval_min)
    except RuntimeError:
        return fetch_yahoo_intraday(symbol, interval_min=interval_min)


def resample_to_minutes(candles: Sequence[Candle], target_minutes: int = 233) -> List[Candle]:
    if not candles:
        return []

    buckets: dict[tuple[datetime.date, int], List[Candle]] = {}
    for c in candles:
        day = c.date.date()
        minute_of_day = c.date.hour * 60 + c.date.minute
        bucket = minute_of_day // target_minutes
        key = (day, bucket)
        buckets.setdefault(key, []).append(c)

    out: List[Candle] = []
    for key in sorted(buckets.keys()):
        chunk = sorted(buckets[key], key=lambda x: x.date)
        if not chunk:
            continue
        out.append(
            Candle(
                date=chunk[-1].date,
                open=chunk[0].open,
                high=max(x.high for x in chunk),
                low=min(x.low for x in chunk),
                close=chunk[-1].close,
                volume=sum(x.volume for x in chunk),
            )
        )
    return out


def sma(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        return math.nan
    return statistics.fmean(values[-period:])


def avg(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def stdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if values else math.nan


def ema_series(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    multiplier = 2.0 / (period + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append((value - out[-1]) * multiplier + out[-1])
    return out


def sma_series(values: Sequence[float], period: int) -> List[float]:
    out = [math.nan] * len(values)
    if period <= 0:
        return out
    for idx in range(period - 1, len(values)):
        out[idx] = statistics.fmean(values[idx - period + 1 : idx + 1])
    return out


def rolling_stdev(values: Sequence[float], period: int) -> List[float]:
    out = [math.nan] * len(values)
    if period <= 0:
        return out
    for idx in range(period - 1, len(values)):
        out[idx] = stdev(values[idx - period + 1 : idx + 1])
    return out


def rsi_series(values: Sequence[float], period: int = 14) -> List[float]:
    """Wilder-style RSI series used by most charting packages."""
    out = [math.nan] * len(values)
    if len(values) < period + 1:
        return out

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = statistics.fmean(gains)
    avg_loss = statistics.fmean(losses)

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    for idx in range(period + 1, len(values)):
        change = values[idx] - values[idx - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        if avg_loss == 0:
            out[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[idx] = 100.0 - (100.0 / (1.0 + rs))

    return out


def shift_series(values: Sequence[float], offset: int) -> List[float]:
    if offset <= 0:
        return list(values)
    out = [math.nan] * len(values)
    for i in range(offset, len(values)):
        out[i] = values[i - offset]
    return out


def rsi(values: Sequence[float], period: int = 14) -> float:
    series = rsi_series(values, period)
    return next((x for x in reversed(series) if not math.isnan(x)), math.nan)


def stoch_rsi(values: Sequence[float], rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple[float, float]:
    if len(values) < rsi_period + stoch_period + smooth_k + smooth_d:
        return math.nan, math.nan

    rsi_vals = rsi_series(values, rsi_period)

    raw_k = []
    for idx in range(len(values)):
        if idx < rsi_period + stoch_period - 1 or math.isnan(rsi_vals[idx]):
            raw_k.append(math.nan)
            continue
        window = [x for x in rsi_vals[idx - stoch_period + 1 : idx + 1] if not math.isnan(x)]
        if not window:
            raw_k.append(math.nan)
            continue
        lo = min(window)
        hi = max(window)
        if hi == lo:
            raw_k.append(0.0)
        else:
            raw_k.append((rsi_vals[idx] - lo) / (hi - lo) * 100.0)

    valid_k = [x for x in raw_k if not math.isnan(x)]
    if len(valid_k) < smooth_k + smooth_d:
        return math.nan, math.nan

    smooth_k_series = []
    for idx in range(len(raw_k)):
        window = raw_k[max(0, idx - smooth_k + 1) : idx + 1]
        window = [x for x in window if not math.isnan(x)]
        smooth_k_series.append(avg(window) if len(window) == smooth_k else math.nan)

    valid_smooth_k = [x for x in smooth_k_series if not math.isnan(x)]
    if len(valid_smooth_k) < smooth_d:
        return math.nan, math.nan

    smooth_d_series = []
    for idx in range(len(smooth_k_series)):
        window = smooth_k_series[max(0, idx - smooth_d + 1) : idx + 1]
        window = [x for x in window if not math.isnan(x)]
        smooth_d_series.append(avg(window) if len(window) == smooth_d else math.nan)

    k = next((x for x in reversed(smooth_k_series) if not math.isnan(x)), math.nan)
    d = next((x for x in reversed(smooth_d_series) if not math.isnan(x)), math.nan)
    return k, d


def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float]:
    if len(values) < slow + signal:
        return math.nan, math.nan
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema_series(macd_line, signal)
    return macd_line[-1], signal_line[-1]


def stoch_rsi_series(values: Sequence[float], rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []

    rsi_vals = rsi_series(values, rsi_period)

    raw_k = [math.nan] * len(values)
    for idx in range(len(values)):
        if idx < rsi_period + stoch_period - 1 or math.isnan(rsi_vals[idx]):
            continue
        window = [x for x in rsi_vals[idx - stoch_period + 1 : idx + 1] if not math.isnan(x)]
        if not window:
            continue
        lo = min(window)
        hi = max(window)
        raw_k[idx] = 0.0 if hi == lo else (rsi_vals[idx] - lo) / (hi - lo) * 100.0

    smooth_k_vals = [math.nan] * len(values)
    for idx in range(len(values)):
        window = raw_k[max(0, idx - smooth_k + 1) : idx + 1]
        window = [x for x in window if not math.isnan(x)]
        if len(window) == smooth_k:
            smooth_k_vals[idx] = avg(window)

    smooth_d_vals = [math.nan] * len(values)
    for idx in range(len(values)):
        window = smooth_k_vals[max(0, idx - smooth_d + 1) : idx + 1]
        window = [x for x in window if not math.isnan(x)]
        if len(window) == smooth_d:
            smooth_d_vals[idx] = avg(window)

    return smooth_k_vals, smooth_d_vals


def macd_series(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float]]:
    if not values:
        return [], []
    fast_ema = ema_series(values, fast)
    slow_ema = ema_series(values, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema_series(macd_line, signal)
    return macd_line, signal_line


def find_crossings(lhs: Sequence[float], rhs: Sequence[float]) -> Tuple[List[int], List[int]]:
    up = []
    down = []
    for i in range(1, min(len(lhs), len(rhs))):
        if any(math.isnan(x) for x in (lhs[i - 1], rhs[i - 1], lhs[i], rhs[i])):
            continue
        if lhs[i - 1] <= rhs[i - 1] and lhs[i] > rhs[i]:
            up.append(i)
        elif lhs[i - 1] >= rhs[i - 1] and lhs[i] < rhs[i]:
            down.append(i)
    return up, down


def plot_symbol_with_crossings(symbol: str, days: int, output_path: str | None) -> int:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("error: matplotlib is required for plotting. Install with `python3 -m pip install matplotlib`.", file=sys.stderr)
        return 1

    candles = fetch_history(symbol)
    if len(candles) < 60:
        print(f"error: not enough data to plot {symbol}", file=sys.stderr)
        return 1

    lookback = max(60, days)
    candles = candles[-lookback:]

    dates = [c.date for c in candles]
    closes = [c.close for c in candles]

    sma50 = sma_series(closes, POWS_SMA_FAST)
    sma89 = sma_series(closes, POWS_SMA_MID)
    sma200 = sma_series(closes, min(POWS_SMA_SLOW, len(closes)))

    bb_mid = sma_series(closes, POWS_BB_LENGTH)
    bb_std = rolling_stdev(closes, POWS_BB_LENGTH)
    bb_upper = [m + POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid, bb_std)]
    bb_lower = [m - POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid, bb_std)]

    stoch_k, stoch_d = stoch_rsi_series(
        closes,
        POWS_RSI_LENGTH,
        POWS_STOCH_PERIOD,
        POWS_STOCH_SMOOTH_K,
        POWS_STOCH_SMOOTH_D,
    )
    macd_line, macd_signal = macd_series(closes, POWS_MACD_FAST, POWS_MACD_SLOW, POWS_MACD_SIGNAL)

    sma_up, sma_down = find_crossings(sma50, sma89)
    macd_up, macd_down = find_crossings(macd_line, macd_signal)
    stoch_up, stoch_down = find_crossings(stoch_k, stoch_d)

    fig, (ax_price, ax_macd, ax_stoch) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax_price.plot(dates, closes, label="Close", color="#1f77b4", linewidth=1.4)
    ax_price.plot(dates, sma50, label="SMA50", color="#ff7f0e", linewidth=1.0)
    ax_price.plot(dates, sma89, label="SMA89", color="#2ca02c", linewidth=1.0)
    ax_price.plot(dates, sma200, label="SMA200", color="#9467bd", linewidth=1.0)
    ax_price.plot(dates, bb_upper, label="BB Upper", color="#7f7f7f", linewidth=0.9, linestyle="--")
    ax_price.plot(dates, bb_lower, label="BB Lower", color="#7f7f7f", linewidth=0.9, linestyle="--")
    ax_price.scatter([dates[i] for i in sma_up], [closes[i] for i in sma_up], marker="^", color="green", s=40, label="SMA50 x SMA89 up")
    ax_price.scatter([dates[i] for i in sma_down], [closes[i] for i in sma_down], marker="v", color="red", s=40, label="SMA50 x SMA89 down")
    ax_price.set_title(f"{symbol.upper()} Price + BB/SMA Crossings")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="upper left", ncol=3, fontsize=8)

    ax_macd.plot(dates, macd_line, label="MACD", color="#17becf", linewidth=1.2)
    ax_macd.plot(dates, macd_signal, label="Signal", color="#bcbd22", linewidth=1.2)
    ax_macd.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_macd.scatter([dates[i] for i in macd_up], [macd_line[i] for i in macd_up], marker="^", color="green", s=32, label="MACD up cross")
    ax_macd.scatter([dates[i] for i in macd_down], [macd_line[i] for i in macd_down], marker="v", color="red", s=32, label="MACD down cross")
    ax_macd.grid(alpha=0.25)
    ax_macd.legend(loc="upper left", ncol=3, fontsize=8)

    ax_stoch.plot(dates, stoch_k, label="StochRSI %K", color="#e377c2", linewidth=1.2)
    ax_stoch.plot(dates, stoch_d, label="StochRSI %D", color="#8c564b", linewidth=1.2)
    ax_stoch.axhline(POWS_STOCH_OB, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_stoch.axhline(POWS_STOCH_OS, color="green", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_stoch.scatter([dates[i] for i in stoch_up], [stoch_k[i] for i in stoch_up], marker="^", color="green", s=28, label="%K up cross")
    ax_stoch.scatter([dates[i] for i in stoch_down], [stoch_k[i] for i in stoch_down], marker="v", color="red", s=28, label="%K down cross")
    ax_stoch.set_ylim(0, 100)
    ax_stoch.grid(alpha=0.25)
    ax_stoch.legend(loc="upper left", ncol=3, fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
        print(f"Saved chart: {output_path}")
    else:
        plt.show()
    return 0


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    dmi_length: int = 14,
    adx_smoothing: int = 14,
) -> tuple[float, float, float]:
    if len(highs) < max(dmi_length, adx_smoothing) + 2 or len(lows) != len(highs) or len(closes) != len(highs):
        return math.nan, math.nan, math.nan

    tr: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for i in range(1, len(highs)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0.0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0.0)

        true_range = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr.append(true_range)

    if len(tr) < dmi_length + adx_smoothing:
        return math.nan, math.nan, math.nan

    plus_di_list: List[float] = []
    minus_di_list: List[float] = []
    dx_list: List[float] = []

    tr_smooth = sum(tr[:dmi_length])
    plus_smooth = sum(plus_dm[:dmi_length])
    minus_smooth = sum(minus_dm[:dmi_length])

    for i in range(dmi_length, len(tr)):
        tr_smooth = tr_smooth - (tr_smooth / dmi_length) + tr[i]
        plus_smooth = plus_smooth - (plus_smooth / dmi_length) + plus_dm[i]
        minus_smooth = minus_smooth - (minus_smooth / dmi_length) + minus_dm[i]

        plus_di = 0.0 if tr_smooth == 0 else (100.0 * plus_smooth / tr_smooth)
        minus_di = 0.0 if tr_smooth == 0 else (100.0 * minus_smooth / tr_smooth)
        plus_di_list.append(plus_di)
        minus_di_list.append(minus_di)
        den = plus_di + minus_di
        dx_list.append(0.0 if den == 0 else (abs(plus_di - minus_di) / den) * 100.0)

    if len(dx_list) < adx_smoothing:
        return math.nan, math.nan, math.nan

    adx_value = avg(dx_list[:adx_smoothing])
    for i in range(adx_smoothing, len(dx_list)):
        adx_value = ((adx_value * (adx_smoothing - 1)) + dx_list[i]) / adx_smoothing

    return adx_value, plus_di_list[-1], minus_di_list[-1]


def historical_band_swing_metrics(
    candles: Sequence[Candle], years: int = HISTORICAL_LOOKBACK_YEARS, horizon_bars: int = 15
) -> tuple[int, float, float]:
    if len(candles) < 260:
        return 0, 0.0, 0.0

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    dates = [c.date for c in candles]
    bb_mid_series = sma_series(closes, POWS_BB_LENGTH)
    bb_std_series = rolling_stdev(closes, POWS_BB_LENGTH)
    bb_upper_series = [m + POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid_series, bb_std_series)]
    bb_lower_series = [m - POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid_series, bb_std_series)]

    cutoff = dates[-1] - timedelta(days=365 * max(1, years))
    start_idx = next((i for i, d in enumerate(dates) if d >= cutoff), 0)
    start_idx = max(start_idx, POWS_BB_LENGTH + 2)
    end_idx = len(candles) - max(2, horizon_bars + 1)
    if end_idx <= start_idx:
        return 0, 0.0, 0.0

    setups = 0
    wins = 0
    returns: List[float] = []

    for i in range(start_idx, end_idx):
        bb_mid = bb_mid_series[i]
        bb_up = bb_upper_series[i]
        bb_lo = bb_lower_series[i]
        if math.isnan(bb_mid) or math.isnan(bb_up) or math.isnan(bb_lo):
            continue

        bullish = lows[i] <= bb_lo and closes[i] > bb_lo and closes[i] > closes[i - 1]
        bearish = highs[i] >= bb_up and closes[i] < bb_up and closes[i] < closes[i - 1]
        if not (bullish or bearish):
            continue

        direction = 1.0 if bullish else -1.0
        setups += 1
        entry = closes[i]
        target = bb_mid
        stop = lows[i] if bullish else highs[i]
        outcome = 0
        j_hit = min(len(candles) - 1, i + horizon_bars)

        for j in range(i + 1, j_hit + 1):
            hit_target = highs[j] >= target if bullish else lows[j] <= target
            hit_stop = lows[j] <= stop if bullish else highs[j] >= stop
            if hit_target and hit_stop:
                outcome = 0
                break
            if hit_target:
                outcome = 1
                wins += 1
                break
            if hit_stop:
                outcome = -1
                break

        if outcome == 1:
            realized = abs((target - entry) / entry)
        elif outcome == -1:
            realized = -abs((entry - stop) / entry)
        else:
            terminal = closes[j_hit]
            realized = ((terminal - entry) / entry) * direction
        returns.append(realized)

    if setups == 0:
        return 0, 0.0, 0.0
    win_rate = wins / setups
    avg_ret = statistics.fmean(returns) if returns else 0.0
    return setups, win_rate, avg_ret


def analyze_candles(symbol: str, candles: Sequence[Candle], timeframe: str = "1D") -> ScanResult | None:
    if len(candles) < 120:
        return None

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    close = closes[-1]
    sma20 = sma(closes, 20)
    sma50 = sma(closes, POWS_SMA_FAST)
    sma89 = sma(closes, POWS_SMA_MID)
    sma200 = sma(closes, POWS_SMA_SLOW)
    bb_mid = sma(closes, POWS_BB_LENGTH)
    bb_std = stdev(closes[-POWS_BB_LENGTH:])
    bb_upper = bb_mid + (POWS_BB_STDDEV * bb_std)
    bb_lower = bb_mid - (POWS_BB_STDDEV * bb_std)
    rsi14 = rsi(closes, POWS_RSI_LENGTH)
    stoch_rsi_k, stoch_rsi_d = stoch_rsi(
        closes,
        POWS_RSI_LENGTH,
        POWS_STOCH_PERIOD,
        POWS_STOCH_SMOOTH_K,
        POWS_STOCH_SMOOTH_D,
    )
    macd_line, macd_signal = macd(closes, POWS_MACD_FAST, POWS_MACD_SLOW, POWS_MACD_SIGNAL)
    sma2_series = shift_series(sma_series(closes, 2), 2)
    sma3_series = shift_series(sma_series(closes, 3), 3)
    stoch_k_series, stoch_d_series = stoch_rsi_series(
        closes,
        POWS_RSI_LENGTH,
        POWS_STOCH_PERIOD,
        POWS_STOCH_SMOOTH_K,
        POWS_STOCH_SMOOTH_D,
    )
    macd_series_vals, macd_signal_series = macd_series(closes, POWS_MACD_FAST, POWS_MACD_SLOW, POWS_MACD_SIGNAL)
    price_up, price_down = find_crossings(sma2_series, sma3_series)
    adx14, plus_di14, minus_di14 = adx(highs, lows, closes, POWS_DMI_LENGTH, POWS_ADX_SMOOTHING)
    avg_volume20 = avg(volumes[-20:])
    dollar_volume20 = avg([(closes[i] * volumes[i]) for i in range(len(closes) - 20, len(closes))])

    if (
        math.isnan(sma20)
        or math.isnan(sma50)
        or math.isnan(sma89)
        or math.isnan(sma200)
        or math.isnan(rsi14)
        or math.isnan(stoch_rsi_k)
        or math.isnan(macd_line)
        or math.isnan(adx14)
        or math.isnan(dollar_volume20)
    ):
        return None

    prior_20_high = max(highs[-21:-1])
    breakout_20d = close >= prior_20_high
    macd_up, macd_down = find_crossings(macd_series_vals, macd_signal_series)
    stoch_up, stoch_down = find_crossings(stoch_k_series, stoch_d_series)
    last_idx = len(closes) - 1
    price_cross_age = (last_idx - price_up[-1]) if price_up else 999
    price_bear_cross_age = (last_idx - price_down[-1]) if price_down else 999
    macd_cross_age = (last_idx - macd_up[-1]) if macd_up else 999
    stoch_cross_age = (last_idx - stoch_up[-1]) if stoch_up else 999
    dual_cross_gap = abs(macd_up[-1] - stoch_up[-1]) if (macd_up and stoch_up) else 999
    macd_bear_cross_age = (last_idx - macd_down[-1]) if macd_down else 999
    stoch_bear_cross_age = (last_idx - stoch_down[-1]) if stoch_down else 999
    dual_bear_cross_gap = abs(macd_down[-1] - stoch_down[-1]) if (macd_down and stoch_down) else 999
    triple_cross_gap = (
        max(price_up[-1], macd_up[-1], stoch_up[-1]) - min(price_up[-1], macd_up[-1], stoch_up[-1])
        if (price_up and macd_up and stoch_up)
        else 999
    )
    triple_bear_cross_gap = (
        max(price_down[-1], macd_down[-1], stoch_down[-1]) - min(price_down[-1], macd_down[-1], stoch_down[-1])
        if (price_down and macd_down and stoch_down)
        else 999
    )

    price_gap_now = sma2_series[-1] - sma3_series[-1]
    price_gap_prev = sma2_series[-2] - sma3_series[-2]
    macd_gap_now = macd_series_vals[-1] - macd_signal_series[-1]
    macd_gap_prev = macd_series_vals[-2] - macd_signal_series[-2]
    stoch_gap_now = stoch_k_series[-1] - stoch_d_series[-1]
    stoch_gap_prev = stoch_k_series[-2] - stoch_d_series[-2]

    bb_mid_series = sma_series(closes, POWS_BB_LENGTH)
    bb_std_series = rolling_stdev(closes, POWS_BB_LENGTH)
    bb_upper_series = [m + POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid_series, bb_std_series)]
    bb_lower_series = [m - POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid_series, bb_std_series)]

    band_touch_lookback = 6
    start = max(0, len(closes) - band_touch_lookback)
    lower_touch_idx = [i for i in range(len(closes)) if (not math.isnan(bb_lower_series[i])) and (lows[i] <= bb_lower_series[i])]
    upper_touch_idx = [i for i in range(len(closes)) if (not math.isnan(bb_upper_series[i])) and (highs[i] >= bb_upper_series[i])]
    touched_lower = any(i >= start for i in lower_touch_idx)
    touched_upper = any(i >= start for i in upper_touch_idx)
    touched_outer_band_recent = touched_lower or touched_upper
    last_touch_idx = max(lower_touch_idx[-1] if lower_touch_idx else -1, upper_touch_idx[-1] if upper_touch_idx else -1)
    outer_touch_age = (last_idx - last_touch_idx) if last_touch_idx >= 0 else 999

    liftoff_from_band = touched_lower and len(closes) >= 3 and closes[-1] > bb_lower and closes[-1] > closes[-2]
    rejection_from_band = touched_upper and len(closes) >= 3 and closes[-1] < bb_upper and closes[-1] < closes[-2]
    bw_now = (bb_upper - bb_lower) / bb_mid if bb_mid else math.nan
    prior_mid_idx = max(0, len(closes) - 6)
    prior_bw = math.nan
    if prior_mid_idx < len(closes) and not math.isnan(bb_mid_series[prior_mid_idx]) and bb_mid_series[prior_mid_idx] != 0:
        prior_bw = (bb_upper_series[prior_mid_idx] - bb_lower_series[prior_mid_idx]) / bb_mid_series[prior_mid_idx]
    band_width_expansion = 0.0
    if not math.isnan(bw_now) and not math.isnan(prior_bw) and prior_bw > 0:
        band_width_expansion = (bw_now - prior_bw) / prior_bw
    bw_recent = [
        (bb_upper_series[i] - bb_lower_series[i]) / bb_mid_series[i]
        for i in range(max(0, len(closes) - 50), len(closes))
        if (
            i < len(bb_mid_series)
            and not math.isnan(bb_upper_series[i])
            and not math.isnan(bb_lower_series[i])
            and not math.isnan(bb_mid_series[i])
            and bb_mid_series[i] != 0
        )
    ]
    bw_median = statistics.median(bw_recent) if bw_recent else math.nan
    # Course timing rule: BB widening should begin roughly 5-14 bars before setup maturity.
    band_widen_start_age = 999
    band_widen_window_ok = False
    if len(closes) >= 20 and not math.isnan(bw_now):
        min_age = 5
        max_age = 14
        start_lo = max(0, last_idx - max_age)
        start_hi = max(0, last_idx - min_age)
        best_idx = -1
        best_bw = math.inf
        for i in range(start_lo, start_hi + 1):
            if i >= len(bb_mid_series):
                continue
            mid_i = bb_mid_series[i]
            up_i = bb_upper_series[i]
            low_i = bb_lower_series[i]
            if math.isnan(mid_i) or math.isnan(up_i) or math.isnan(low_i) or mid_i == 0:
                continue
            bw_i = (up_i - low_i) / mid_i
            if bw_i < best_bw:
                best_bw = bw_i
                best_idx = i
        if best_idx >= 0 and best_bw > 0 and bw_now > 0:
            band_widen_start_age = last_idx - best_idx
            step_count = 0
            up_steps = 0
            prev_bw = best_bw
            for i in range(best_idx + 1, last_idx + 1):
                if i >= len(bb_mid_series):
                    continue
                mid_i = bb_mid_series[i]
                up_i = bb_upper_series[i]
                low_i = bb_lower_series[i]
                if math.isnan(mid_i) or math.isnan(up_i) or math.isnan(low_i) or mid_i == 0:
                    continue
                bw_i = (up_i - low_i) / mid_i
                if bw_i >= prev_bw:
                    up_steps += 1
                step_count += 1
                prev_bw = bw_i
            growth = (bw_now - best_bw) / best_bw
            up_ratio = (up_steps / step_count) if step_count > 0 else 0.0
            band_widen_window_ok = (
                min_age <= band_widen_start_age <= max_age
                and growth >= 0.08
                and up_ratio >= 0.55
            )

    def _prox(gap: float, scale: float) -> float:
        if math.isnan(gap) or scale <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (abs(gap) / scale)))

    def _toward_bull(now: float, prev: float) -> float:
        if math.isnan(now) or math.isnan(prev):
            return 0.0
        return 1.0 if now >= prev else 0.0

    def _toward_bear(now: float, prev: float) -> float:
        if math.isnan(now) or math.isnan(prev):
            return 0.0
        return 1.0 if now <= prev else 0.0

    pre3x_bull_score = (
        2.0 * _prox(price_gap_now, max(0.05, close * 0.003))
        + 2.0 * _prox(macd_gap_now, max(0.05, abs(macd_signal) * 0.75))
        + 2.0 * _prox(stoch_gap_now, 8.0)
        + _toward_bull(price_gap_now, price_gap_prev)
        + _toward_bull(macd_gap_now, macd_gap_prev)
        + _toward_bull(stoch_gap_now, stoch_gap_prev)
        + (1.0 if band_width_expansion >= 0 else 0.0)
        + (1.0 if close >= bb_mid else 0.0)
    )
    pre3x_bear_score = (
        2.0 * _prox(price_gap_now, max(0.05, close * 0.003))
        + 2.0 * _prox(macd_gap_now, max(0.05, abs(macd_signal) * 0.75))
        + 2.0 * _prox(stoch_gap_now, 8.0)
        + _toward_bear(price_gap_now, price_gap_prev)
        + _toward_bear(macd_gap_now, macd_gap_prev)
        + _toward_bear(stoch_gap_now, stoch_gap_prev)
        + (1.0 if band_width_expansion >= 0 else 0.0)
        + (1.0 if close <= bb_mid else 0.0)
    )
    hist_setups_5y, hist_win_rate_5y, hist_avg_return_5y = historical_band_swing_metrics(candles, years=HISTORICAL_LOOKBACK_YEARS)

    bull_evidence = (
        pre3x_bull_score
        + (1.0 if macd_line >= macd_signal else 0.0)
        + (1.0 if stoch_rsi_k >= stoch_rsi_d else 0.0)
        + (1.0 if close >= bb_mid else 0.0)
    )
    bear_evidence = (
        pre3x_bear_score
        + (1.0 if macd_line <= macd_signal else 0.0)
        + (1.0 if stoch_rsi_k <= stoch_rsi_d else 0.0)
        + (1.0 if close <= bb_mid else 0.0)
    )
    setup_direction = "bull" if bull_evidence >= bear_evidence else "bear"
    if setup_direction == "bull":
        if liftoff_from_band:
            setup_type = "Band Bounce"
        elif band_width_expansion > 0 and close >= bb_mid:
            setup_type = "Band Expansion"
        else:
            setup_type = "Developing"
        target_mid_pct = max(0.0, (bb_mid - close) / close)
        target_band_pct = max(0.0, (bb_upper - close) / close)
        risk_anchor = max(bb_lower, lows[-1])
        risk_pct = max(0.0, (close - risk_anchor) / close)
    else:
        if rejection_from_band:
            setup_type = "Band Reject"
        elif band_width_expansion > 0 and close <= bb_mid:
            setup_type = "Band Expansion"
        else:
            setup_type = "Developing"
        target_mid_pct = max(0.0, (close - bb_mid) / close)
        target_band_pct = max(0.0, (close - bb_lower) / close)
        risk_anchor = min(bb_upper, highs[-1])
        risk_pct = max(0.0, (risk_anchor - close) / close)

    rr_mid = (target_mid_pct / risk_pct) if risk_pct > 0 else 0.0

    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    hist_quality = _clamp01((hist_win_rate_5y - 0.45) / 0.25)
    hist_density = _clamp01(hist_setups_5y / 30.0)
    move_quality = _clamp01(target_mid_pct / 0.025)
    rr_quality = _clamp01(rr_mid / 2.0)
    pre3x_quality = _clamp01(max(pre3x_bull_score, pre3x_bear_score) / 10.0)
    liq_quality = _clamp01((math.log10(max(dollar_volume20, 1.0)) - 6.0) / 2.0)
    options_setup_score = (
        35.0 * pre3x_quality
        + 20.0 * hist_quality
        + 10.0 * hist_density
        + 15.0 * move_quality
        + 10.0 * rr_quality
        + 10.0 * liq_quality
    )
    bb_compression_score = 0.0
    if not math.isnan(bw_now) and not math.isnan(bw_median) and bw_median > 0:
        bb_compression_score = _clamp01((bw_median - bw_now) / bw_median)

    # Course-aligned signal quality score (0-100): trend + fresh momentum + BB behavior.
    if setup_direction == "bull":
        trend_quality = 1.0 if (close >= sma50 and sma50 >= sma89) else 0.0
        momentum_quality = 1.0 if (macd_line >= macd_signal and stoch_rsi_k >= stoch_rsi_d) else 0.0
        band_action_quality = 1.0 if (liftoff_from_band and outer_touch_age <= 3) else 0.0
        fresh_age = min(price_cross_age, macd_cross_age, stoch_cross_age, outer_touch_age)
    else:
        trend_quality = 1.0 if (close <= sma50 and sma50 <= sma89) else 0.0
        momentum_quality = 1.0 if (macd_line <= macd_signal and stoch_rsi_k <= stoch_rsi_d) else 0.0
        band_action_quality = 1.0 if (rejection_from_band and outer_touch_age <= 3) else 0.0
        fresh_age = min(price_bear_cross_age, macd_bear_cross_age, stoch_bear_cross_age, outer_touch_age)
    spread_quality = _clamp01((band_width_expansion - 0.02) / 0.12)
    widen_window_quality = 1.0 if band_widen_window_ok else 0.0
    fresh_quality = _clamp01(1.0 - (fresh_age / 3.0))
    target_quality = _clamp01(target_mid_pct / 0.012)
    rr_course_quality = _clamp01(rr_mid / 1.5)
    course_pattern_score = 100.0 * (
        0.20 * trend_quality
        + 0.20 * momentum_quality
        + 0.20 * band_action_quality
        + 0.10 * spread_quality
        + 0.15 * widen_window_quality
        + 0.10 * fresh_quality
        + 0.05 * target_quality
        + 0.05 * rr_course_quality
    )

    # Composite ranking: trend, liquidity, and moderate RSI preferred.
    trend_component = (close / sma20) + (sma20 / sma50)
    liquidity_component = math.log10(max(dollar_volume20, 1.0))
    rsi_component = 1.0 - abs(rsi14 - 55.0) / 55.0
    adx_component = min(adx14, 50.0) / 50.0
    macd_component = 1.0 if macd_line > macd_signal else 0.5
    stoch_component = 1.0 - abs(stoch_rsi_k - 50.0) / 50.0
    bull_cross_component = 1.0 if (macd_cross_age <= 4 and stoch_cross_age <= 4) else 0.0
    bear_cross_component = 1.0 if (macd_bear_cross_age <= 4 and stoch_bear_cross_age <= 4) else 0.0
    liftoff_component = 1.0 if liftoff_from_band and band_width_expansion > 0 else 0.0
    rejection_component = 1.0 if rejection_from_band and band_width_expansion > 0 else 0.0
    score = (
        trend_component
        + liquidity_component
        + rsi_component
        + adx_component
        + macd_component
        + stoch_component
        + max(bull_cross_component, bear_cross_component)
        + max(liftoff_component, rejection_component)
        + min(1.0, hist_setups_5y / 40.0)
        + (hist_win_rate_5y - 0.5)
        + (hist_avg_return_5y * 10.0)
        + (options_setup_score / 100.0)
        + (course_pattern_score / 25.0)
    )

    return ScanResult(
        symbol=symbol,
        close=close,
        sma20=sma20,
        sma50=sma50,
        sma89=sma89,
        sma200=sma200,
        bb_mid=bb_mid,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        rsi14=rsi14,
        stoch_rsi_k=stoch_rsi_k,
        stoch_rsi_d=stoch_rsi_d,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_gap_now=macd_gap_now,
        macd_gap_prev=macd_gap_prev,
        stoch_gap_now=stoch_gap_now,
        stoch_gap_prev=stoch_gap_prev,
        adx14=adx14,
        plus_di14=plus_di14,
        minus_di14=minus_di14,
        avg_volume20=avg_volume20,
        dollar_volume20=dollar_volume20,
        breakout_20d=breakout_20d,
        price_cross_age=price_cross_age,
        price_bear_cross_age=price_bear_cross_age,
        macd_cross_age=macd_cross_age,
        stoch_cross_age=stoch_cross_age,
        dual_cross_gap=dual_cross_gap,
        macd_bear_cross_age=macd_bear_cross_age,
        stoch_bear_cross_age=stoch_bear_cross_age,
        dual_bear_cross_gap=dual_bear_cross_gap,
        triple_cross_gap=triple_cross_gap,
        triple_bear_cross_gap=triple_bear_cross_gap,
        pre3x_bull_score=pre3x_bull_score,
        pre3x_bear_score=pre3x_bear_score,
        hist_setups_5y=hist_setups_5y,
        hist_win_rate_5y=hist_win_rate_5y,
        hist_avg_return_5y=hist_avg_return_5y,
        setup_direction=setup_direction,
        setup_type=setup_type,
        target_mid_pct=target_mid_pct,
        target_band_pct=target_band_pct,
        risk_pct=risk_pct,
        rr_mid=rr_mid,
        options_setup_score=options_setup_score,
        course_pattern_score=course_pattern_score,
        bb_compression_score=bb_compression_score,
        touched_outer_band_recent=touched_outer_band_recent,
        outer_touch_age=outer_touch_age,
        band_width_expansion=band_width_expansion,
        band_widen_start_age=band_widen_start_age,
        band_widen_window_ok=band_widen_window_ok,
        liftoff_from_band=liftoff_from_band,
        rejection_from_band=rejection_from_band,
        timeframe=timeframe,
        score=score,
    )


def analyze_symbol(symbol: str) -> ScanResult | None:
    candles = fetch_history(symbol)
    return analyze_candles(symbol, candles, "1D")


def _direction_match(result: ScanResult, args: argparse.Namespace) -> tuple[bool, bool]:
    bull = True
    bear = True

    if args.signal_direction == "bull":
        bear = False
    elif args.signal_direction == "bear":
        bull = False
    return bull, bear


def _passes_directional_setup(
    result: ScanResult, args: argparse.Namespace, require_primary_uptrend: bool, secondary_confirmation: bool = False
) -> bool:
    require_uptrend = args.require_uptrend
    require_macd_bull = args.require_macd_bull
    require_di_bull = args.require_di_bull
    require_macd_stoch_cross = args.require_macd_stoch_cross
    require_band_liftoff = args.require_band_liftoff
    bb_spread_watchlist = getattr(args, "bb_spread_watchlist", False)
    require_simultaneous_cross = args.require_simultaneous_cross
    # Hard recency ceiling: only current/potential setups near present are allowed.
    recent_cross_limit = 3
    max_macd_cross_age = min(max(1, int(getattr(args, "max_macd_cross_age", 3))), recent_cross_limit)
    max_stoch_cross_age = min(max(1, int(getattr(args, "max_stoch_cross_age", 3))), recent_cross_limit)
    want_bull, want_bear = _direction_match(result, args)

    bull_ok = True
    bear_ok = True

    if require_uptrend:
        if require_primary_uptrend:
            bull_ok = bull_ok and (result.close > result.sma20 > result.sma50)
            bear_ok = bear_ok and (result.close < result.sma20 < result.sma50)
        else:
            bull_ok = bull_ok and (result.close > result.sma50 > result.sma89 > result.sma200)
            bear_ok = bear_ok and (result.close < result.sma50 < result.sma89 < result.sma200)
    if args.pows:
        # POWS context: trend bias without requiring perfect full-stack moving-average order.
        if bb_spread_watchlist:
            # Watchlist mode allows near-trend structures so early setups can surface.
            bull_ok = bull_ok and (result.close >= result.sma50 * 0.985 and result.sma50 >= result.sma89 * 0.99)
            bear_ok = bear_ok and (result.close <= result.sma50 * 1.015 and result.sma50 <= result.sma89 * 1.01)
        else:
            bull_ok = bull_ok and (result.close > result.sma50 and result.sma50 >= result.sma89)
            bear_ok = bear_ok and (result.close < result.sma50 and result.sma50 <= result.sma89)
        # Course 3x language: accept either confirmed cross timing or strong pre-3x development.
        bull_confirmed = result.price_cross_age <= args.cross_lookback and result.triple_cross_gap <= 4
        bear_confirmed = result.price_bear_cross_age <= args.cross_lookback and result.triple_bear_cross_gap <= 4
        bull_developing = bb_spread_watchlist and result.pre3x_bull_score >= 6.0
        bear_developing = bb_spread_watchlist and result.pre3x_bear_score >= 6.0
        bull_ok = bull_ok and (bull_confirmed or bull_developing)
        bear_ok = bear_ok and (bear_confirmed or bear_developing)

    if require_macd_bull:
        bull_ok = bull_ok and (result.macd > result.macd_signal)
        bear_ok = bear_ok and (result.macd < result.macd_signal)
    if require_di_bull:
        bull_ok = bull_ok and (result.plus_di14 > result.minus_di14)
        bear_ok = bear_ok and (result.plus_di14 < result.minus_di14)
    if require_macd_stoch_cross:
        bull_ok = bull_ok and (result.macd_cross_age <= args.cross_lookback and result.stoch_cross_age <= args.cross_lookback)
        bear_ok = bear_ok and (result.macd_bear_cross_age <= args.cross_lookback and result.stoch_bear_cross_age <= args.cross_lookback)
    if require_simultaneous_cross:
        bull_ok = bull_ok and (result.dual_cross_gap <= 1)
        bear_ok = bear_ok and (result.dual_bear_cross_gap <= 1)

    # Core timing rule: MACD cross must be recent or imminently approaching within 3 bars.
    macd_gap_thresh = max(0.03, abs(result.macd_signal) * 0.25)
    bull_macd_cross_up = (
        result.macd_cross_age <= max_macd_cross_age
        and result.macd_cross_age <= result.macd_bear_cross_age
    )
    bear_macd_cross_down = (
        result.macd_bear_cross_age <= max_macd_cross_age
        and result.macd_bear_cross_age <= result.macd_cross_age
    )
    bull_macd_imminent = (
        result.macd <= result.macd_signal
        and result.macd_gap_prev < 0
        and result.macd_gap_now > result.macd_gap_prev
        and result.macd_gap_now >= -macd_gap_thresh
    )
    bear_macd_imminent = (
        result.macd >= result.macd_signal
        and result.macd_gap_prev > 0
        and result.macd_gap_now < result.macd_gap_prev
        and result.macd_gap_now <= macd_gap_thresh
    )
    bull_ok = bull_ok and (bull_macd_cross_up or bull_macd_imminent)
    bear_ok = bear_ok and (bear_macd_cross_down or bear_macd_imminent)

    stoch_gap_thresh = 4.0
    bull_stoch_cross_up = (
        result.stoch_cross_age <= max_stoch_cross_age
        and result.stoch_cross_age <= result.stoch_bear_cross_age
    )
    bear_stoch_cross_down = (
        result.stoch_bear_cross_age <= max_stoch_cross_age
        and result.stoch_bear_cross_age <= result.stoch_cross_age
    )
    bull_stoch_imminent = (
        result.stoch_rsi_k <= result.stoch_rsi_d
        and result.stoch_gap_prev < 0
        and result.stoch_gap_now > result.stoch_gap_prev
        and result.stoch_gap_now >= -stoch_gap_thresh
    )
    bear_stoch_imminent = (
        result.stoch_rsi_k >= result.stoch_rsi_d
        and result.stoch_gap_prev > 0
        and result.stoch_gap_now < result.stoch_gap_prev
        and result.stoch_gap_now <= stoch_gap_thresh
    )
    bull_ok = bull_ok and (bull_stoch_cross_up or bull_stoch_imminent)
    bear_ok = bear_ok and (bear_stoch_cross_down or bear_stoch_imminent)

    # Pre-liftoff discovery path for setups that can evolve into 3x confirmation.
    spread_floor = max(0.0, args.min_band_expansion)
    bull_bb_spread_ok = (
        result.band_width_expansion >= spread_floor
        and result.touched_outer_band_recent
        and result.outer_touch_age <= args.band_touch_lookback
        and result.close >= result.bb_mid
    )
    bear_bb_spread_ok = (
        result.band_width_expansion >= spread_floor
        and result.touched_outer_band_recent
        and result.outer_touch_age <= args.band_touch_lookback
        and result.close <= result.bb_mid
    )

    if require_band_liftoff:
        bull_ok = bull_ok and (
            result.touched_outer_band_recent
            and result.outer_touch_age <= args.band_touch_lookback
            and result.liftoff_from_band
            and result.band_width_expansion >= args.min_band_expansion
        )
        bear_ok = bear_ok and (
            result.touched_outer_band_recent
            and result.outer_touch_age <= args.band_touch_lookback
            and result.rejection_from_band
            and result.band_width_expansion >= args.min_band_expansion
        )
    elif bb_spread_watchlist:
        bull_ok = bull_ok and (bull_bb_spread_ok and result.pre3x_bull_score >= 6.0)
        bear_ok = bear_ok and (bear_bb_spread_ok and result.pre3x_bear_score >= 6.0)

    # Only apply full options viability gates on primary (daily) timeframe.
    if not secondary_confirmation:
        if bb_spread_watchlist and not require_band_liftoff:
            if require_macd_stoch_cross:
                min_target_mid = 0.002
                min_setup_score = 22.0
            else:
                min_target_mid = 0.0
                min_setup_score = 12.0
        else:
            min_target_mid = 0.01
            min_setup_score = 35.0
        bull_ok = bull_ok and (result.target_mid_pct >= min_target_mid and result.options_setup_score >= min_setup_score)
        bear_ok = bear_ok and (result.target_mid_pct >= min_target_mid and result.options_setup_score >= min_setup_score)
        # Enforce user rule: widening should start about 5-14 bars before "ready" setups.
        if result.timeframe == "1D":
            bull_ok = bull_ok and bool(getattr(result, "band_widen_window_ok", False))
            bear_ok = bear_ok and bool(getattr(result, "band_widen_window_ok", False))

    # Avoid extremely stretched end-of-move candles in either direction.
    bull_ok = bull_ok and not (result.close > result.bb_upper * 1.03)
    bear_ok = bear_ok and not (result.close < result.bb_lower * 0.97)

    # Freshness gate: keep surfaced daily setups in-the-moment (or very recent).
    max_setup_age = int(getattr(args, "max_setup_age", 0) or 0)
    if max_setup_age > 0:
        max_setup_age = min(max_setup_age, recent_cross_limit)
    if not secondary_confirmation and result.timeframe == "1D" and max_setup_age > 0:
        bull_age = min(result.price_cross_age, result.macd_cross_age, result.stoch_cross_age, result.outer_touch_age)
        bear_age = min(result.price_bear_cross_age, result.macd_bear_cross_age, result.stoch_bear_cross_age, result.outer_touch_age)
        bull_ok = bull_ok and (bull_age <= max_setup_age)
        bear_ok = bear_ok and (bear_age <= max_setup_age)

    return (want_bull and bull_ok) or (want_bear and bear_ok)


def passes_filters(result: ScanResult, args: argparse.Namespace) -> bool:
    require_breakout = args.require_breakout and not args.pows

    if not (args.min_price <= result.close <= args.max_price):
        return False
    if result.dollar_volume20 < args.min_dollar_volume:
        return False
    if args.signal_direction == "bull":
        if result.rsi14 > args.max_rsi:
            return False
        if result.stoch_rsi_k > args.max_stoch_rsi_k:
            return False
    if result.adx14 < args.min_adx:
        return False
    if result.course_pattern_score < float(getattr(args, "min_course_pattern_score", 0.0)):
        return False
    if require_breakout and not result.breakout_20d:
        return False
    return _passes_directional_setup(result, args, require_primary_uptrend=True, secondary_confirmation=False)


def passes_secondary_timeframe_filters(result: ScanResult, args: argparse.Namespace) -> bool:
    return _passes_directional_setup(result, args, require_primary_uptrend=False, secondary_confirmation=True)


def format_money(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def print_results(results: Iterable[ScanResult], top: int) -> None:
    ordered = sorted(results, key=lambda r: r.score, reverse=True)[:top]
    if not ordered:
        print("No symbols matched the scan filters.")
        return

    headers = ["Symbol", "Close", "SMA50", "SMA89", "SMA200", "StochK", "MACD", "ADX", "MxAge", "SxAge", "MxAgeB", "SxAgeB", "BBExp", "$Vol20", "Score"]
    print(" ".join(h.ljust(10) for h in headers))
    for row in ordered:
        print(
            f"{row.symbol:<10}"
            f"{row.close:<10.2f}"
            f"{row.sma50:<10.2f}"
            f"{row.sma89:<10.2f}"
            f"{row.sma200:<10.2f}"
            f"{row.stoch_rsi_k:<10.2f}"
            f"{(row.macd - row.macd_signal):<10.3f}"
            f"{row.adx14:<10.2f}"
            f"{row.macd_cross_age:<10}"
            f"{row.stoch_cross_age:<10}"
            f"{row.macd_bear_cross_age:<10}"
            f"{row.stoch_bear_cross_age:<10}"
            f"{row.band_width_expansion:<10.3f}"
            f"{format_money(row.dollar_volume20):<10}"
            f"{row.score:<10.3f}"
        )


def scan_symbols(symbols: Sequence[str], args: argparse.Namespace, workers: int = 12) -> tuple[List[ScanResult], int]:
    workers = max(1, int(workers))
    results: List[ScanResult] = []
    failures = 0

    def _run_one(symbol: str) -> tuple[str, ScanResult | None, RuntimeError | None]:
        try:
            daily = analyze_symbol(symbol)
            if not daily:
                return symbol, None, None

            if args.require_daily_and_233:
                intraday = fetch_intraday(symbol, interval_min=args.intraday_interval_min)
                c233 = resample_to_minutes(intraday, target_minutes=233)
                tf233 = analyze_candles(symbol, c233, "233m")
                if not tf233:
                    return symbol, None, None
                if not passes_secondary_timeframe_filters(tf233, args):
                    return symbol, None, None

            return symbol, daily, None
        except RuntimeError as exc:
            return symbol, None, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_one, symbol) for symbol in symbols]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            symbol, analyzed, err = fut.result()
            if args.verbose:
                print(f"Progress: {idx}/{len(symbols)} ({symbol})", file=sys.stderr, flush=True)
            if err:
                failures += 1
                if args.verbose:
                    print(f"warn: {err}", file=sys.stderr)
                continue
            if analyzed and passes_filters(analyzed, args):
                results.append(analyzed)

    return results, failures


def main() -> int:
    args = parse_args()
    configure_data_source(args.data_source, args.polygon_api_key)
    if args.plot_symbol:
        return plot_symbol_with_crossings(args.plot_symbol, args.plot_days, args.plot_output)

    if args.sp500 and args.qqq:
        try:
            symbols = _dedupe_symbols(fetch_sp500_symbols() + fetch_nasdaq100_symbols())
        except RuntimeError as exc:
            print(f"error loading symbols: {exc}", file=sys.stderr)
            return 1
    elif args.sp500:
        try:
            symbols = fetch_sp500_symbols()
        except RuntimeError as exc:
            print(f"error loading symbols: {exc}", file=sys.stderr)
            return 1
    elif args.qqq:
        try:
            symbols = fetch_nasdaq100_symbols()
        except RuntimeError as exc:
            print(f"error loading symbols: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            symbols = load_symbols(args.symbols, args.symbols_file)
        except OSError as exc:
            print(f"error loading symbols: {exc}", file=sys.stderr)
            return 1

    if not symbols:
        print("No symbols provided.", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Scanning {len(symbols)} symbols...", file=sys.stderr, flush=True)

    results, failures = scan_symbols(symbols, args, workers=args.workers)

    print_results(results, args.top)

    if failures and args.verbose:
        print(f"\nCompleted with {failures} fetch failure(s).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
