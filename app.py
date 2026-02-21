#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import copy
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from html import unescape
from types import SimpleNamespace
from typing import Any

from flask import Flask, Response, render_template, render_template_string, request, stream_with_context

from stock_scanner import (
    Candle,
    POWS_BB_LENGTH,
    POWS_BB_STDDEV,
    POWS_MACD_FAST,
    POWS_MACD_SIGNAL,
    POWS_MACD_SLOW,
    POWS_RSI_LENGTH,
    analyze_candles,
    configure_data_source,
    fetch_nasdaq100_symbols,
    fetch_sp500_symbols,
    fetch_history,
    fetch_intraday,
    find_crossings,
    format_money,
    load_symbols,
    macd_series,
    passes_filters,
    passes_secondary_timeframe_filters,
    resample_to_minutes,
    rolling_stdev,
    scan_symbols,
    sma_series,
    stoch_rsi_series,
)

app = Flask(__name__)

FED_CALENDAR_ROOT = "https://www.federalreserve.gov"
FED_WHATS_NEXT_URL = "https://www.federalreserve.gov/whatsnext.htm"
BLS_EMPLOYMENT_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/empsit.htm"
YAHOO_QUOTE_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary"
EVENT_LOOKAHEAD_DAYS = int(os.getenv("EVENT_LOOKAHEAD_DAYS", "120"))
EVENT_CACHE_TTL_SEC = int(os.getenv("EVENT_CACHE_TTL_SEC", "1800"))
EVENT_BLOCK_DAYS = int(os.getenv("EVENT_BLOCK_DAYS", "7"))
EVENT_HTTP_TIMEOUT_SEC = max(2, int(os.getenv("EVENT_HTTP_TIMEOUT_SEC", "4")))
FED_MONTH_PAGE_LIMIT = max(1, int(os.getenv("FED_MONTH_PAGE_LIMIT", "6")))
PATTERN_REF_SYMBOL_DEFAULT = str(os.getenv("PATTERN_REF_SYMBOL", "GOOGL")).strip().upper() or "GOOGL"
PATTERN_REF_WEIGHT = float(os.getenv("PATTERN_REF_WEIGHT", "1000"))
_EVENT_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_EVENT_CACHE_LOCK = threading.Lock()
CHART_CACHE_TTL_SEC = int(os.getenv("CHART_CACHE_TTL_SEC", "300"))
_CHART_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CHART_CACHE_LOCK = threading.Lock()
SCAN_SNAPSHOT_TTL_SEC = max(30, int(os.getenv("SCAN_SNAPSHOT_TTL_SEC", "180")))
_SCAN_SNAPSHOT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SCAN_SNAPSHOT_LOCK = threading.Lock()


def _http_get_text(url: str, timeout: int = EVENT_HTTP_TIMEOUT_SEC) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (YEPSTOCKS/1.0)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    raw = _http_get_text(url, timeout=timeout)
    return json.loads(raw)


def _strip_tags(html_text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html_text)


def _month_num(name: str) -> int | None:
    lookup = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    return lookup.get(name.strip().lower().rstrip("."))


def _parse_date_flexible(value: str) -> date | None:
    cleaned = value.strip().replace("Sept.", "Sep.").replace("Sept", "Sep")
    for fmt in ("%b. %d, %Y", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _cache_get(key: str) -> list[dict[str, Any]] | None:
    now = time.time()
    with _EVENT_CACHE_LOCK:
        item = _EVENT_CACHE.get(key)
        if not item:
            return None
        ts, payload = item
        if now - ts > EVENT_CACHE_TTL_SEC:
            _EVENT_CACHE.pop(key, None)
            return None
        return payload


def _cache_set(key: str, payload: list[dict[str, Any]]) -> None:
    with _EVENT_CACHE_LOCK:
        _EVENT_CACHE[key] = (time.time(), payload)


def _chart_cache_get(key: str) -> dict[str, Any] | None:
    now = time.time()
    with _CHART_CACHE_LOCK:
        item = _CHART_CACHE.get(key)
        if not item:
            return None
        ts, payload = item
        if now - ts > CHART_CACHE_TTL_SEC:
            _CHART_CACHE.pop(key, None)
            return None
        return payload


def _chart_cache_set(key: str, payload: dict[str, Any]) -> None:
    with _CHART_CACHE_LOCK:
        _CHART_CACHE[key] = (time.time(), payload)


def _scan_snapshot_get(key: str) -> dict[str, Any] | None:
    now = time.time()
    with _SCAN_SNAPSHOT_LOCK:
        item = _SCAN_SNAPSHOT_CACHE.get(key)
        if not item:
            return None
        ts, payload = item
        if now - ts > SCAN_SNAPSHOT_TTL_SEC:
            _SCAN_SNAPSHOT_CACHE.pop(key, None)
            return None
        return payload


def _scan_snapshot_set(key: str, payload: dict[str, Any]) -> None:
    with _SCAN_SNAPSHOT_LOCK:
        _SCAN_SNAPSHOT_CACHE[key] = (time.time(), payload)


def _scan_snapshot_key(symbol: str, args: SimpleNamespace) -> str:
    source = str(getattr(args, "data_source", os.getenv("SCANNER_DATA_SOURCE", "auto"))).lower()
    intraday_interval = max(1, int(getattr(args, "intraday_interval_min", 1) or 1))
    return f"{symbol.upper()}:{source}:{intraday_interval}"


def _build_symbol_snapshot(symbol: str, args: SimpleNamespace, include_intraday: bool) -> dict[str, Any] | None:
    daily_candles = fetch_history(symbol)
    daily = analyze_candles(symbol, daily_candles, "1D")
    if not daily:
        return None

    snapshot: dict[str, Any] = {
        "symbol": symbol.upper(),
        "daily_candles": daily_candles,
        "daily": daily,
        "weekly": None,
        "intraday": [],
        "intraday_tfs": {},
        "intraday_ready": False,
    }

    weekly_candles = _resample_daily_to_weekly(daily_candles)
    snapshot["weekly"] = analyze_candles(symbol, weekly_candles, "1W") if weekly_candles else None

    if include_intraday:
        intraday_interval = max(1, int(getattr(args, "intraday_interval_min", 1) or 1))
        intraday = fetch_intraday(symbol, interval_min=intraday_interval)
        intraday_tfs: dict[int, Any | None] = {}
        for minutes in (233, 60, 55, 34, 21, 13, 8, 5):
            bars = resample_to_minutes(intraday, target_minutes=minutes)
            intraday_tfs[minutes] = analyze_candles(symbol, bars, f"{minutes}m") if bars else None
        snapshot["intraday"] = intraday
        snapshot["intraday_tfs"] = intraday_tfs
        snapshot["intraday_ready"] = True

    return snapshot


def _load_symbol_snapshot(symbol: str, args: SimpleNamespace, include_intraday: bool) -> dict[str, Any] | None:
    key = _scan_snapshot_key(symbol, args)
    cached = _scan_snapshot_get(key)
    if cached and (not include_intraday or bool(cached.get("intraday_ready", False))):
        return cached

    built = _build_symbol_snapshot(symbol, args, include_intraday)
    if built is None:
        return None
    _scan_snapshot_set(key, built)
    return built


def _fetch_fed_events(days_ahead: int = EVENT_LOOKAHEAD_DAYS) -> list[dict[str, Any]]:
    cache_key = f"fed:{days_ahead}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    events: list[dict[str, Any]] = []
    try:
        root_html = _http_get_text(FED_WHATS_NEXT_URL, timeout=8)
        links = re.findall(r'href="(/newsevents/20\d{2}[-/][a-z0-9]+\.htm)"', root_html, flags=re.I)
        month_links = sorted({urllib.parse.urljoin(FED_CALENDAR_ROOT, p) for p in links})[-FED_MONTH_PAGE_LIMIT:]
        for url in month_links:
            page = _http_get_text(url, timeout=8)
            if "FOMC Meetings" not in page:
                continue
            text = _strip_tags(unescape(page))
            text = " ".join(text.split())
            ym = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})", text)
            if not ym:
                continue
            month = _month_num(ym.group(1))
            year = int(ym.group(2))
            if not month:
                continue

            # Expected phrasing: "FOMC Meeting Two-day meeting, April 28 - 29 Press Conference 29"
            for m in re.finditer(
                r"FOMC Meeting.*?Two-day meeting,\s*([A-Za-z]+)\s*(\d{1,2})\s*-\s*(\d{1,2})",
                text,
                flags=re.I,
            ):
                m_month = _month_num(m.group(1)) or month
                end_day = int(m.group(3))
                try:
                    d = date(year, m_month, end_day)
                except ValueError:
                    continue
                if today <= d <= end_date:
                    events.append(
                        {
                            "date": d.isoformat(),
                            "label": "FOMC Decision",
                            "category": "macro",
                            "severity": "high",
                            "source": "federalreserve.gov",
                        }
                    )
    except Exception:
        pass

    # de-dup + sort
    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    for e in events:
        uniq[(e["date"], e["label"])] = e
    out = sorted(uniq.values(), key=lambda x: x["date"])
    _cache_set(cache_key, out)
    return out


def _fetch_labor_events(days_ahead: int = EVENT_LOOKAHEAD_DAYS) -> list[dict[str, Any]]:
    cache_key = f"labor:{days_ahead}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    events: list[dict[str, Any]] = []
    try:
        html_text = _http_get_text(BLS_EMPLOYMENT_SCHEDULE_URL, timeout=8)
        text = _strip_tags(unescape(html_text))
        text = " ".join(text.split())
        # Table contains: "Reference Month Release Date Release Time ... Jan. 09, 2026 08:30 AM"
        for m in re.finditer(r"([A-Za-z]{3,9}\.?\s+\d{1,2},\s+20\d{2})\s+08:30\s+AM", text):
            d = _parse_date_flexible(m.group(1))
            if not d:
                continue
            if today <= d <= end_date:
                events.append(
                    {
                        "date": d.isoformat(),
                        "label": "BLS Employment Situation",
                        "category": "macro",
                        "severity": "high",
                        "source": "bls.gov",
                    }
                )
    except Exception:
        pass

    out = sorted(events, key=lambda x: x["date"])
    _cache_set(cache_key, out)
    return out


def _fetch_symbol_earnings_event(symbol: str, days_ahead: int = EVENT_LOOKAHEAD_DAYS) -> list[dict[str, Any]]:
    cache_key = f"earn:{symbol.upper()}:{days_ahead}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    today = date.today()
    end_date = today + timedelta(days=days_ahead)
    events: list[dict[str, Any]] = []
    try:
        safe_symbol = urllib.parse.quote(symbol.upper().replace(".", "-"))
        url = f"{YAHOO_QUOTE_SUMMARY_URL}/{safe_symbol}?modules=calendarEvents"
        payload = _http_get_json(url, timeout=8)
        result = ((payload.get("quoteSummary", {}).get("result") or [None])[0] or {})
        earnings = ((result.get("calendarEvents") or {}).get("earnings") or {})
        earnings_dates = earnings.get("earningsDate") or []
        for item in earnings_dates:
            ts = item.get("raw")
            if ts is None:
                continue
            d = datetime.utcfromtimestamp(int(ts)).date()
            if today <= d <= end_date:
                events.append(
                    {
                        "date": d.isoformat(),
                        "label": f"{symbol.upper()} Earnings",
                        "category": "earnings",
                        "severity": "high",
                        "source": "yahoo",
                    }
                )
    except Exception:
        pass

    out = sorted(events, key=lambda x: x["date"])
    _cache_set(cache_key, out)
    return out


def _risk_events_for_symbol(symbol: str, chart_dates: list[str]) -> list[dict[str, Any]]:
    # Include recent and near-future events around the chart window.
    if chart_dates:
        start = datetime.strptime(chart_dates[0], "%Y-%m-%d").date() - timedelta(days=7)
        end = datetime.strptime(chart_dates[-1], "%Y-%m-%d").date() + timedelta(days=30)
    else:
        start = date.today() - timedelta(days=7)
        end = date.today() + timedelta(days=30)

    events = _fetch_fed_events(EVENT_LOOKAHEAD_DAYS) + _fetch_labor_events(EVENT_LOOKAHEAD_DAYS) + _fetch_symbol_earnings_event(symbol, EVENT_LOOKAHEAD_DAYS)
    filtered = [e for e in events if start <= datetime.strptime(e["date"], "%Y-%m-%d").date() <= end]
    filtered.sort(key=lambda x: x["date"])
    return filtered


def _upcoming_events_for_symbol(
    symbol: str, days_ahead: int = EVENT_BLOCK_DAYS, macro_events: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    today = date.today()
    end = today + timedelta(days=max(1, int(days_ahead)))
    macro = macro_events if macro_events is not None else (_fetch_fed_events(days_ahead) + _fetch_labor_events(days_ahead))
    combined = list(macro) + _fetch_symbol_earnings_event(symbol, days_ahead)
    out: list[dict[str, Any]] = []
    for event in combined:
        raw = str(event.get("date", "")).strip()
        try:
            event_date = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        if today <= event_date <= end:
            out.append(event)
    out.sort(key=lambda e: (e.get("date", ""), e.get("label", "")))
    return out


def _safe_num(value: Any, ndigits: int = 3) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number) or math.isinf(number):
        return 0.0
    return round(number, ndigits)


def _as_float(form: Any, key: str, default: float) -> float:
    raw = str(form.get(key, "")).strip()
    if not raw:
        return default
    return float(raw)


def _as_int(form: Any, key: str, default: int) -> int:
    raw = str(form.get(key, "")).strip()
    if not raw:
        return default
    return int(raw)


def _build_args(form: Any) -> SimpleNamespace:
    return SimpleNamespace(
        min_price=_as_float(form, "min_price", 5.0),
        max_price=_as_float(form, "max_price", 1000.0),
        min_dollar_volume=_as_float(form, "min_dollar_volume", 2_000_000.0),
        max_rsi=_as_float(form, "max_rsi", 80.0),
        max_stoch_rsi_k=_as_float(form, "max_stoch_rsi_k", 95.0),
        min_adx=_as_float(form, "min_adx", 10.0),
        require_uptrend=(form.get("require_uptrend") == "on"),
        require_breakout=(form.get("require_breakout") == "on"),
        require_macd_bull=(form.get("require_macd_bull") == "on"),
        require_di_bull=(form.get("require_di_bull") == "on"),
        require_macd_stoch_cross=(form.get("require_macd_stoch_cross") == "on"),
        require_simultaneous_cross=(form.get("require_simultaneous_cross") == "on"),
        require_band_liftoff=(form.get("require_band_liftoff") == "on"),
        bb_spread_watchlist=(form.get("bb_spread_watchlist") == "on"),
        signal_direction="both",
        cross_lookback=_as_int(form, "cross_lookback", 4),
        max_macd_cross_age=_as_int(form, "max_macd_cross_age", 3),
        max_stoch_cross_age=_as_int(form, "max_stoch_cross_age", 3),
        band_touch_lookback=_as_int(form, "band_touch_lookback", 6),
        min_band_expansion=_as_float(form, "min_band_expansion", 0.05),
        min_course_pattern_score=_as_float(form, "min_course_pattern_score", 55.0),
        max_setup_age=_as_int(form, "max_setup_age", 3),
        require_daily_and_233=(form.get("require_daily_and_233") == "on"),
        intraday_interval_min=5,
        auto_fallback=(form.get("auto_fallback") == "on"),
        no_skips=(form.get("no_skips") == "on"),
        max_retries=_as_int(form, "max_retries", 4),
        data_source=str(form.get("data_source", "auto")),
        polygon_api_key=str(form.get("polygon_api_key", "")),
        pows=(form.get("pows") == "on"),
        verbose=False,
    )


def _to_json_series(values: list[float]) -> list[float | None]:
    return [None if (isinstance(v, float) and math.isnan(v)) else float(v) for v in values]


def _chart_from_candles(symbol: str, timeframe: str, candles: list[Candle]) -> dict[str, Any] | None:
    if len(candles) < 35:
        return None

    if timeframe == "1D":
        dates = [c.date.strftime("%Y-%m-%d") for c in candles]
    else:
        dates = [c.date.strftime("%Y-%m-%d %H:%M") for c in candles]

    opens = [c.open for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    closes = [c.close for c in candles]

    sma50 = sma_series(closes, 50)
    sma89 = sma_series(closes, 89)
    sma200 = sma_series(closes, 200)
    bb_mid = sma_series(closes, POWS_BB_LENGTH)
    bb_std = rolling_stdev(closes, POWS_BB_LENGTH)
    bb_upper = [m + POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid, bb_std)]
    bb_lower = [m - POWS_BB_STDDEV * s if not (math.isnan(m) or math.isnan(s)) else math.nan for m, s in zip(bb_mid, bb_std)]
    stoch_k, stoch_d = stoch_rsi_series(closes, POWS_RSI_LENGTH, 21, 3, 5)
    macd_line, macd_signal = macd_series(closes, POWS_MACD_FAST, POWS_MACD_SLOW, POWS_MACD_SIGNAL)

    sma_up, sma_down = find_crossings(sma50, sma89)
    macd_up, macd_down = find_crossings(macd_line, macd_signal)
    stoch_up, stoch_down = find_crossings(stoch_k, stoch_d)

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "dates": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "sma50": _to_json_series(sma50),
        "sma89": _to_json_series(sma89),
        "sma200": _to_json_series(sma200),
        "bb_upper": _to_json_series(bb_upper),
        "bb_lower": _to_json_series(bb_lower),
        "macd": _to_json_series(macd_line),
        "macd_signal": _to_json_series(macd_signal),
        "stoch_k": _to_json_series(stoch_k),
        "stoch_d": _to_json_series(stoch_d),
        "sma_cross_up_idx": sma_up,
        "sma_cross_down_idx": sma_down,
        "macd_cross_up_idx": macd_up,
        "macd_cross_down_idx": macd_down,
        "stoch_cross_up_idx": stoch_up,
        "stoch_cross_down_idx": stoch_down,
    }


def _result_row(analyzed: Any) -> dict[str, Any]:
    raw_setup_score = float(getattr(analyzed, "options_setup_score", 0.0) or 0.0)
    raw_score = float(getattr(analyzed, "score", 0.0) or 0.0)
    raw_pre3x = float(max(getattr(analyzed, "pre3x_bull_score", 0.0), getattr(analyzed, "pre3x_bear_score", 0.0)))
    raw_hist_wr = float(getattr(analyzed, "hist_win_rate_5y", 0.0) or 0.0)
    raw_bb_expansion = float(getattr(analyzed, "band_width_expansion", 0.0) or 0.0)
    raw_pattern_similarity = float(getattr(analyzed, "pattern_similarity", 0.0) or 0.0)
    raw_pattern_direct = float(getattr(analyzed, "pattern_similarity_direct", 0.0) or 0.0)
    raw_course_score = float(getattr(analyzed, "course_pattern_score", 0.0) or 0.0)
    raw_lifecycle_score = float(getattr(analyzed, "lifecycle_score", 0.0) or 0.0)
    rank_score = (
        (raw_setup_score * 1000.0)
        + (raw_score * 10.0)
        + (raw_pre3x * 2.0)
        + (raw_hist_wr * 100.0)
        + (max(0.0, raw_bb_expansion) * 300.0)
        + (raw_pattern_similarity * PATTERN_REF_WEIGHT)
        + (raw_course_score * 20.0)
        + (raw_lifecycle_score * 40.0)
    )
    return {
        "symbol": analyzed.symbol,
        "dir": str(getattr(analyzed, "setup_direction", "n/a")).upper(),
        "setup": str(getattr(analyzed, "setup_type", "n/a")),
        "stage": str(getattr(analyzed, "lifecycle_stage", "P1 Pre-Setup")),
        "phase": int(getattr(analyzed, "lifecycle_phase", 1)),
        "close": _safe_num(analyzed.close, 2),
        "sma50": _safe_num(analyzed.sma50, 2),
        "sma89": _safe_num(analyzed.sma89, 2),
        "sma200": _safe_num(analyzed.sma200, 2),
        "stoch_k": _safe_num(analyzed.stoch_rsi_k, 2),
        "macd_hist": _safe_num(analyzed.macd - analyzed.macd_signal, 3),
        "adx": _safe_num(analyzed.adx14, 2),
        "macd_cross_age": analyzed.macd_cross_age,
        "stoch_cross_age": analyzed.stoch_cross_age,
        "macd_bear_cross_age": analyzed.macd_bear_cross_age,
        "stoch_bear_cross_age": analyzed.stoch_bear_cross_age,
        "bb_expansion": _safe_num(analyzed.band_width_expansion, 3),
        "bb_width_now": _safe_num(getattr(analyzed, "band_width_now", 0.0), 3),
        "bb_width_pct": _safe_num(getattr(analyzed, "band_width_percentile", 0.0) * 100.0, 1),
        "bb_slope3": _safe_num(getattr(analyzed, "band_width_slope3", 0.0), 4),
        "bb_osc": _safe_num(getattr(analyzed, "band_oscillation_score", 0.0), 3),
        "bb_regime": bool(getattr(analyzed, "band_widening_regime", False)),
        "bb_prev_month_avg": _safe_num(getattr(analyzed, "band_width_prev_month_avg", 0.0), 3),
        "bb_vs_prev_month": _safe_num(getattr(analyzed, "band_width_vs_prev_month", 0.0), 2),
        "bb_double_age": int(getattr(analyzed, "band_width_double_age", 999)),
        "bb_double_recent_ok": bool(getattr(analyzed, "band_width_double_recent_ok", False)),
        "band_ride_score": _safe_num(
            getattr(analyzed, "bull_band_ride_score", 0.0)
            if str(getattr(analyzed, "setup_direction", "bull")).lower() == "bull"
            else getattr(analyzed, "bear_band_ride_score", 0.0),
            3,
        ),
        "bb_widen_start_age": int(getattr(analyzed, "band_widen_start_age", 999)),
        "bb_widen_window_ok": bool(getattr(analyzed, "band_widen_window_ok", False)),
        "pre3x": _safe_num(max(analyzed.pre3x_bull_score, analyzed.pre3x_bear_score), 2),
        "hist_n": int(getattr(analyzed, "hist_setups_5y", 0)),
        "hist_wr": _safe_num(getattr(analyzed, "hist_win_rate_5y", 0.0) * 100.0, 1),
        "hist_avg": _safe_num(getattr(analyzed, "hist_avg_return_5y", 0.0) * 100.0, 2),
        "tgt_mid": _safe_num(getattr(analyzed, "target_mid_pct", 0.0) * 100.0, 2),
        "tgt_band": _safe_num(getattr(analyzed, "target_band_pct", 0.0) * 100.0, 2),
        "risk": _safe_num(getattr(analyzed, "risk_pct", 0.0) * 100.0, 2),
        "rr": _safe_num(getattr(analyzed, "rr_mid", 0.0), 2),
        "setup_score": _safe_num(raw_setup_score, 1),
        "lifecycle_score": _safe_num(raw_lifecycle_score, 1),
        "course_score": _safe_num(raw_course_score, 1),
        "dollar_volume": format_money(analyzed.dollar_volume20),
        "score": _safe_num(raw_score, 3),
        "raw_setup_score": raw_setup_score,
        "raw_score": raw_score,
        "raw_pre3x": raw_pre3x,
        "raw_hist_wr": raw_hist_wr,
        "raw_bb_expansion": raw_bb_expansion,
        "raw_course_score": raw_course_score,
        "raw_lifecycle_score": raw_lifecycle_score,
        "pattern_similarity": _safe_num(raw_pattern_similarity * 100.0, 1),
        "raw_pattern_similarity": raw_pattern_similarity,
        "pattern_mode": str(getattr(analyzed, "setup_direction", "n/a")).upper(),
        "pattern_direct": _safe_num(raw_pattern_direct * 100.0, 1),
        "rank_score": rank_score,
    }


def _strategy_args() -> SimpleNamespace:
    """Locked scanner profile: no UI tuning, fixed strategy rules."""
    polygon_key = str(os.getenv("POLYGON_API_KEY", "")).strip()
    default_source = str(os.getenv("SCANNER_DATA_SOURCE", "polygon" if polygon_key else "auto")).strip().lower() or "auto"
    return SimpleNamespace(
        min_price=5.0,
        max_price=1000.0,
        min_dollar_volume=2_000_000.0,
        max_rsi=80.0,
        max_stoch_rsi_k=80.0,
        min_adx=8.0,
        require_uptrend=False,
        require_breakout=False,
        require_macd_bull=False,
        require_di_bull=False,
        require_macd_stoch_cross=True,
        require_simultaneous_cross=False,
        require_band_liftoff=True,
        bb_spread_watchlist=True,
        signal_direction="both",
        cross_lookback=6,
        max_macd_cross_age=3,
        max_stoch_cross_age=3,
        triple_gap_max=6,
        band_touch_lookback=8,
        min_band_expansion=0.03,
        min_band_width_now=0.08,
        min_band_width_percentile=0.60,
        min_band_slope3=0.0,
        require_widening_oscillation=True,
        min_band_oscillation_score=0.35,
        min_band_vs_prev_month=2.0,
        max_band_double_age=14,
        require_band_ride=True,
        min_band_ride_score=0.62,
        band_ride_lookback=8,
        require_band_widen_window=True,
        recent_daily_bars=15,
        recent_233_bars=15,
        min_target_band_pct=0.01,
        min_course_pattern_score=55.0,
        max_setup_age=3,
        require_daily_and_233=True,
        require_hourly=True,
        require_weekly_context=True,
        require_precision_entry=True,
        intraday_interval_min=1,
        auto_fallback=True,
        no_skips=False,
        max_retries=4,
        data_source=default_source,
        polygon_api_key=polygon_key,
        pows=True,
        verbose=False,
    )


def _strict_fallback_tiers(base_args: SimpleNamespace) -> list[tuple[str, SimpleNamespace]]:
    strict = copy.deepcopy(base_args)
    strict.pows = True
    strict.require_macd_stoch_cross = True
    strict.require_band_liftoff = True
    strict.bb_spread_watchlist = False
    strict.require_simultaneous_cross = False
    strict.require_daily_and_233 = True
    strict.require_hourly = True
    strict.require_weekly_context = True
    strict.require_precision_entry = True
    strict.cross_lookback = min(strict.cross_lookback, 5)
    strict.triple_gap_max = 4
    strict.band_touch_lookback = min(strict.band_touch_lookback, 8)
    strict.min_band_expansion = max(strict.min_band_expansion, 0.03)
    strict.min_adx = max(strict.min_adx, 10.0)

    confirm = copy.deepcopy(strict)
    confirm.require_simultaneous_cross = False
    confirm.require_daily_and_233 = True
    confirm.require_hourly = True
    confirm.require_weekly_context = True
    confirm.require_precision_entry = True
    confirm.require_band_liftoff = False
    confirm.bb_spread_watchlist = True
    confirm.cross_lookback = max(confirm.cross_lookback, 7)
    confirm.triple_gap_max = 6
    confirm.band_touch_lookback = max(confirm.band_touch_lookback, 10)
    confirm.min_band_expansion = min(confirm.min_band_expansion, 0.01)
    confirm.min_adx = min(confirm.min_adx, 8.0)

    developing = copy.deepcopy(confirm)
    developing.pows = True
    developing.require_macd_stoch_cross = False
    developing.require_band_liftoff = False
    developing.bb_spread_watchlist = True
    developing.cross_lookback = max(developing.cross_lookback, 10)
    developing.triple_gap_max = 8
    developing.band_touch_lookback = max(developing.band_touch_lookback, 12)
    developing.min_band_expansion = min(developing.min_band_expansion, 0.0)
    developing.min_adx = min(developing.min_adx, 6.0)
    developing.require_daily_and_233 = True
    developing.require_hourly = True
    developing.require_weekly_context = True
    developing.require_precision_entry = False

    watchlist = copy.deepcopy(developing)
    watchlist.require_daily_and_233 = False
    watchlist.require_hourly = False
    watchlist.require_weekly_context = False
    watchlist.require_precision_entry = False
    watchlist.require_macd_stoch_cross = False
    watchlist.require_band_liftoff = False
    watchlist.bb_spread_watchlist = True
    watchlist.cross_lookback = max(watchlist.cross_lookback, 12)
    watchlist.triple_gap_max = 10
    watchlist.band_touch_lookback = max(watchlist.band_touch_lookback, 14)
    watchlist.min_band_expansion = min(watchlist.min_band_expansion, -0.02)
    watchlist.min_adx = min(watchlist.min_adx, 4.0)
    watchlist.max_rsi = max(watchlist.max_rsi, 90.0)
    watchlist.max_stoch_rsi_k = max(watchlist.max_stoch_rsi_k, 95.0)

    broad = copy.deepcopy(watchlist)
    broad.require_daily_and_233 = False
    broad.require_hourly = False
    broad.require_weekly_context = False
    broad.require_precision_entry = False
    broad.require_macd_stoch_cross = False
    broad.require_band_liftoff = False
    broad.bb_spread_watchlist = True
    broad.cross_lookback = max(broad.cross_lookback, 14)
    broad.triple_gap_max = 12
    broad.min_adx = min(broad.min_adx, 3.0)
    broad.min_dollar_volume = min(broad.min_dollar_volume, 1_000_000.0)
    broad.max_rsi = max(broad.max_rsi, 95.0)
    broad.max_stoch_rsi_k = max(broad.max_stoch_rsi_k, 95.0)

    return [("strict", strict), ("confirm", confirm), ("developing", developing), ("watchlist", watchlist), ("broad_3x", broad)]


def _ranked_scan_args() -> SimpleNamespace:
    """Single-pass ranked scan over all symbols with hourly-only setup confirmation."""
    args = _strategy_args()
    args.require_macd_stoch_cross = False
    args.require_band_liftoff = False
    args.bb_spread_watchlist = True
    args.require_daily_and_233 = False
    args.require_hourly = True
    args.require_weekly_context = False
    args.require_precision_entry = False
    args.require_secondary_confirmation = True
    args.scan_intraday_3x = False
    args.max_macd_cross_age = min(int(getattr(args, "max_macd_cross_age", 3)), 3)
    args.max_stoch_cross_age = min(int(getattr(args, "max_stoch_cross_age", 3)), 3)
    args.cross_lookback = max(args.cross_lookback, 14)
    args.triple_gap_max = 12
    args.min_adx = min(args.min_adx, 3.0)
    args.max_rsi = max(args.max_rsi, 95.0)
    args.max_stoch_rsi_k = max(args.max_stoch_rsi_k, 95.0)
    args.min_dollar_volume = min(args.min_dollar_volume, 1_000_000.0)
    args.min_band_expansion = max(args.min_band_expansion, 0.10)
    # Align with "within 2 weeks" BB timing rule.
    args.band_touch_lookback = max(args.band_touch_lookback, 14)
    args.min_course_pattern_score = max(float(getattr(args, "min_course_pattern_score", 55.0)), 62.0)
    return args


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _sim_by_distance(a: float, b: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    if any(math.isnan(v) or math.isinf(v) for v in (a, b)):
        return 0.0
    return _clamp01(1.0 - (abs(a - b) / scale))


def _build_reference_pattern(symbol: str = "GOOGL") -> Any | None:
    candles = fetch_history(symbol)
    if len(candles) < 120:
        return None
    return analyze_candles(symbol, candles, "1D")


def _pattern_similarity(reference: Any, candidate: Any) -> float:
    if not reference or not candidate:
        return 0.0
    ref_macd_hist = float(reference.macd - reference.macd_signal)
    cand_macd_hist = float(candidate.macd - candidate.macd_signal)
    ref_pre3x = float(max(reference.pre3x_bull_score, reference.pre3x_bear_score))
    cand_pre3x = float(max(candidate.pre3x_bull_score, candidate.pre3x_bear_score))
    ref_direction = str(getattr(reference, "setup_direction", ""))
    cand_direction = str(getattr(candidate, "setup_direction", ""))

    similarity = (
        0.28 * _sim_by_distance(float(reference.band_width_expansion), float(candidate.band_width_expansion), 0.18)
        + 0.18 * _sim_by_distance(ref_pre3x, cand_pre3x, 3.0)
        + 0.16 * _sim_by_distance(ref_macd_hist, cand_macd_hist, 0.45)
        + 0.12 * _sim_by_distance(float(reference.stoch_rsi_k), float(candidate.stoch_rsi_k), 26.0)
        + 0.12 * _sim_by_distance(float(reference.adx14), float(candidate.adx14), 14.0)
        + 0.10 * _sim_by_distance(float(reference.target_mid_pct), float(candidate.target_mid_pct), 0.03)
        + 0.04 * (1.0 if ref_direction and ref_direction == cand_direction else 0.0)
    )
    return _clamp01(similarity)


def _resample_daily_to_weekly(candles: list[Candle]) -> list[Candle]:
    if not candles:
        return []
    out: list[Candle] = []
    bucket: list[Candle] = []
    current_key: tuple[int, int] | None = None
    for c in candles:
        key = c.date.isocalendar()[:2]
        if current_key is None:
            current_key = key
        if key != current_key and bucket:
            out.append(
                Candle(
                    date=bucket[-1].date,
                    open=bucket[0].open,
                    high=max(x.high for x in bucket),
                    low=min(x.low for x in bucket),
                    close=bucket[-1].close,
                    volume=sum(x.volume for x in bucket),
                )
            )
            bucket = []
            current_key = key
        bucket.append(c)
    if bucket:
        out.append(
            Candle(
                date=bucket[-1].date,
                open=bucket[0].open,
                high=max(x.high for x in bucket),
                low=min(x.low for x in bucket),
                close=bucket[-1].close,
                volume=sum(x.volume for x in bucket),
            )
        )
    return out


def _weekly_context_ok(daily: Any, weekly: Any) -> bool:
    if not weekly:
        return False
    direction = str(getattr(daily, "setup_direction", "bull")).lower()
    if direction == "bear":
        return bool(weekly.close <= weekly.sma50 and weekly.sma50 <= weekly.sma89)
    return bool(weekly.close >= weekly.sma50 and weekly.sma50 >= weekly.sma89)


def _has_triple_cross(result: Any, lookback: int, gap_max: int) -> bool:
    direction = str(getattr(result, "setup_direction", "bull")).lower()
    if direction == "bear":
        return bool(
            result.price_bear_cross_age <= lookback
            and result.macd_bear_cross_age <= lookback
            and result.stoch_bear_cross_age <= lookback
            and result.triple_bear_cross_gap <= gap_max
        )
    return bool(
        result.price_cross_age <= lookback
        and result.macd_cross_age <= lookback
        and result.stoch_cross_age <= lookback
        and result.triple_cross_gap <= gap_max
    )


def _recent_lifecycle_window_ok(result: Any, window_bars: int) -> bool:
    if not result:
        return False
    w = max(1, int(window_bars))
    direction = str(getattr(result, "setup_direction", "bull")).lower()
    touch_age = int(getattr(result, "outer_touch_age", 999))
    dbl_age = int(getattr(result, "band_width_double_age", 999))
    widen_age = int(getattr(result, "band_widen_start_age", 999))
    macd_gap_now = float(getattr(result, "macd_gap_now", 0.0) or 0.0)
    macd_signal = float(getattr(result, "macd_signal", 0.0) or 0.0)
    stoch_gap_now = float(getattr(result, "stoch_gap_now", 0.0) or 0.0)
    band_width_slope3 = float(getattr(result, "band_width_slope3", 0.0) or 0.0)
    ride_score = (
        float(getattr(result, "bear_band_ride_score", 0.0) or 0.0)
        if direction == "bear"
        else float(getattr(result, "bull_band_ride_score", 0.0) or 0.0)
    )
    if direction == "bear":
        cross_age = min(int(getattr(result, "macd_bear_cross_age", 999)), int(getattr(result, "stoch_bear_cross_age", 999)))
    else:
        cross_age = min(int(getattr(result, "macd_cross_age", 999)), int(getattr(result, "stoch_cross_age", 999)))
    macd_near = abs(macd_gap_now) <= max(0.08, abs(macd_signal) * 0.5)
    stoch_near = abs(stoch_gap_now) <= 6.0
    momentum_recent = (cross_age <= w) or macd_near or stoch_near
    band_recent = (
        (touch_age <= w and (ride_score >= 0.40 or band_width_slope3 >= -0.002))
        or (dbl_age <= w)
        or (widen_age <= w)
    )
    # New setup rule: recent band interaction + recent/near momentum, not historical matches.
    return bool(band_recent and momentum_recent)


def _passes_watchlist_loose(result: Any, args: SimpleNamespace) -> bool:
    """Fallback inclusion path for live scans when strict filters are too sparse."""
    if not result:
        return False
    close = float(getattr(result, "close", 0.0) or 0.0)
    dollar_volume = float(getattr(result, "dollar_volume20", 0.0) or 0.0)
    if close <= 0 or not (float(getattr(args, "min_price", 5.0)) <= close <= float(getattr(args, "max_price", 1000.0))):
        return False
    if dollar_volume < min(float(getattr(args, "min_dollar_volume", 2_000_000.0)), 800_000.0):
        return False

    recent_daily = int(getattr(args, "recent_daily_bars", 30) or 30)
    if not _recent_lifecycle_window_ok(result, recent_daily):
        return False

    direction = str(getattr(result, "setup_direction", "bull")).lower()
    if direction == "bear":
        cross_age = min(int(getattr(result, "macd_bear_cross_age", 999)), int(getattr(result, "stoch_bear_cross_age", 999)))
    else:
        cross_age = min(int(getattr(result, "macd_cross_age", 999)), int(getattr(result, "stoch_cross_age", 999)))
    near_cross = abs(float(getattr(result, "macd_gap_now", 0.0) or 0.0)) <= 0.15 or abs(float(getattr(result, "stoch_gap_now", 0.0) or 0.0)) <= 10.0
    band_recent = int(getattr(result, "outer_touch_age", 999)) <= recent_daily
    return bool(cross_age <= recent_daily or near_cross or band_recent)


def _passes_candidate_floor(result: Any, args: SimpleNamespace) -> bool:
    """Last-resort floor so scanner can still surface ranked candidates."""
    if not result:
        return False
    close = float(getattr(result, "close", 0.0) or 0.0)
    if close <= 0:
        return False
    if not (float(getattr(args, "min_price", 5.0)) <= close <= float(getattr(args, "max_price", 1000.0))):
        return False
    if float(getattr(result, "dollar_volume20", 0.0) or 0.0) < 250_000.0:
        return False
    # In force-candidate mode, ranking handles quality; floor only enforces tradable liquidity.
    if bool(getattr(args, "force_candidate_mode", False)):
        return True
    recent = int(getattr(args, "recent_daily_bars", 30) or 30)
    if _recent_lifecycle_window_ok(result, recent):
        return True
    direction = str(getattr(result, "setup_direction", "bull")).lower()
    if direction == "bear":
        cross_age = min(int(getattr(result, "macd_bear_cross_age", 999)), int(getattr(result, "stoch_bear_cross_age", 999)))
    else:
        cross_age = min(int(getattr(result, "macd_cross_age", 999)), int(getattr(result, "stoch_cross_age", 999)))
    near_cross = abs(float(getattr(result, "macd_gap_now", 0.0) or 0.0)) <= 0.25 or abs(float(getattr(result, "stoch_gap_now", 0.0) or 0.0)) <= 14.0
    return bool(cross_age <= max(8, recent) or near_cross)


def _precision_entry_ok(intraday_tfs: dict[int, Any | None], daily: Any, args: SimpleNamespace) -> bool:
    # Course precision ladder: 21 (daily), 13 (233), 8/5 (55/34 style intraday refinement).
    direction = str(getattr(daily, "setup_direction", "bull")).lower()
    lookback = max(6, int(getattr(args, "cross_lookback", 6)))
    for minutes in (21, 13, 8, 5):
        tf = intraday_tfs.get(minutes)
        if not tf:
            continue
        tf_dir = str(getattr(tf, "setup_direction", "")).lower()
        if tf_dir and tf_dir != direction:
            continue
        if direction == "bear":
            if tf.macd_bear_cross_age <= lookback and tf.stoch_bear_cross_age <= lookback:
                return True
        else:
            if tf.macd_cross_age <= lookback and tf.stoch_cross_age <= lookback:
                return True
    return False


def _analyze_for_args(symbol: str, args: SimpleNamespace) -> Any | None:
    retries = 2
    for attempt in range(retries + 1):
        try:
            scan_intraday_3x = bool(getattr(args, "scan_intraday_3x", False))
            need_intraday = bool(
                args.require_daily_and_233
                or getattr(args, "require_hourly", False)
                or getattr(args, "require_precision_entry", False)
                or scan_intraday_3x
            )
            snapshot = _load_symbol_snapshot(symbol, args, include_intraday=need_intraday)
            if not snapshot:
                return None
            daily_candles = snapshot.get("daily_candles") or []
            daily_base = snapshot.get("daily")
            if not daily_base:
                return None
            daily = copy.deepcopy(daily_base)
            if not daily:
                return None
            strict_ok = passes_filters(daily, args)
            loose_ok = bool(getattr(args, "allow_momentum_watchlist", False)) and _passes_watchlist_loose(daily, args)
            floor_ok = bool(getattr(args, "force_candidate_mode", False)) and _passes_candidate_floor(daily, args)
            if not (strict_ok or loose_ok or floor_ok):
                return None
            lookback = max(4, int(getattr(args, "cross_lookback", 6)))
            gap_max = max(2, int(getattr(args, "triple_gap_max", 4)))
            daily_has_3x = _has_triple_cross(daily, lookback, gap_max)
            tf233_has_3x = False
            tf55_has_3x = False
            tf34_has_3x = False
            daily_window_bars = int(getattr(args, "recent_daily_bars", 15) or 15)
            tf233_window_bars = int(getattr(args, "recent_233_bars", 15) or 15)
            daily_recent_ok = _recent_lifecycle_window_ok(daily, daily_window_bars)
            tf233_recent_ok = False

            need_secondary_confirmation = bool(
                getattr(args, "require_secondary_confirmation", True)
                and (args.require_daily_and_233 or getattr(args, "require_hourly", False))
            )

            if need_intraday:
                intraday_tfs: dict[int, Any | None] = snapshot.get("intraday_tfs") or {}
                tf233 = None
                secondary_pass = False

                if args.require_daily_and_233:
                    tf233 = intraday_tfs.get(233)
                    tf233_recent_ok = _recent_lifecycle_window_ok(tf233, tf233_window_bars)
                    if tf233 and passes_secondary_timeframe_filters(tf233, args):
                        secondary_pass = True

                if getattr(args, "require_hourly", False):
                    tf60 = intraday_tfs.get(60)
                    if tf60 and passes_secondary_timeframe_filters(tf60, args):
                        secondary_pass = True

                tf233_has_3x = bool(tf233 and _has_triple_cross(tf233, lookback, gap_max))

                # Compute 55/34 when needed for intraday 3x discovery, or if daily/233 didn't satisfy 3x.
                if scan_intraday_3x or not (daily_has_3x or tf233_has_3x):
                    tf55 = intraday_tfs.get(55)
                    tf34 = intraday_tfs.get(34)
                    # Allow both with-trend and counter-trend 55/34 candidates; ranking handles quality.
                    tf55_has_3x = bool(tf55 and _has_triple_cross(tf55, lookback, gap_max))
                    tf34_has_3x = bool(tf34 and _has_triple_cross(tf34, lookback, gap_max))

                # Timeframe-to-trend mapping from the course:
                # 3x on daily/233 => confirm trend with weekly.
                if getattr(args, "require_weekly_context", False) and (daily_has_3x or tf233_has_3x):
                    weekly = snapshot.get("weekly")
                    if not _weekly_context_ok(daily, weekly):
                        return None

                # 3x on 55/34 => use daily for trend (enforced by direction match above).

                if need_secondary_confirmation and not secondary_pass:
                    return None

                if getattr(args, "require_precision_entry", False):
                    if not _precision_entry_ok(intraday_tfs, daily, args):
                        return None

            enforce_recent_gate = bool(getattr(args, "enforce_recent_lifecycle_gate", True))
            enforce_3x_gate = bool(getattr(args, "enforce_three_x_gate", True))
            if enforce_recent_gate and not (daily_recent_ok or tf233_recent_ok):
                return None
            if enforce_3x_gate and not (
                daily_has_3x or tf233_has_3x or tf55_has_3x or tf34_has_3x or daily_recent_ok or tf233_recent_ok
            ):
                return None
            return daily
        except RuntimeError:
            if attempt >= retries:
                raise
            continue

    return None


def build_chart_payload(symbol: str, days: int, requested_timeframes: list[str] | None = None) -> dict[str, Any]:
    if requested_timeframes:
        normalized = sorted({str(tf).strip() for tf in requested_timeframes if str(tf).strip()})
    else:
        normalized = []
    tf_key = ",".join(normalized) if normalized else "all"
    cache_key = f"{symbol.upper()}:{int(days)}:{tf_key}"
    cached = _chart_cache_get(cache_key)
    if cached is not None:
        return cached

    daily_candles = fetch_history(symbol)
    lookback = max(60, days)
    daily_candles = daily_candles[-lookback:]
    if len(daily_candles) < 35:
        raise RuntimeError(f"Not enough data to chart {symbol}")

    timeframes: dict[str, dict[str, Any]] = {}
    include_daily = (not normalized) or ("1D" in normalized)
    if include_daily:
        daily_chart = _chart_from_candles(symbol, "1D", daily_candles)
        if daily_chart:
            timeframes["1D"] = daily_chart

    intraday_keep = {60: 120}
    intraday_targets: list[int]
    if normalized:
        intraday_targets = []
        for tf in normalized:
            if tf.endswith("m"):
                try:
                    minutes = int(tf[:-1])
                except ValueError:
                    continue
                if minutes in intraday_keep:
                    intraday_targets.append(minutes)
    else:
        intraday_targets = [60]
    try:
        if intraday_targets:
            intraday = fetch_intraday(symbol, interval_min=1)
        else:
            intraday = []
        for minutes in intraday_targets:
            bars = resample_to_minutes(intraday, target_minutes=minutes)
            keep = intraday_keep.get(minutes, 120)
            if len(bars) > keep:
                bars = bars[-keep:]
            tf_chart = _chart_from_candles(symbol, f"{minutes}m", bars)
            if tf_chart:
                timeframes[f"{minutes}m"] = tf_chart
    except Exception:
        pass

    if not timeframes:
        raise RuntimeError(f"Not enough data to chart {symbol}")

    daily_dates = timeframes.get("1D", {}).get("dates", [])
    events = _risk_events_for_symbol(symbol, daily_dates)
    for chart in timeframes.values():
        chart["events"] = events

    payload = {"symbol": symbol.upper(), "timeframes": timeframes}
    _chart_cache_set(cache_key, payload)
    return payload


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    fallback_error = ""
    if request.method == "POST":
        fallback_error = "Browser script did not start scan. Please reload and try again."
    return render_template("index.html", error=fallback_error, results=[], chart_payload=None, form_values={})


@app.route("/healthz", methods=["GET", "HEAD"])
def healthz() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.route("/scan_stream", methods=["POST"])
def scan_stream() -> Response:
    workers = max(1, min(8, int(os.getenv("SCAN_WORKERS", "4"))))
    plot_days = 260

    try:
        symbols = list(dict.fromkeys(fetch_sp500_symbols() + ["QQQ"]))
    except Exception as exc:
        payload = json.dumps({"type": "error", "message": f"Universe source unavailable (S&P 500 + QQQ): {exc}"})
        return Response(payload + "\n", mimetype="application/x-ndjson")
    else:
        fallback_msg = ""

    base_args = _ranked_scan_args()
    configure_data_source(base_args.data_source, base_args.polygon_api_key)
    max_retries = min(1, max(0, int(base_args.max_retries)))
    output_limit = max(1, int(os.getenv("OUTPUT_LIMIT", "30")))
    prefilter_limit = max(output_limit * 3, int(os.getenv("PREFILTER_LIMIT", "180")))
    confirmation_limit = max(output_limit, int(os.getenv("CONFIRMATION_LIMIT", str(max(output_limit * 2, 60)))))
    max_tune_steps = max(1, int(os.getenv("MAX_TUNE_STEPS", "6")))
    max_scan_seconds = max(60, int(os.getenv("MAX_SCAN_SECONDS", "420")))

    def _tuned_args(step: int) -> SimpleNamespace:
        a = copy.deepcopy(base_args)
        # Keep scan responsive: hourly-only confirmation.
        a.require_daily_and_233 = False
        a.require_hourly = True
        a.require_secondary_confirmation = True
        a.require_precision_entry = False
        a.scan_intraday_3x = False
        a.enforce_recent_lifecycle_gate = True
        a.enforce_three_x_gate = True
        # Step 0: strict baseline.
        if step >= 1:
            a.allow_momentum_watchlist = True
            a.cross_lookback = max(int(getattr(a, "cross_lookback", 10)), 10)
            a.recent_daily_bars = max(int(getattr(a, "recent_daily_bars", 15)), 20)
            a.recent_233_bars = max(int(getattr(a, "recent_233_bars", 15)), 20)
            a.enforce_recent_lifecycle_gate = False
        if step >= 2:
            a.require_band_widen_window = False
            a.require_widening_oscillation = False
            a.min_course_pattern_score = max(25.0, float(getattr(a, "min_course_pattern_score", 62.0)) - 20.0)
            a.force_candidate_mode = True
        if step >= 3:
            a.require_band_ride = False
            a.min_band_vs_prev_month = min(float(getattr(a, "min_band_vs_prev_month", 2.0)), 1.4)
            a.max_band_double_age = max(int(getattr(a, "max_band_double_age", 14)), 20)
            a.min_band_width_now = min(float(getattr(a, "min_band_width_now", 0.08)), 0.04)
            a.min_band_width_percentile = min(float(getattr(a, "min_band_width_percentile", 0.60)), 0.35)
            a.enforce_three_x_gate = False
        if step >= 4:
            a.pows = False
            a.min_course_pattern_score = 0.0
            a.min_target_band_pct = min(float(getattr(a, "min_target_band_pct", 0.01)), 0.003)
            a.min_band_vs_prev_month = min(float(getattr(a, "min_band_vs_prev_month", 1.4)), 1.0)
            a.min_band_width_now = 0.0
            a.min_band_width_percentile = 0.0
            a.min_band_slope3 = -1.0
            a.min_band_expansion = -1.0
            a.band_touch_lookback = max(int(getattr(a, "band_touch_lookback", 14)), 30)
        if step >= 5:
            a.recent_daily_bars = max(int(getattr(a, "recent_daily_bars", 20)), 40)
            a.recent_233_bars = max(int(getattr(a, "recent_233_bars", 20)), 40)
            a.force_candidate_mode = True
        return a
    reference_symbol = PATTERN_REF_SYMBOL_DEFAULT
    reference_pattern = None
    reference_error = ""
    try:
        reference_pattern = _build_reference_pattern(reference_symbol)
        if not reference_pattern:
            reference_error = f"reference pattern unavailable for {reference_symbol}"
    except Exception as exc:
        reference_error = f"reference pattern unavailable for {reference_symbol}: {exc}"

    @stream_with_context
    def _generate() -> Any:
        scan_start_ts = time.time()
        found = 0
        total_symbols = len(symbols)
        candidate_rows: dict[str, dict[str, Any]] = {}
        prefilter_rows: dict[str, dict[str, Any]] = {}
        scan_pool: list[str] = list(symbols)
        yield json.dumps({"type": "status", "message": f"Universe size: {total_symbols} symbols"}) + "\n"
        strict_args = _tuned_args(0)
        src = str(getattr(strict_args, "data_source", "auto")).lower()
        key_present = bool(str(getattr(strict_args, "polygon_api_key", "")).strip() or os.getenv("POLYGON_API_KEY", "").strip())
        yield json.dumps(
            {"type": "status", "message": f"Data source: {src} | Polygon key: {'present' if key_present else 'missing'}"}
        ) + "\n"
        if fallback_msg:
            yield json.dumps({"type": "status", "message": fallback_msg}) + "\n"
        m = date.today().month
        if m in (10, 11, 12, 1):
            season_mode = "Fall/Winter profile: Daily+233 with weekly context"
        elif m in (5, 6, 7, 8, 9):
            season_mode = "Summer profile: Intraday emphasis (55m and lower)"
        else:
            season_mode = "Transition profile: blended Daily/233 and intraday"
        yield json.dumps({"type": "status", "message": season_mode}) + "\n"
        if reference_pattern:
            yield json.dumps(
                {
                    "type": "status",
                    "message": f"Reference pattern: {reference_symbol} ({str(getattr(reference_pattern, 'setup_direction', 'n/a')).upper()}) with direct BULL/BEAR bias",
                }
            ) + "\n"
        elif reference_error:
            yield json.dumps({"type": "status", "message": f"Reference pattern skipped: {reference_error}"}) + "\n"
        yield json.dumps({"type": "metrics", "scanned": 0, "matches": 0, "total": total_symbols, "tier": "live"}) + "\n"
        yield json.dumps(
            {"type": "status", "message": f"Scanning full universe; returning top {output_limit} closest setups...", "tier": "live"}
        ) + "\n"

        scanned_symbols: set[str] = set()
        hard_failures: list[tuple[str, str]] = []
        def _run_pass(
            pass_label: str,
            pass_args: SimpleNamespace,
            pass_symbols: list[str],
            out_rows: dict[str, dict[str, Any]],
            count_toward_found: bool = True,
        ) -> None:
            nonlocal found
            if not pass_symbols:
                return
            yield_msg = {
                "type": "status",
                "message": f"{pass_label}: analyzing {len(pass_symbols)} symbols (candidates={found})",
                "tier": "live",
            }
            yield json.dumps(yield_msg) + "\n"

            pending_symbols = list(pass_symbols)
            retry_attempt = 0
            while pending_symbols:
                if (time.time() - scan_start_ts) > max_scan_seconds:
                    return
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    future_map = {pool.submit(_analyze_for_args, sym, pass_args): sym for sym in pending_symbols}
                    retry_symbols: list[str] = []

                    for fut in concurrent.futures.as_completed(future_map):
                        sym = future_map[fut]
                        scanned_symbols.add(sym)
                        scanned = len(scanned_symbols)
                        yield json.dumps(
                            {"type": "progress", "message": f"live: {scanned}/{len(symbols)} analyzed ({sym})"}
                        ) + "\n"
                        yield json.dumps(
                            {"type": "metrics", "scanned": scanned, "matches": found, "total": total_symbols, "tier": "live"}
                        ) + "\n"

                        try:
                            analyzed = fut.result()
                        except Exception as exc:
                            if retry_attempt < max_retries:
                                retry_symbols.append(sym)
                            else:
                                hard_failures.append((sym, str(exc)))
                            continue

                        if not analyzed:
                            continue

                        if reference_pattern:
                            direct_similarity = _pattern_similarity(reference_pattern, analyzed)
                            setattr(analyzed, "pattern_similarity_direct", direct_similarity)
                            setattr(analyzed, "pattern_similarity", direct_similarity)
                            setattr(analyzed, "pattern_mode", str(getattr(analyzed, "setup_direction", "n/a")).upper())

                        row = _result_row(analyzed)
                        prev = out_rows.get(sym)
                        if prev is None:
                            if count_toward_found:
                                found += 1
                            out_rows[sym] = row
                        elif float(row.get("rank_score", 0.0)) > float(prev.get("rank_score", 0.0)):
                            out_rows[sym] = row

                if retry_symbols:
                    retry_attempt += 1
                    yield json.dumps(
                        {
                            "type": "status",
                            "message": f"{pass_label}: retrying {len(retry_symbols)} symbol(s), attempt {retry_attempt}/{max_retries}",
                        }
                    ) + "\n"
                pending_symbols = retry_symbols

        # Full-universe prefilter: daily-only candidate ranking (fast path).
        prefilter_args = copy.deepcopy(base_args)
        prefilter_args.require_daily_and_233 = False
        prefilter_args.require_hourly = False
        prefilter_args.require_secondary_confirmation = False
        prefilter_args.require_precision_entry = False
        prefilter_args.scan_intraday_3x = False
        prefilter_args.allow_momentum_watchlist = True
        prefilter_args.force_candidate_mode = True
        prefilter_args.require_macd_stoch_cross = False
        prefilter_args.require_band_liftoff = False
        prefilter_args.require_widening_oscillation = False
        prefilter_args.require_band_widen_window = False
        prefilter_args.require_band_ride = False
        prefilter_args.min_course_pattern_score = 0.0
        prefilter_args.min_adx = 0.0
        prefilter_args.max_rsi = 99.0
        prefilter_args.max_stoch_rsi_k = 99.0
        prefilter_args.enforce_recent_lifecycle_gate = False
        prefilter_args.enforce_three_x_gate = False

        yield from _run_pass("prefilter", prefilter_args, list(symbols), prefilter_rows, count_toward_found=False)

        ordered_prefilter = sorted(prefilter_rows.values(), key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)
        scan_pool = [
            str(r.get("symbol", "")).upper()
            for r in ordered_prefilter[: min(prefilter_limit, len(ordered_prefilter))]
            if str(r.get("symbol", "")).strip()
        ]
        if not scan_pool:
            scan_pool = list(symbols)[: min(prefilter_limit, len(symbols))]

        yield json.dumps(
            {
                "type": "status",
                "message": f"Prefilter kept {len(scan_pool)} symbols for hourly confirmation.",
                "tier": "live",
            }
        ) + "\n"

        hourly_pool = scan_pool[: min(len(scan_pool), confirmation_limit)]
        if len(scan_pool) > len(hourly_pool):
            yield json.dumps(
                {
                    "type": "status",
                    "message": f"Hourly confirmation capped to top {len(hourly_pool)} symbols for speed.",
                    "tier": "live",
                }
            ) + "\n"

        # Pass 1: strict hourly confirmation on top-ranked shortlist only.
        yield from _run_pass("pass 1/2", _tuned_args(0), hourly_pool, candidate_rows)

        # Pass 2 (optional): relaxed pass only for symbols not matched in pass 1.
        if len(candidate_rows) < output_limit and max_tune_steps > 1 and (time.time() - scan_start_ts) <= max_scan_seconds:
            remaining_symbols = [s for s in hourly_pool if s not in candidate_rows]
            relaxed_step = min(2, max_tune_steps - 1)
            yield from _run_pass("pass 2/2", _tuned_args(relaxed_step), remaining_symbols, candidate_rows)

        if (time.time() - scan_start_ts) > max_scan_seconds:
            yield json.dumps(
                {
                    "type": "status",
                    "message": f"Scan time budget reached ({max_scan_seconds}s). Returning best current setups.",
                    "tier": "live",
                }
            ) + "\n"

        if hard_failures and base_args.no_skips:
            first = hard_failures[0]
            yield json.dumps(
                {"type": "error", "message": f"No-skips mode failed. Could not fetch {len(hard_failures)} symbol(s). First: {first[0]} ({first[1]})"}
            ) + "\n"
            yield json.dumps({"type": "done", "count": min(len(candidate_rows), output_limit), "tier": "live"}) + "\n"
            return

        if len(candidate_rows) < output_limit and prefilter_rows:
            for row in ordered_prefilter:
                sym = str(row.get("symbol", "")).upper()
                if not sym or sym in candidate_rows:
                    continue
                candidate_rows[sym] = row
                if len(candidate_rows) >= output_limit:
                    break

        if not candidate_rows:
            emergency_pool = [s for s in scan_pool if str(s).strip()] or [s for s in symbols if str(s).strip()]
            emergency_count = min(output_limit, len(emergency_pool))
            for sym in emergency_pool[:emergency_count]:
                key = str(sym).upper()
                candidate_rows[key] = {
                    "symbol": key,
                    "dir": "BULL",
                    "setup": "Fallback",
                    "phase": 1,
                    "stage": "watch",
                    "setup_score": 0.0,
                    "lifecycle_score": 0.0,
                    "bb_expansion": 0.0,
                    "bb_width_now": 0.0,
                    "bb_width_pct": 0.0,
                    "bb_slope3": 0.0,
                    "bb_osc": 0.0,
                    "bb_regime": False,
                    "bb_vs_prev_month": 0.0,
                    "bb_double_age": 999,
                    "band_ride_score": 0.0,
                    "bb_widen_window_ok": False,
                    "bb_widen_start_age": 999,
                    "rank_score": 0.0,
                }
            yield json.dumps(
                {
                    "type": "status",
                    "message": "Live data returned no ranked setups; showing fallback symbols so results are not empty.",
                    "tier": "live",
                }
            ) + "\n"

        ordered_rows = sorted(candidate_rows.values(), key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)[:output_limit]
        for row in ordered_rows:
            yield json.dumps({"type": "match", "tier": "live", "row": row}) + "\n"

        yield json.dumps(
            {
                "type": "status",
                "message": f"Full scan complete. Showing top {len(ordered_rows)} closest setups.",
                "tier": "live",
            }
        ) + "\n"

        yield json.dumps({"type": "done", "count": len(ordered_rows), "tier": "live"}) + "\n"

    return Response(_generate(), mimetype="application/x-ndjson")


@app.route("/chart_payload", methods=["GET"])
def chart_payload() -> Response:
    symbol = str(request.args.get("symbol", "")).strip().upper()
    if not symbol:
        return Response(json.dumps({"error": "missing symbol"}), status=400, mimetype="application/json")
    tf = str(request.args.get("tf", "")).strip()
    requested = [tf] if tf else None
    try:
        payload = build_chart_payload(symbol, 260, requested_timeframes=requested)
        return Response(json.dumps(payload), mimetype="application/json")
    except Exception as exc:
        return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")


@app.route("/calendar_events", methods=["GET"])
def calendar_events() -> Response:
    raw_days = str(request.args.get("days", str(EVENT_BLOCK_DAYS))).strip()
    try:
        days = max(1, min(21, int(raw_days)))
    except ValueError:
        days = EVENT_BLOCK_DAYS

    raw_symbols = str(request.args.get("symbols", "")).strip().upper()
    symbols = [s for s in dict.fromkeys([x.strip() for x in raw_symbols.split(",") if x.strip()]) if re.fullmatch(r"[A-Z]{1,5}", s)]
    symbols = symbols[:40]

    events: list[dict[str, Any]] = []
    try:
        events.extend(_fetch_fed_events(days))
    except Exception:
        pass
    try:
        events.extend(_fetch_labor_events(days))
    except Exception:
        pass
    for sym in symbols:
        try:
            events.extend(_fetch_symbol_earnings_event(sym, days))
        except Exception:
            continue

    uniq: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (str(event.get("date", "")), str(event.get("label", "")))
        uniq[key] = event
    ordered = sorted(uniq.values(), key=lambda e: (e.get("date", ""), e.get("label", "")))
    return Response(json.dumps({"days": days, "events": ordered}), mimetype="application/json")


@app.route("/simple", methods=["GET", "POST"])
def simple() -> str:
    message = ""
    rows: list[dict[str, Any]] = []
    symbols = "AAPL,MSFT,NVDA,TSLA,AMZN,META"
    use_sp500 = False
    use_qqq = False
    workers = 4
    top = 20

    if request.method == "POST":
        symbols = str(request.form.get("symbols", symbols))
        use_sp500 = request.form.get("use_sp500") == "on"
        use_qqq = request.form.get("use_qqq") == "on"
        workers = max(1, _as_int(request.form, "workers", workers))
        top = max(1, _as_int(request.form, "top", top))
        args = _build_args(request.form)
        configure_data_source(args.data_source, args.polygon_api_key)

        try:
            if use_sp500 and use_qqq:
                universe = list(dict.fromkeys(fetch_sp500_symbols() + fetch_nasdaq100_symbols()))
            elif use_sp500:
                universe = fetch_sp500_symbols()
            elif use_qqq:
                universe = fetch_nasdaq100_symbols()
            else:
                universe = load_symbols(symbols, "data/default_symbols.txt")
            scanned, failures = scan_symbols(universe, args, workers=workers)
            scanned.sort(key=lambda r: r.score, reverse=True)
            rows = scanned[:top]
            message = f"Universe: {len(universe)} | Matches: {len(scanned)} | Failures: {failures}"
        except Exception as exc:
            message = f"Error: {exc}"

    return render_template_string(
        """
<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YEPSTOCKS Simple</title>
<style>
body{font-family:Arial,sans-serif;padding:16px} input,select{padding:6px;margin:4px} table{border-collapse:collapse;width:100%}
th,td{border:1px solid #ddd;padding:6px;font-size:13px;text-align:right} th:first-child,td:first-child{text-align:left}
</style></head><body>
<h2>YEPSTOCKS Simple (No JS)</h2>
<form method="post">
<div><label>Symbols:</label><input name="symbols" value="{{ symbols }}" size="60"></div>
<div><label><input type="checkbox" name="use_sp500" {% if use_sp500 %}checked{% endif %}> Use S&P500</label></div>
<div><label><input type="checkbox" name="use_qqq" {% if use_qqq %}checked{% endif %}> Use QQQ (Nasdaq-100)</label></div>
<div><label>Workers:</label><input name="workers" value="{{ workers }}" size="4">
<label>Top:</label><input name="top" value="{{ top }}" size="4"></div>
<div><label>Data Source:</label>
<select name="data_source"><option value="polygon">polygon</option><option value="auto">auto</option><option value="stooq">stooq</option></select>
<label>Polygon API Key:</label><input name="polygon_api_key" size="40"></div>
<div><button type="submit">Scan</button></div>
</form>
<p>{{ message }}</p>
{% if rows %}
<table>
<tr><th>Symbol</th><th>Close</th><th>SMA50</th><th>SMA89</th><th>SMA200</th><th>Score</th></tr>
{% for r in rows %}
<tr><td>{{ r.symbol }}</td><td>{{ '%.2f'|format(r.close) }}</td><td>{{ '%.2f'|format(r.sma50) }}</td><td>{{ '%.2f'|format(r.sma89) }}</td><td>{{ '%.2f'|format(r.sma200) }}</td><td>{{ '%.3f'|format(r.score) }}</td></tr>
{% endfor %}
</table>
{% endif %}
</body></html>
""",
        message=message,
        rows=rows,
        symbols=symbols,
        use_sp500=use_sp500,
        use_qqq=use_qqq,
        workers=workers,
        top=top,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5055"))
    host = os.getenv("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False, use_reloader=False)
