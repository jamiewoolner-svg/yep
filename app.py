#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import copy
import json
import math
from types import SimpleNamespace
from typing import Any

from flask import Flask, Response, render_template, render_template_string, request, stream_with_context

from stock_scanner import (
    POWS_BB_LENGTH,
    POWS_BB_STDDEV,
    POWS_MACD_FAST,
    POWS_MACD_SIGNAL,
    POWS_MACD_SLOW,
    POWS_RSI_LENGTH,
    analyze_candles,
    analyze_symbol,
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
        band_touch_lookback=_as_int(form, "band_touch_lookback", 6),
        min_band_expansion=_as_float(form, "min_band_expansion", 0.05),
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


def _result_row(analyzed: Any) -> dict[str, Any]:
    return {
        "symbol": analyzed.symbol,
        "dir": str(getattr(analyzed, "setup_direction", "n/a")).upper(),
        "setup": str(getattr(analyzed, "setup_type", "n/a")),
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
        "pre3x": _safe_num(max(analyzed.pre3x_bull_score, analyzed.pre3x_bear_score), 2),
        "hist_n": int(getattr(analyzed, "hist_setups_5y", 0)),
        "hist_wr": _safe_num(getattr(analyzed, "hist_win_rate_5y", 0.0) * 100.0, 1),
        "hist_avg": _safe_num(getattr(analyzed, "hist_avg_return_5y", 0.0) * 100.0, 2),
        "tgt_mid": _safe_num(getattr(analyzed, "target_mid_pct", 0.0) * 100.0, 2),
        "tgt_band": _safe_num(getattr(analyzed, "target_band_pct", 0.0) * 100.0, 2),
        "risk": _safe_num(getattr(analyzed, "risk_pct", 0.0) * 100.0, 2),
        "rr": _safe_num(getattr(analyzed, "rr_mid", 0.0), 2),
        "setup_score": _safe_num(getattr(analyzed, "options_setup_score", 0.0), 1),
        "dollar_volume": format_money(analyzed.dollar_volume20),
        "score": _safe_num(analyzed.score, 3),
    }


def _pows_args() -> SimpleNamespace:
    """Locked scanner profile: no UI tuning, fixed POWS-oriented rules."""
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
        band_touch_lookback=8,
        min_band_expansion=0.03,
        require_daily_and_233=True,
        intraday_interval_min=1,
        auto_fallback=True,
        no_skips=False,
        max_retries=4,
        data_source="auto",
        polygon_api_key="",
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
    strict.cross_lookback = min(strict.cross_lookback, 5)
    strict.band_touch_lookback = min(strict.band_touch_lookback, 8)
    strict.min_band_expansion = max(strict.min_band_expansion, 0.03)
    strict.min_adx = max(strict.min_adx, 10.0)

    confirm = copy.deepcopy(strict)
    confirm.require_simultaneous_cross = False
    confirm.require_daily_and_233 = True
    confirm.require_band_liftoff = False
    confirm.bb_spread_watchlist = True
    confirm.cross_lookback = max(confirm.cross_lookback, 7)
    confirm.band_touch_lookback = max(confirm.band_touch_lookback, 10)
    confirm.min_band_expansion = min(confirm.min_band_expansion, 0.01)
    confirm.min_adx = min(confirm.min_adx, 8.0)

    developing = copy.deepcopy(confirm)
    developing.pows = True
    developing.require_macd_stoch_cross = False
    developing.require_band_liftoff = False
    developing.bb_spread_watchlist = True
    developing.cross_lookback = max(developing.cross_lookback, 10)
    developing.band_touch_lookback = max(developing.band_touch_lookback, 12)
    developing.min_band_expansion = min(developing.min_band_expansion, 0.0)
    developing.min_adx = min(developing.min_adx, 6.0)
    developing.require_daily_and_233 = True

    return [("pows_strict", strict), ("pows_confirm", confirm), ("pows_developing", developing)]


def _analyze_for_args(symbol: str, args: SimpleNamespace) -> Any | None:
    retries = 2
    for attempt in range(retries + 1):
        try:
            daily = analyze_symbol(symbol)
            if not daily:
                return None
            if not passes_filters(daily, args):
                return None
            if args.require_daily_and_233:
                intraday = fetch_intraday(symbol, interval_min=args.intraday_interval_min)
                c233 = resample_to_minutes(intraday, target_minutes=233)
                tf233 = analyze_candles(symbol, c233, "233m")
                if not tf233:
                    return None
                if not passes_secondary_timeframe_filters(tf233, args):
                    return None
            return daily
        except RuntimeError:
            if attempt >= retries:
                raise
            continue

    return None


def build_chart_payload(symbol: str, days: int) -> dict[str, Any]:
    candles = fetch_history(symbol)
    lookback = max(60, days)
    candles = candles[-lookback:]
    if len(candles) < 60:
        raise RuntimeError(f"Not enough data to chart {symbol}")

    dates = [c.date.strftime("%Y-%m-%d") for c in candles]
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
        "dates": dates,
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


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    fallback_error = ""
    if request.method == "POST":
        fallback_error = "Browser script did not start scan. Please reload and try again."
    return render_template("index.html", error=fallback_error, results=[], chart_payload=None, form_values={})


@app.route("/scan_stream", methods=["POST"])
def scan_stream() -> Response:
    workers = 16
    plot_days = 260

    try:
        symbols = list(dict.fromkeys(fetch_sp500_symbols() + fetch_nasdaq100_symbols()))
    except Exception as exc:
        payload = json.dumps({"type": "error", "message": str(exc)})
        return Response(payload + "\n", mimetype="application/x-ndjson")

    base_args = _pows_args()
    configure_data_source(base_args.data_source, base_args.polygon_api_key)
    max_retries = max(0, int(base_args.max_retries))
    tiers = _strict_fallback_tiers(base_args) if base_args.auto_fallback else [("strict", base_args)]

    @stream_with_context
    def _generate() -> Any:
        found = 0
        total_symbols = len(symbols)
        yield json.dumps({"type": "status", "message": f"Universe size: {total_symbols} symbols"}) + "\n"
        yield json.dumps({"type": "metrics", "scanned": 0, "matches": 0, "total": total_symbols, "tier": "init"}) + "\n"
        for tier_name, tier_args in tiers:
            yield json.dumps({"type": "status", "message": f"Scanning tier: {tier_name}", "tier": tier_name}) + "\n"
            tier_matches = 0
            pending_symbols = list(symbols)
            attempt = 0
            scanned = 0
            hard_failures: list[tuple[str, str]] = []

            while pending_symbols:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    future_map = {pool.submit(_analyze_for_args, sym, tier_args): sym for sym in pending_symbols}
                    retry_symbols: list[str] = []

                    for fut in concurrent.futures.as_completed(future_map):
                        sym = future_map[fut]
                        scanned += 1
                        yield json.dumps(
                            {"type": "progress", "message": f"{tier_name}: {scanned}/{len(symbols)} analyzed ({sym})"}
                        ) + "\n"
                        yield json.dumps(
                            {"type": "metrics", "scanned": scanned, "matches": found, "total": total_symbols, "tier": tier_name}
                        ) + "\n"

                        try:
                            analyzed = fut.result()
                        except Exception as exc:
                            if attempt < max_retries:
                                retry_symbols.append(sym)
                            else:
                                hard_failures.append((sym, str(exc)))
                            continue

                        if not analyzed:
                            continue
                        tier_matches += 1
                        found += 1
                        row = _result_row(analyzed)
                        chart = build_chart_payload(analyzed.symbol, plot_days)
                        yield json.dumps({"type": "match", "tier": tier_name, "row": row, "chart": chart}) + "\n"
                        yield json.dumps(
                            {"type": "metrics", "scanned": scanned, "matches": found, "total": total_symbols, "tier": tier_name}
                        ) + "\n"

                if retry_symbols:
                    attempt += 1
                    yield json.dumps(
                        {"type": "status", "message": f"{tier_name}: retrying {len(retry_symbols)} symbol(s), attempt {attempt}/{max_retries}"}
                    ) + "\n"
                pending_symbols = retry_symbols

            if hard_failures and base_args.no_skips:
                first = hard_failures[0]
                yield json.dumps(
                    {"type": "error", "message": f"No-skips mode failed. Could not fetch {len(hard_failures)} symbol(s). First: {first[0]} ({first[1]})"}
                ) + "\n"
                yield json.dumps({"type": "done", "count": found, "tier": tier_name}) + "\n"
                return

            if tier_matches > 0:
                yield json.dumps({"type": "done", "count": found, "tier": tier_name}) + "\n"
                return
            yield json.dumps({"type": "status", "message": f"No matches at {tier_name}, falling back."}) + "\n"

        yield json.dumps({"type": "done", "count": found, "tier": "none"}) + "\n"

    return Response(_generate(), mimetype="application/x-ndjson")


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
    app.run(host="127.0.0.1", port=5055, debug=False, use_reloader=False)
