# Stock Scanner CLI

A lightweight Python application that scans stocks using daily price/volume data and technical filters.

## Features

- Pulls historical candles from Polygon (recommended) or Stooq fallback.
- Computes `SMA20`, `SMA50`, Bollinger Bands(20,2), `RSI(14)`, `StochRSI(14,14,3,3)`, `MACD(12,26,9)`, and `ADX(14)`.
- Filters by price range, dollar volume, RSI/StochRSI, trend, breakout, MACD, and ADX/+DI.
- Ranks candidates by a composite score and prints a table.

## Requirements

- Python 3.10+
- Internet connection (to fetch market data)
- Polygon API key (recommended)

No third-party Python packages are required.

## Usage

Set your Polygon API key:

```bash
export POLYGON_API_KEY="YOUR_KEY_HERE"
```

Run with default symbol universe:

```bash
python3 stock_scanner.py --data-source polygon
```

Scan custom symbols:

```bash
python3 stock_scanner.py --symbols AAPL,MSFT,NVDA,TSLA --require-uptrend --top 10
```

Run POWS-style preset (BB(21,2) + SMA(50/89/200) + StochRSI + MACD + ADX):

```bash
python3 stock_scanner.py --pows --symbols AAPL,MSFT,NVDA,TSLA --top 10
```

Use a custom symbols file:

```bash
python3 stock_scanner.py --symbols-file data/default_symbols.txt --require-breakout
```

## Web App

Run a browser-based scanner with interactive chart visuals:

```bash
python3 -m pip install -r requirements-web.txt
python3 app.py
```

Then open `http://127.0.0.1:5055` if you run on port 5055.

What the web app includes:

- Scan form (symbols or full S&P 500)
- Results table (only matching stocks)
- Interactive chart with crossings for:
  `SMA50/SMA89`, `MACD/Signal`, and `StochRSI %K/%D`
- Price panel with `BB(21,2)` and `SMA50/89/200`

## Deploy Online (Render)

This repo includes:
- `Procfile` (Gunicorn start command)
- `render.yaml` (Render blueprint)

Steps:
1. Push this repo to GitHub.
2. In Render, create a new Web Service from this repo.
3. Build command: `pip install -r requirements-web.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300`
5. Set env vars:
   - `POLYGON_API_KEY` = your real key
   - `SCANNER_DATA_SOURCE` = `polygon`
   - `POLYGON_MIN_GAP_SEC` = `0.4` (increase if 429s appear)
6. Deploy and open the Render URL.

Scan the full S&P 500:

```bash
python3 stock_scanner.py --sp500 --pows --workers 16 --top 50
```

Plot one symbol with crossover markers:

```bash
python3 stock_scanner.py --plot-symbol AAPL --plot-days 300
```

Save chart to file instead of opening a window:

```bash
python3 stock_scanner.py --plot-symbol AAPL --plot-days 300 --plot-output aapl_chart.png
```

## Key Flags

- `--min-price` / `--max-price`: Price bounds.
- `--sp500`: Uses an online S&P 500 constituents list as the scan universe.
- `--data-source`: `polygon`, `stooq`, or `auto`.
- `--polygon-api-key`: Polygon API key (or use `POLYGON_API_KEY` env var).
- `--plot-symbol`: Renders a chart with BB/SMA, MACD, and StochRSI cross markers.
- `--plot-days`: Number of candles used in plot mode.
- `--plot-output`: Save plot image to file.
- `--verbose`: Show progress and fetch warnings during scans.
- `--workers`: Number of concurrent symbol fetch workers.
- `--min-dollar-volume`: Minimum 20-day average dollar volume.
- `--max-rsi`: Maximum RSI(14).
- `--max-stoch-rsi-k`: Maximum StochRSI %K.
- `--min-adx`: Minimum ADX(14) trend strength.
- `--require-uptrend`: Requires `close > SMA20 > SMA50`.
- `--require-breakout`: Requires close at/above 20-day breakout level.
- `--require-macd-bull`: Requires `MACD > signal`.
- `--require-di-bull`: Requires `+DI > -DI`.
- `--require-macd-stoch-cross`: Requires directional MACD and StochRSI crosses within lookback.
- `--require-simultaneous-cross`: Requires directional MACD/Stoch crosses on near same bar.
- `--require-band-liftoff`: Requires outer BB touch + widening bands + directional rejection/lift-off.
- `--signal-direction`: `bull`, `bear`, or `both` for directional cross/liftoff logic.
- `--require-daily-and-233`: Requires signal confirmation on both Daily and 233-minute charts.
- `--intraday-interval-min`: Intraday feed interval used before 233-minute resampling.
- `--cross-lookback`: Bars allowed since directional cross.
- `--band-touch-lookback`: Bars allowed since BB outer-band touch.
- `--min-band-expansion`: Minimum Bollinger-width expansion fraction.
- `--pows`: Applies bundled conditions intended to mirror your linked indicator stack:
  `BB(21,2)`, `SMA(50/89/200)`, `StochRSI(14,14,3,3)`, `MACD(12,26,9)`, and `ADX(14)`.
- `--top`: Max output rows.

## Notes

- The scanner is for research/education, not financial advice.
- Some symbols may fail due to provider symbol mapping or temporary network issues.
- TradingView scripts can include custom logic and visual state rules; this scanner approximates with standard indicator math on daily candles.
- Plot mode requires `matplotlib` (`python3 -m pip install matplotlib`).
