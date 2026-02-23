import csv
import datetime as dt
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Set


def load_env(path: str = '.env') -> None:
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


def polygon_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch_aggs(api_key: str, ticker: str, from_date: str, to_date: str) -> Dict[str, dict]:
    encoded_ticker = urllib.parse.quote(ticker)
    encoded_key = urllib.parse.quote(api_key)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{encoded_ticker}/range/1/day/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={encoded_key}"
    )
    payload = polygon_json(url)
    results = payload.get('results') or []
    by_date: Dict[str, dict] = {}
    for row in results:
        day = dt.datetime.utcfromtimestamp(row['t'] / 1000).strftime('%Y-%m-%d')
        by_date[day] = row
    return by_date


def compare_ohlcv(csv_path: str, api_key: str) -> dict:
    rows = list(csv.DictReader(open(csv_path, 'r', encoding='utf-8')))
    from_date = rows[0]['date']
    to_date = rows[-1]['date']
    polygon = fetch_aggs(api_key, 'AAPL', from_date, to_date)

    mismatches: List[dict] = []
    missing_dates: List[str] = []
    csv_dates = {r['date'] for r in rows}

    for row in rows:
        day = row['date']
        if day not in polygon:
            missing_dates.append(day)
            continue
        source = polygon[day]
        mapping = [('open', 'o'), ('high', 'h'), ('low', 'l'), ('close', 'c'), ('volume', 'v')]
        for csv_key, source_key in mapping:
            csv_value = float(row[csv_key])
            source_value = float(source[source_key])
            tolerance = 0.0 if csv_key == 'volume' else 1e-9
            if abs(csv_value - source_value) > tolerance:
                mismatches.append(
                    {
                        'date': day,
                        'field': csv_key,
                        'csv': csv_value,
                        'polygon': source_value,
                    }
                )

    extra_dates = sorted([day for day in polygon.keys() if day not in csv_dates and from_date <= day <= to_date])

    return {
        'file': csv_path,
        'rows_csv': len(rows),
        'rows_polygon': len([day for day in polygon.keys() if from_date <= day <= to_date]),
        'missing_dates_in_polygon': missing_dates,
        'extra_dates_in_polygon': extra_dates,
        'mismatch_count': len(mismatches),
        'mismatch_examples': mismatches[:10],
    }


def parse_polygon_titles(blob: str) -> List[str]:
    titles: List[str] = []
    for segment in (blob or '').split(' || '):
        match = re.search(r'\(polygon\):\s*(.*)$', segment)
        if match:
            title = match.group(1).strip()
            if title:
                titles.append(title)
    return titles


def fetch_news_titles(api_key: str, ticker: str, start_date: str, end_date: str) -> Set[str]:
    encoded_key = urllib.parse.quote(api_key)
    encoded_ticker = urllib.parse.quote(ticker)
    next_url = (
        f"https://api.polygon.io/v2/reference/news?ticker={encoded_ticker}"
        f"&published_utc.gte={start_date}&published_utc.lte={end_date}"
        f"&limit=100&sort=published_utc&order=asc"
    )

    titles: Set[str] = set()
    page_guard = 0

    while next_url and page_guard < 100:
        page_guard += 1
        sep = '&' if '?' in next_url else '?'
        payload = polygon_json(f"{next_url}{sep}apiKey={encoded_key}")
        for item in payload.get('results') or []:
            title = (item.get('title') or '').strip()
            if title:
                titles.add(title)
        next_url = payload.get('next_url')

    return titles


def compare_setup_news(csv_path: str, api_key: str) -> dict:
    rows = list(csv.DictReader(open(csv_path, 'r', encoding='utf-8')))
    headers = rows[0].keys() if rows else []
    headline_col = next((h for h in headers if 'headline' in h.lower() or 'titles' in h.lower()), None)

    if not headline_col:
        return {
            'file': csv_path,
            'rows_checked': len(rows),
            'headline_col': None,
            'error': 'No headline/titles column found',
        }

    per_row = []
    totals = {'polygon_titles': 0, 'matched': 0, 'unmatched': 0}

    for row in rows:
        ticker = (row.get('ticker') or row.get('symbol') or '').strip()
        setup_date = (row.get('setup_date') or row.get('date') or '').strip()
        if not ticker or not setup_date:
            continue

        window_days = int((row.get('window_days') or '5').strip())
        base_date = dt.datetime.strptime(setup_date, '%Y-%m-%d').date()
        start_date = (base_date - dt.timedelta(days=window_days)).strftime('%Y-%m-%d')
        end_date = (base_date + dt.timedelta(days=window_days)).strftime('%Y-%m-%d')

        stored_titles = parse_polygon_titles(row.get(headline_col, ''))
        live_titles = fetch_news_titles(api_key, ticker, start_date, end_date) if stored_titles else set()

        matched = sum(1 for title in stored_titles if title in live_titles)
        unmatched = len(stored_titles) - matched

        totals['polygon_titles'] += len(stored_titles)
        totals['matched'] += matched
        totals['unmatched'] += unmatched

        per_row.append(
            {
                'ticker': ticker,
                'setup_date': setup_date,
                'polygon_titles': len(stored_titles),
                'matched': matched,
                'unmatched': unmatched,
                'match_rate_pct': round((matched / len(stored_titles) * 100), 2) if stored_titles else 100.0,
            }
        )

    return {
        'file': csv_path,
        'rows_checked': len(rows),
        'headline_col': headline_col,
        'totals': totals,
        'row_results': per_row,
    }


def main() -> None:
    load_env()
    api_key = os.environ.get('POLYGON_API_KEY') or os.environ.get('POLYGON_API_TOKEN')
    if not api_key:
        raise SystemExit('Missing POLYGON_API_KEY/POLYGON_API_TOKEN in environment')

    report = {
        'ohlcv_check': compare_ohlcv('data/aapl_tv_indicators_2025-11-14_to_2026-02-25.csv', api_key),
        'setup_news_check': compare_setup_news('data/setup_news_table.csv', api_key),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
