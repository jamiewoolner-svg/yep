import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


def resolve_band_columns(fieldnames: List[str]) -> tuple[str, str]:
    names = set(fieldnames)
    if {'upper_band', 'lower_band'}.issubset(names):
        return 'upper_band', 'lower_band'
    if {'bb_upper_21_2', 'bb_lower_21_2'}.issubset(names):
        return 'bb_upper_21_2', 'bb_lower_21_2'
    raise ValueError(
        "Could not find Bollinger band columns. Expected either "
        "('upper_band','lower_band') or ('bb_upper_21_2','bb_lower_21_2')."
    )


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    with open(csv_path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError('CSV is empty or missing header row.')
        upper_col, lower_col = resolve_band_columns(reader.fieldnames)

        rows: List[Dict[str, object]] = []
        for row in reader:
            day = datetime.strptime(row['date'], '%Y-%m-%d')
            upper = float(row[upper_col])
            lower = float(row[lower_col])
            rows.append(
                {
                    'date': day,
                    'date_str': row['date'],
                    'bb_width': upper - lower,
                }
            )

    rows.sort(key=lambda r: r['date'])
    return rows


def calculate_bb_expansion(
    csv_path: Path, trade_dates: List[str], lookback_days: int, lookback_mode: str
) -> List[Dict[str, object]]:
    rows = load_rows(csv_path)
    date_to_index = {str(row['date_str']): idx for idx, row in enumerate(rows)}

    results: List[Dict[str, object]] = []
    for trade_date in trade_dates:
        if trade_date not in date_to_index:
            raise ValueError(f"Trade date not found in CSV: {trade_date}")

        entry_index = date_to_index[trade_date]
        entry = rows[entry_index]

        if lookback_mode == 'rows':
            start_index = max(0, entry_index - lookback_days)
            lookback_slice = rows[start_index:entry_index]
        else:
            window_start = entry['date'] - timedelta(days=lookback_days)
            lookback_slice = [
                row
                for row in rows[:entry_index]
                if window_start <= row['date'] < entry['date']
            ]

        if not lookback_slice:
            raise ValueError(f"No lookback rows for trade date: {trade_date}")

        narrowing = min(lookback_slice, key=lambda row: float(row['bb_width']))
        narrowing_index = date_to_index[str(narrowing['date_str'])]

        narrowing_width = float(narrowing['bb_width'])
        entry_width = float(entry['bb_width'])
        expansion_pct = ((entry_width / narrowing_width) - 1.0) * 100.0
        days_between_rows = entry_index - narrowing_index
        days_between_calendar = int((entry['date'] - narrowing['date']).days)

        results.append(
            {
                'trade_date': trade_date,
                'narrowing_date': str(narrowing['date_str']),
                'narrowing_width': round(narrowing_width, 6),
                'entry_width': round(entry_width, 6),
                'expansion_pct': round(expansion_pct, 4),
                'days_between_rows': days_between_rows,
                'days_between_calendar': days_between_calendar,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calculate Bollinger Band expansion from narrowing to trade date.'
    )
    parser.add_argument(
        '--csv',
        default='data/aapl_tv_indicators_2025-11-14_to_2026-02-25.csv',
        help='Input CSV path.',
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=40,
        help='Lookback window size; interpreted by --lookback-mode.',
    )
    parser.add_argument(
        '--lookback-mode',
        choices=['rows', 'calendar'],
        default='rows',
        help='Use prior N rows or prior N calendar days before each trade date.',
    )
    parser.add_argument(
        '--trade-dates',
        nargs='+',
        default=['2025-12-19', '2026-01-26', '2026-02-12'],
        help='Trade entry dates in YYYY-MM-DD format.',
    )
    parser.add_argument(
        '--out',
        default='data/bb_expansion_summary.csv',
        help='Output CSV path.',
    )
    args = parser.parse_args()

    summary = calculate_bb_expansion(
        Path(args.csv), args.trade_dates, args.lookback, args.lookback_mode
    )
    columns = [
        'trade_date',
        'narrowing_date',
        'narrowing_width',
        'entry_width',
        'expansion_pct',
        'days_between_rows',
        'days_between_calendar',
    ]

    print(', '.join(columns))
    for row in summary:
        print(', '.join(str(row[col]) for col in columns))

    with open(args.out, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved summary: {args.out}")


if __name__ == '__main__':
    main()
