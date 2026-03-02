#!/usr/bin/env python3
"""
KING KAM STRATEGY ANALYZER
Supplements king_kam_market_context.py with:
- Trade performance by regime/side/exit type
- Debug reports for specific trades
- Pattern observations
- Connects "what's the market doing" to "how should Kona behave"
Run after market context script, or on-demand when debugging.
Usage:
    python3 king_kam_strategy_analyzer.py                    # Full analysis
    python3 king_kam_strategy_analyzer.py --debug AAPL       # Debug specific ticker
    python3 king_kam_strategy_analyzer.py --regime BULL_MOM  # Regime deep dive
"""
import os
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(os.environ.get('KONA_DATA_DIR', '.'))
OUTPUT_FILE = DATA_DIR / 'king_kam_strategy_analysis.json'


# =============================================================================
# LOAD DATA
# =============================================================================

def load_trades(filepath=None):
    """Load canonical or live trades."""
    if filepath is None:
        # Try canonical first, then live state
        for f in ['kona_canonical_trades.csv', 'trade_log.csv']:
            if (DATA_DIR / f).exists():
                filepath = DATA_DIR / f
                break
    if not filepath or not Path(filepath).exists():
        return []

    trades = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['pnl'] = float(row.get('pnl', row.get('net_pnl', 0)))
            row['hold_days'] = int(row.get('hold_days', row.get('hold_cal_days', 0)))
            trades.append(row)
    return trades


def load_market_context():
    """Load current market context if available."""
    ctx_file = DATA_DIR / 'king_kam_market_context.json'
    if ctx_file.exists():
        with open(ctx_file) as f:
            return json.load(f)
    return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_group(trades):
    """Calculate stats for a group of trades."""
    if not trades:
        return None
    wins = [t for t in trades if t['pnl'] > 0]
    total_pnl = sum(t['pnl'] for t in trades)
    return {
        'n': len(trades),
        'wins': len(wins),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'pnl': round(total_pnl, 0),
        'avg': round(total_pnl / len(trades), 0),
    }


def performance_by(trades, key):
    """Break down performance by a dimension."""
    groups = defaultdict(list)
    for t in trades:
        groups[t.get(key, 'unknown')].append(t)
    return {k: analyze_group(v) for k, v in groups.items()}


def regime_side_matrix(trades):
    """2D breakdown: regime x side."""
    matrix = defaultdict(lambda: defaultdict(list))
    for t in trades:
        regime = t.get('regime', 'unknown')
        side = t.get('side', 'unknown')
        matrix[regime][side].append(t)

    result = {}
    for regime, sides in matrix.items():
        result[regime] = {}
        for side, side_trades in sides.items():
            stats = analyze_group(side_trades)
            if stats:
                result[regime][side] = stats
    return result


def hold_time_analysis(trades):
    """Performance by hold duration."""
    buckets = [
        ('0-2d', 0, 2),
        ('3-5d', 3, 5),
        ('6-10d', 6, 10),
        ('11-20d', 11, 20),
        ('21d+', 21, 999),
    ]
    result = {}
    for name, lo, hi in buckets:
        bucket = [t for t in trades if lo <= t.get('hold_days', 0) <= hi]
        stats = analyze_group(bucket)
        if stats:
            result[name] = stats
    return result


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def find_problems(trades):
    """Identify issues King Kam should warn about."""
    problems = []

    # 1. Regime/side combos that are losing
    rs = regime_side_matrix(trades)
    for regime, sides in rs.items():
        for side, stats in sides.items():
            if stats['n'] >= 5 and stats['wr'] < 35:
                problems.append({
                    'type': 'bad_regime_side',
                    'severity': 'high' if stats['wr'] < 25 else 'medium',
                    'regime': regime,
                    'side': side,
                    'stats': stats,
                    'msg': f"{side}s in {regime}: {stats['wr']}% WR over {stats['n']} trades (${stats['pnl']:+,.0f})"
                })

    # 2. Stop losses destroying PnL
    by_exit = performance_by(trades, 'exit_type')
    for exit_type, stats in by_exit.items():
        if 'stop' in exit_type.lower() and stats['n'] >= 5 and stats['wr'] < 10:
            problems.append({
                'type': 'stops_killing_pnl',
                'severity': 'high',
                'exit_type': exit_type,
                'stats': stats,
                'msg': f"{exit_type}: {stats['n']} trades, {stats['wr']}% WR, ${stats['pnl']:,.0f} lost"
            })

    # 3. Quick exits failing
    hold = hold_time_analysis(trades)
    if '0-2d' in hold and hold['0-2d']['n'] >= 5 and hold['0-2d']['wr'] < 30:
        problems.append({
            'type': 'quick_exits_failing',
            'severity': 'medium',
            'stats': hold['0-2d'],
            'msg': f"0-2 day holds: {hold['0-2d']['wr']}% WR — mean reversion needs mo time brah"
        })

    # 4. Recent losing streak
    recent = sorted(trades, key=lambda t: t.get('exit_date', ''))[-20:]
    recent_losses = len([t for t in recent if t['pnl'] <= 0])
    if len(recent) >= 10 and recent_losses / len(recent) > 0.6:
        problems.append({
            'type': 'recent_drawdown',
            'severity': 'high',
            'msg': f"Last {len(recent)} trades: {recent_losses} losses ({recent_losses/len(recent)*100:.0f}%)"
        })

    return problems


def find_opportunities(trades, market_ctx):
    """Identify what's working that King Kam should highlight."""
    opportunities = []

    # Best exit types
    by_exit = performance_by(trades, 'exit_type')
    for exit_type, stats in by_exit.items():
        if stats['n'] >= 10 and stats['wr'] > 75:
            opportunities.append({
                'type': 'strong_exit_type',
                'exit_type': exit_type,
                'stats': stats,
                'msg': f"{exit_type}: {stats['wr']}% WR, ${stats['pnl']:+,.0f} — dis da moneymaker"
            })

    # Best regime/side combos
    rs = regime_side_matrix(trades)
    for regime, sides in rs.items():
        for side, stats in sides.items():
            if stats['n'] >= 5 and stats['wr'] > 65:
                opportunities.append({
                    'type': 'strong_regime_side',
                    'regime': regime,
                    'side': side,
                    'stats': stats,
                    'msg': f"{side}s in {regime}: {stats['wr']}% WR — sweet spot"
                })

    # If current regime matches a strong combo
    if market_ctx:
        current_regime = market_ctx.get('current', {}).get('regime')
        if current_regime and current_regime in rs:
            for side, stats in rs[current_regime].items():
                if stats['wr'] > 60:
                    opportunities.append({
                        'type': 'current_regime_opportunity',
                        'regime': current_regime,
                        'side': side,
                        'stats': stats,
                        'msg': f"Current regime {current_regime}: {side}s historically {stats['wr']}% WR"
                    })

    return opportunities


# =============================================================================
# DEBUG SPECIFIC TRADE
# =============================================================================

def debug_trade(trades, ticker=None, trade_idx=None):
    """Deep dive on a specific trade."""
    if ticker:
        matches = [t for t in trades if t.get('ticker', '').upper() == ticker.upper()]
        if not matches:
            return {'error': f'No trades found for {ticker}'}
        trade = matches[-1]  # Most recent
    elif trade_idx is not None:
        trade = trades[trade_idx]
    else:
        trade = trades[-1]  # Last trade

    debug = {
        'trade': trade,
        'outcome': 'WIN' if trade['pnl'] > 0 else 'LOSS',
        'analysis': [],
        'comparisons': {},
    }

    # Compare to same side
    same_side = [t for t in trades if t.get('side') == trade.get('side') and t != trade]
    if same_side:
        stats = analyze_group(same_side)
        debug['comparisons']['same_side'] = stats
        if trade['pnl'] <= 0 and stats['wr'] > 55:
            debug['analysis'].append(f"Your {trade['side']}s usually work ({stats['wr']}% WR) — dis one unlucky")

    # Compare to same regime
    same_regime = [t for t in trades if t.get('regime') == trade.get('regime') and t != trade]
    if same_regime:
        stats = analyze_group(same_regime)
        debug['comparisons']['same_regime'] = stats
        if trade['pnl'] <= 0 and stats['wr'] < 40:
            debug['analysis'].append(f"{trade['regime']} regime historically weak ({stats['wr']}% WR) — maybe skip next time")

    # Check exit type
    exit_type = trade.get('exit_type', '')
    if 'stop' in exit_type.lower() and trade['pnl'] < 0:
        debug['analysis'].append("Stopped out — historically stops hurt dis strategy. Mean reversion needs time.")

    # Check hold time
    hold = trade.get('hold_days', 0)
    if hold <= 2 and trade['pnl'] < 0:
        debug['analysis'].append(f"Only held {hold} days — quick exits usually lose. Patience, brah.")

    if trade['pnl'] > 0 and 'opposite_band' in exit_type.lower():
        debug['analysis'].append("Textbook exit — hit da opposite band. Dis how we like um.")

    return debug


# =============================================================================
# GENERATE KING KAM BRIEFING
# =============================================================================

def generate_briefing(trades, market_ctx):
    """Generate the full briefing for King Kam."""
    briefing = {
        'generated_at': datetime.now().isoformat(),
        'trade_count': len(trades),

        # Overall stats
        'overall': analyze_group(trades),

        # Breakdowns
        'by_side': performance_by(trades, 'side'),
        'by_regime': performance_by(trades, 'regime'),
        'by_exit_type': performance_by(trades, 'exit_type'),
        'regime_side_matrix': regime_side_matrix(trades),
        'by_hold_time': hold_time_analysis(trades),

        # Insights
        'problems': find_problems(trades),
        'opportunities': find_opportunities(trades, market_ctx),

        # Recent activity
        'recent_trades': [
            {
                'ticker': t.get('ticker'),
                'side': t.get('side'),
                'date': t.get('exit_date'),
                'pnl': t.get('pnl'),
                'exit': t.get('exit_type'),
            }
            for t in sorted(trades, key=lambda x: x.get('exit_date', ''))[-10:]
        ],
    }

    # King Kam summary in pidgin
    summary_lines = []
    overall = briefing['overall']
    if overall:
        summary_lines.append(f"Overall: {overall['n']} trades, {overall['wr']}% WR, ${overall['pnl']:+,.0f}")

    # Highlight problems
    for p in briefing['problems'][:3]:
        summary_lines.append(f"\u26a0\ufe0f {p['msg']}")

    # Highlight opportunities
    for o in briefing['opportunities'][:2]:
        summary_lines.append(f"\u2705 {o['msg']}")

    briefing['king_kam_summary'] = '\n'.join(summary_lines)

    return briefing


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='King Kam Strategy Analyzer')
    parser.add_argument('--debug', type=str, help='Debug specific ticker')
    parser.add_argument('--regime', type=str, help='Deep dive on regime')
    parser.add_argument('--trades', type=str, help='Path to trades CSV')
    args = parser.parse_args()

    # Load data
    trades = load_trades(args.trades)
    market_ctx = load_market_context()

    if not trades:
        print("No trades found!")
        print(f"Looking in: {DATA_DIR}")
        return

    print(f"Loaded {len(trades)} trades")

    # Debug mode
    if args.debug:
        result = debug_trade(trades, ticker=args.debug)
        print(json.dumps(result, indent=2, default=str))
        return

    # Regime deep dive
    if args.regime:
        regime_trades = [t for t in trades if t.get('regime') == args.regime]
        print(f"\n{args.regime} REGIME: {len(regime_trades)} trades")
        print(json.dumps(regime_side_matrix(regime_trades), indent=2))
        print("\nBy exit type:")
        print(json.dumps(performance_by(regime_trades, 'exit_type'), indent=2))
        return

    # Full analysis
    briefing = generate_briefing(trades, market_ctx)

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(briefing, f, indent=2, default=str)
    print(f"Saved to {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("KING KAM STRATEGY BRIEFING")
    print("=" * 60)
    print(briefing['king_kam_summary'])

    # Print problems
    if briefing['problems']:
        print("\n\u26a0\ufe0f PROBLEMS:")
        for p in briefing['problems']:
            print(f"  [{p['severity'].upper()}] {p['msg']}")

    # Print opportunities
    if briefing['opportunities']:
        print("\n\u2705 OPPORTUNITIES:")
        for o in briefing['opportunities']:
            print(f"  {o['msg']}")


if __name__ == '__main__':
    main()
