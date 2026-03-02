#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██╗  ██╗ █████╗ ██╗██╗     ██╗   ██╗ █████╗     ██╗  ██╗ ██████╗ ███╗   ██╗ █████╗  ║
║   ██║ ██╔╝██╔══██╗██║██║     ██║   ██║██╔══██╗    ██║ ██╔╝██╔═══██╗████╗  ██║██╔══██╗ ║
║   █████╔╝ ███████║██║██║     ██║   ██║███████║    █████╔╝ ██║   ██║██╔██╗ ██║███████║ ║
║   ██╔═██╗ ██╔══██║██║██║     ██║   ██║██╔══██║    ██╔═██╗ ██║   ██║██║╚██╗██║██╔══██║ ║
║   ██║  ██╗██║  ██║██║███████╗╚██████╔╝██║  ██║    ██║  ██╗╚██████╔╝██║ ╚████║██║  ██║ ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ║
║                                                                               ║
║   Kona v56 - Unified Options Trading System                                   ║
║   P55/P56: MA Cross Entry + Earnings Filter (DTE>21) + Strict Put Filters    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
USAGE:
  python3 kailua_kona.py                     # Run trading engine
  python3 kailua_kona.py --refresh-earnings  # Update earnings cache (daily cron)
  python3 kailua_kona.py --exits-only        # Process exits only
CRON SETUP (add to crontab -e):
  0 6 * * * cd ~/kona_aws && python3 kailua_kona.py --refresh-earnings >> earnings.log 2>&1
"""
import os
import sys
import json
import math
import time
import threading
import logging
from datetime import datetime, timedelta, date
from collections import defaultdict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from typing import Optional, Dict, List, Tuple, Any
# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('kona')
# ═══════════════════════════════════════════════════════════════════════════════
# API KEYS (from environment — set POLYGON_API_KEY in .env)
# ═══════════════════════════════════════════════════════════════════════════════
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
# ═══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═══════════════════════════════════════════════════════════════════════════════
KONA_STATE_FILE = os.environ.get('KONA_STATE_FILE', 'kona_state.json')
KONA_CONTROLS_FILE = os.environ.get('KONA_CONTROLS_FILE', 'kona_controls.json')
EARNINGS_CACHE_FILE = os.environ.get('KONA_EARNINGS_CACHE', 'earnings_cache.json')
UNIVERSE_FILE = os.environ.get('KONA_UNIVERSE_FILE', 'top_300_options_liquidity_companies_polygon.csv')
# ═══════════════════════════════════════════════════════════════════════════════
# TRADING PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
# Options
DTE_CAL_DAYS = 30
TARGET_DELTA = 0.70
IV_ASSUMPTION = 0.30
RISK_FREE = 0.05
# Signals
BB_PERIOD = 21
BB_STD_DEV = 2.0
BB_GATE_REL = 0.08
STOCH_LONG_THRESHOLD = 30
STOCH_SHORT_THRESHOLD = 70
# Position Sizing
KELLY_FRACTION = 0.15
MAX_PREMIUM_PER_TRADE = 20000
MAX_CONTRACTS = 500
MAX_ENTRIES_PER_WEEK = 20
# Exits
MAX_HOLD_DAYS = 30
PROFIT_TARGET_PCT = 0.30
# Guard Rails
MIN_STOCK_PRICE = 5.0
MIN_OPTION_PRICE = 0.50
MAX_SPREAD_PCT = 0.10
# ═══════════════════════════════════════════════════════════════════════════════
# P55/P56 ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
# MA Cross Confirmation
P55_MA_CROSS_ENABLED = True
P55_MA_FAST = 2
P55_MA_SLOW = 3
P55_MAX_WAIT_DAYS = 5
# Earnings Filter
P55_EARNINGS_FILTER_ENABLED = True
P55_MIN_DTE_FOR_ENTRY = 21  # Skip if earnings within 21 days
# Put Filters
P55_PUT_SKIP_MONTHS = {5, 7}  # May, July
P55_PUT_RET_5D_MAX = 5.0
P55_PUT_DIST_SMA50_MAX = 5.0
P55_PUT_RSI_MAX = 65.0
P55_PUT_RED_DAY_CONFIRM = True
# ═══════════════════════════════════════════════════════════════════════════════
# EARNINGS CACHE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
class EarningsCache:
    """
    Manages earnings dates with Polygon API + local cache.
    Usage:
        cache = EarningsCache()
        cache.load()  # Load from file on startup
        dte = cache.days_to_earnings("AAPL", "2026-03-02")
        if dte and dte <= 21:
            skip_trade()
    """
    _instance = None

    def __init__(self):
        self.data = {'tickers': {}, 'last_updated': None}
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, filepath=None):
        """Load earnings cache from file."""
        filepath = filepath or EARNINGS_CACHE_FILE
        try:
            with open(filepath) as f:
                self.data = json.load(f)
            self._loaded = True
            log.info(f"[Earnings] Loaded cache: {len(self.data.get('tickers', {}))} tickers, "
                     f"updated {self.data.get('last_updated', 'never')}")
        except FileNotFoundError:
            log.warning(f"[Earnings] Cache not found: {filepath} - run with --refresh-earnings")
        except Exception as e:
            log.warning(f"[Earnings] Error loading cache: {e}")

    def save(self, filepath=None):
        """Save earnings cache to file."""
        filepath = filepath or EARNINGS_CACHE_FILE
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        log.info(f"[Earnings] Saved cache: {len(self.data.get('tickers', {}))} tickers")

    def refresh_from_polygon(self, tickers=None):
        """
        Fetch fresh earnings dates from Polygon for all tickers.
        Run daily via cron: python3 kailua_kona.py --refresh-earnings
        """
        if tickers is None:
            tickers = self._load_universe_tickers()
        log.info(f"[Earnings] Refreshing {len(tickers)} tickers from Polygon...")
        self.data = {
            'last_updated': datetime.now().isoformat(),
            'tickers': {}
        }
        today = datetime.now().date()
        success = 0
        for i, ticker in enumerate(tickers):
            try:
                earnings_dates = self._fetch_ticker_earnings(ticker)
                if earnings_dates:
                    next_earn = None
                    last_earn = None
                    for date_str in sorted(earnings_dates):
                        try:
                            earn_date = datetime.strptime(date_str[:10], '%Y-%m-%d').date()
                            if earn_date >= today and next_earn is None:
                                next_earn = date_str[:10]
                            if earn_date < today:
                                last_earn = date_str[:10]
                        except:
                            pass
                    self.data['tickers'][ticker] = {
                        'next_earnings': next_earn,
                        'last_earnings': last_earn,
                    }
                    success += 1
                # Rate limit (5 req/sec)
                if i > 0 and i % 5 == 0:
                    time.sleep(1.1)
                if (i + 1) % 50 == 0:
                    log.info(f"[Earnings] Progress: {i+1}/{len(tickers)} ({success} with data)")
            except Exception as e:
                log.debug(f"[Earnings] Error on {ticker}: {e}")
        self.save()
        log.info(f"[Earnings] Refresh complete: {success}/{len(tickers)} tickers with earnings data")

    def _fetch_ticker_earnings(self, ticker):
        """Fetch earnings from Polygon vX/reference/tickers/{ticker}/events"""
        url = f"https://api.polygon.io/vX/reference/tickers/{ticker}/events?apiKey={POLYGON_API_KEY}"
        try:
            req = Request(url, headers={'User-Agent': 'Kona/1.0'})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get('status') != 'OK':
                return None
            events = data.get('results', {}).get('events', [])
            earnings = []
            for event in events:
                if 'earnings' in event.get('type', '').lower():
                    date_str = event.get('date')
                    if date_str:
                        earnings.append(date_str)
            return earnings if earnings else None
        except HTTPError as e:
            if e.code == 404:
                return None
            raise
        except:
            return None

    def _load_universe_tickers(self):
        """Load tickers from universe file."""
        import csv
        tickers = []
        try:
            with open(UNIVERSE_FILE) as f:
                for row in csv.DictReader(f):
                    tickers.append(row['ticker'])
        except:
            log.warning(f"[Earnings] Could not load {UNIVERSE_FILE}")
        return tickers

    def days_to_earnings(self, ticker, check_date):
        """
        Return days until next earnings.
        Args:
            ticker: Stock symbol
            check_date: Date string 'YYYY-MM-DD' or datetime
        Returns:
            Integer days, or None if unknown
        """
        if not self._loaded:
            self.load()
        ticker_data = self.data.get('tickers', {}).get(ticker)
        if not ticker_data:
            return None
        next_earn = ticker_data.get('next_earnings')
        if not next_earn:
            return None
        try:
            if isinstance(check_date, str):
                check_dt = datetime.strptime(check_date, '%Y-%m-%d').date()
            elif hasattr(check_date, 'date'):
                check_dt = check_date.date()
            else:
                check_dt = check_date
            earn_dt = datetime.strptime(next_earn, '%Y-%m-%d').date()
            days = (earn_dt - check_dt).days
            return days if days >= 0 else None
        except:
            return None

    def should_skip_for_earnings(self, ticker, check_date, min_dte=None):
        """Check if trade should be skipped due to earnings proximity."""
        if not P55_EARNINGS_FILTER_ENABLED:
            return False, None
        min_dte = min_dte or P55_MIN_DTE_FOR_ENTRY
        dte = self.days_to_earnings(ticker, check_date)
        if dte is not None and dte <= min_dte:
            return True, f"earnings in {dte} days (need >{min_dte})"
        return False, None


# Global instance
def get_earnings_cache():
    return EarningsCache.get_instance()


# ═══════════════════════════════════════════════════════════════════════════════
# MA CROSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def p55_check_ma_cross_up(closes):
    """Check if 2MA crossed above 3MA (bullish)."""
    if len(closes) < 4:
        return False
    ma2_now = (closes[-1] + closes[-2]) / 2
    ma3_now = (closes[-1] + closes[-2] + closes[-3]) / 3
    ma2_prev = (closes[-2] + closes[-3]) / 2
    ma3_prev = (closes[-2] + closes[-3] + closes[-4]) / 3
    return ma2_prev <= ma3_prev and ma2_now > ma3_now


def p55_check_ma_cross_down(closes):
    """Check if 2MA crossed below 3MA (bearish)."""
    if len(closes) < 4:
        return False
    ma2_now = (closes[-1] + closes[-2]) / 2
    ma3_now = (closes[-1] + closes[-2] + closes[-3]) / 3
    ma2_prev = (closes[-2] + closes[-3]) / 2
    ma3_prev = (closes[-2] + closes[-3] + closes[-4]) / 3
    return ma2_prev >= ma3_prev and ma2_now < ma3_now


def p55_check_ma_bullish(closes):
    """Check if 2MA > 3MA."""
    if len(closes) < 3:
        return False
    ma2 = (closes[-1] + closes[-2]) / 2
    ma3 = (closes[-1] + closes[-2] + closes[-3]) / 3
    return ma2 > ma3


def p55_check_ma_bearish(closes):
    """Check if 2MA < 3MA."""
    if len(closes) < 3:
        return False
    ma2 = (closes[-1] + closes[-2]) / 2
    ma3 = (closes[-1] + closes[-2] + closes[-3]) / 3
    return ma2 < ma3


# ═══════════════════════════════════════════════════════════════════════════════
# P55 MASTER ENTRY CHECK
# ═══════════════════════════════════════════════════════════════════════════════
def p55_entry_check(ticker, side, entry_date, closes, ret_5d=0, dist_sma50=0, rsi_val=50):
    """
    Master P55/P56 entry check. Call BEFORE entering any trade.
    Args:
        ticker: Stock symbol
        side: 'long' or 'short'
        entry_date: Date string 'YYYY-MM-DD' or datetime
        closes: List of recent closes (at least 4, most recent last)
        ret_5d: 5-day return %
        dist_sma50: Distance from SMA50 %
        rsi_val: Current RSI
    Returns:
        (should_enter: bool, reject_reason: str)
    """
    # 1. MA Cross Confirmation
    if P55_MA_CROSS_ENABLED:
        if side == 'long':
            if not (p55_check_ma_cross_up(closes) or p55_check_ma_bullish(closes)):
                return False, "MA not bullish - waiting for 2MA > 3MA"
        else:  # short
            if not (p55_check_ma_cross_down(closes) or p55_check_ma_bearish(closes)):
                return False, "MA not bearish - waiting for 2MA < 3MA"
    # 2. Earnings Proximity
    cache = get_earnings_cache()
    skip, reason = cache.should_skip_for_earnings(ticker, entry_date)
    if skip:
        return False, reason
    # 3. Put-specific filters
    if side == 'short':
        # Month filter
        if isinstance(entry_date, str):
            month = int(entry_date.split('-')[1])
        else:
            month = entry_date.month
        if month in P55_PUT_SKIP_MONTHS:
            month_names = ['','Jan','Feb','Mar','Apr','May','Jun',
                           'Jul','Aug','Sep','Oct','Nov','Dec']
            return False, f"puts skipped in {month_names[month]}"
        # Strict filters
        if ret_5d > P55_PUT_RET_5D_MAX:
            return False, f"ret_5d {ret_5d:.1f}% > {P55_PUT_RET_5D_MAX}%"
        if dist_sma50 > P55_PUT_DIST_SMA50_MAX:
            return False, f"dist_sma50 {dist_sma50:.1f}% > {P55_PUT_DIST_SMA50_MAX}%"
        if rsi_val > P55_PUT_RSI_MAX:
            return False, f"RSI {rsi_val:.0f} > {P55_PUT_RSI_MAX}"
    return True, "passed P55 checks"


# ═══════════════════════════════════════════════════════════════════════════════
# PENDING SIGNAL TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
class PendingSignalTracker:
    """Track signals waiting for MA cross confirmation."""
    _instance = None

    def __init__(self):
        self.pending = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add(self, ticker, side, signal_date, signal_data=None):
        """Add signal waiting for MA confirmation."""
        key = (ticker, side)
        if isinstance(signal_date, str):
            signal_date = datetime.strptime(signal_date, '%Y-%m-%d')
        expires = signal_date + timedelta(days=P55_MAX_WAIT_DAYS)
        self.pending[key] = {
            'signal_date': signal_date,
            'expires': expires,
            'data': signal_data or {},
        }
        log.info(f"[P55] Pending: {ticker} {side} - waiting for MA (expires {expires.date()})")

    def check(self, ticker, side, closes, current_date):
        """Check if pending signal is now confirmed."""
        key = (ticker, side)
        if key not in self.pending:
            return None
        p = self.pending[key]
        if isinstance(current_date, str):
            current_date = datetime.strptime(current_date, '%Y-%m-%d')
        if current_date > p['expires']:
            log.info(f"[P55] Expired: {ticker} {side}")
            del self.pending[key]
            return None
        confirmed = False
        if side == 'long':
            confirmed = p55_check_ma_cross_up(closes) or p55_check_ma_bullish(closes)
        else:
            confirmed = p55_check_ma_cross_down(closes) or p55_check_ma_bearish(closes)
        if confirmed:
            log.info(f"[P55] Confirmed: {ticker} {side} - MA cross!")
            data = p['data']
            del self.pending[key]
            return data
        return None

    def get_all_pending(self):
        return dict(self.pending)


def get_pending_tracker():
    return PendingSignalTracker.get_instance()


# ═══════════════════════════════════════════════════════════════════════════════
# POLYGON DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════
def polygon_request(url, timeout=10):
    """Make Polygon API request."""
    if '?' in url:
        url += f"&apiKey={POLYGON_API_KEY}"
    else:
        url += f"?apiKey={POLYGON_API_KEY}"
    req = Request(url, headers={'User-Agent': 'Kona/1.0'})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_bars(ticker, days=60):
    """Fetch daily OHLCV bars from Polygon."""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc"
    try:
        data = polygon_request(url)
        if data.get('status') != 'OK' or 'results' not in data:
            return []
        return [{
            'date': datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d'),
            'o': r['o'], 'h': r['h'], 'l': r['l'], 'c': r['c'],
            'v': r.get('v', 0)
        } for r in data['results']]
    except Exception as e:
        log.warning(f"Error fetching {ticker}: {e}")
        return []


def fetch_news(ticker, limit=5):
    """Fetch recent news from Polygon."""
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit={limit}&sort=published_utc"
    try:
        data = polygon_request(url)
        return data.get('results', [])
    except:
        return []


def get_news_sentiment(ticker):
    """
    Get news sentiment score for ticker.
    Returns: (sentiment_score, num_articles) where -1 to +1
    """
    news = fetch_news(ticker, limit=10)
    if not news:
        return 0, 0
    # Simple keyword sentiment
    positive_words = {'beat', 'surge', 'rally', 'upgrade', 'bullish', 'growth', 'profit', 'record'}
    negative_words = {'miss', 'drop', 'fall', 'downgrade', 'bearish', 'loss', 'decline', 'cut'}
    score = 0
    for article in news:
        title = article.get('title', '').lower()
        desc = article.get('description', '').lower()
        text = title + ' ' + desc
        for word in positive_words:
            if word in text:
                score += 1
        for word in negative_words:
            if word in text:
                score -= 1
    # Normalize to -1 to +1
    if len(news) > 0:
        score = max(-1, min(1, score / len(news) / 2))
    return score, len(news)


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_indicators(bars):
    """Compute all technical indicators from bars."""
    if len(bars) < 50:
        return None
    closes = [b['c'] for b in bars]
    highs = [b['h'] for b in bars]
    lows = [b['l'] for b in bars]
    # SMAs
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50
    # Bollinger Bands
    bb_std = (sum((c - sma20)**2 for c in closes[-20:]) / 20) ** 0.5
    bb_upper = sma20 + BB_STD_DEV * bb_std
    bb_lower = sma20 - BB_STD_DEV * bb_std
    bb_pct = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    bb_width = (bb_upper - bb_lower) / sma20 if sma20 > 0 else 0
    # Previous BB%
    if len(closes) >= 21:
        prev_sma20 = sum(closes[-21:-1]) / 20
        prev_std = (sum((c - prev_sma20)**2 for c in closes[-21:-1]) / 20) ** 0.5
        prev_upper = prev_sma20 + BB_STD_DEV * prev_std
        prev_lower = prev_sma20 - BB_STD_DEV * prev_std
        prev_bb_pct = (closes[-2] - prev_lower) / (prev_upper - prev_lower) if prev_upper != prev_lower else 0.5
    else:
        prev_bb_pct = bb_pct
    # Stochastic K
    low_14 = min(lows[-14:])
    high_14 = max(highs[-14:])
    stoch_k = ((closes[-1] - low_14) / (high_14 - low_14) * 100) if high_14 != low_14 else 50
    # RSI
    gains, losses = [], []
    for i in range(1, min(15, len(closes))):
        chg = closes[-i] - closes[-i-1]
        if chg > 0: gains.append(chg)
        else: losses.append(abs(chg))
    avg_gain = sum(gains)/14 if gains else 0.001
    avg_loss = sum(losses)/14 if losses else 0.001
    rsi = 100 - (100 / (1 + avg_gain/avg_loss))
    # Returns and distances
    ret_5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
    dist_sma50 = (closes[-1] - sma50) / sma50 * 100 if sma50 > 0 else 0
    return {
        'close': closes[-1],
        'closes': closes[-10:],
        'sma20': sma20,
        'sma50': sma50,
        'bb_pct': bb_pct,
        'prev_bb_pct': prev_bb_pct,
        'bb_width': bb_width,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'stoch_k': stoch_k,
        'rsi': rsi,
        'ret_5d': ret_5d,
        'dist_sma50': dist_sma50,
        'date': bars[-1]['date'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def check_call_signal(ind):
    """Check for call (long) signal - BB liftoff from lower."""
    # BB liftoff: was below lower band, now above
    if ind['prev_bb_pct'] < 0.05 and ind['bb_pct'] >= 0.05:
        # Stoch oversold
        if ind['stoch_k'] < STOCH_LONG_THRESHOLD:
            # BB width gate
            if ind['bb_width'] >= BB_GATE_REL:
                return True
    return False


def check_put_signal(ind):
    """Check for put (short) signal - BB liftoff from upper."""
    # BB liftoff: was above upper band, now below
    if ind['prev_bb_pct'] > 0.95 and ind['bb_pct'] <= 0.95:
        # Stoch overbought
        if ind['stoch_k'] > STOCH_SHORT_THRESHOLD:
            # BB width gate
            if ind['bb_width'] >= BB_GATE_REL:
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# WATCHLIST SCANNER (for webapp)
# ═══════════════════════════════════════════════════════════════════════════════
def scan_watchlist(tickers, fetch_data=True):
    """
    Scan tickers for approaching signals.
    Returns list of watchlist items for webapp display.
    """
    watchlist = []
    cache = get_earnings_cache()
    today = datetime.now().strftime('%Y-%m-%d')
    for ticker in tickers:
        try:
            if fetch_data:
                bars = fetch_bars(ticker, days=60)
                if not bars:
                    continue
                ind = compute_indicators(bars)
                if not ind:
                    continue
            else:
                continue
            # Check approaching CALL
            if ind['bb_pct'] < 0.15 and ind['stoch_k'] < 40:
                dte = cache.days_to_earnings(ticker, today)
                skip, reason = cache.should_skip_for_earnings(ticker, today)
                score = 0
                if ind['bb_pct'] < 0.15:
                    score += (0.15 - ind['bb_pct']) / 0.15 * 40
                if ind['stoch_k'] < 40:
                    score += (40 - ind['stoch_k']) / 40 * 40
                if p55_check_ma_bullish(ind['closes']):
                    score += 20
                if score >= 25:
                    watchlist.append({
                        'ticker': ticker,
                        'side': 'call',
                        'score': min(100, score),
                        'bb_pct': ind['bb_pct'],
                        'stoch_k': ind['stoch_k'],
                        'ma_state': 'bullish' if p55_check_ma_bullish(ind['closes']) else 'pending',
                        'dte': dte,
                        'earnings_ok': not skip,
                        'triggered': check_call_signal(ind),
                    })
            # Check approaching PUT
            if ind['bb_pct'] > 0.85 and ind['stoch_k'] > 60:
                dte = cache.days_to_earnings(ticker, today)
                skip_earn, _ = cache.should_skip_for_earnings(ticker, today)
                month = int(today.split('-')[1])
                month_ok = month not in P55_PUT_SKIP_MONTHS
                strict_ok = (ind['ret_5d'] < P55_PUT_RET_5D_MAX and
                            ind['dist_sma50'] < P55_PUT_DIST_SMA50_MAX and
                            ind['rsi'] < P55_PUT_RSI_MAX)
                score = 0
                if ind['bb_pct'] > 0.85:
                    score += (ind['bb_pct'] - 0.85) / 0.15 * 40
                if ind['stoch_k'] > 60:
                    score += (ind['stoch_k'] - 60) / 40 * 40
                if p55_check_ma_bearish(ind['closes']):
                    score += 20
                if score >= 25:
                    watchlist.append({
                        'ticker': ticker,
                        'side': 'put',
                        'score': min(100, score),
                        'bb_pct': ind['bb_pct'],
                        'stoch_k': ind['stoch_k'],
                        'rsi': ind['rsi'],
                        'ret_5d': ind['ret_5d'],
                        'dist_sma50': ind['dist_sma50'],
                        'ma_state': 'bearish' if p55_check_ma_bearish(ind['closes']) else 'pending',
                        'dte': dte,
                        'earnings_ok': not skip_earn,
                        'month_ok': month_ok,
                        'strict_ok': strict_ok,
                        'triggered': check_put_signal(ind),
                    })
            time.sleep(0.12)  # Rate limit
        except Exception as e:
            log.debug(f"Error scanning {ticker}: {e}")
    watchlist.sort(key=lambda x: (-x.get('triggered', False), -x['score']))
    return watchlist


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Kona Trading Engine v56')
    parser.add_argument('--refresh-earnings', action='store_true',
                        help='Refresh earnings cache from Polygon')
    parser.add_argument('--exits-only', action='store_true',
                        help='Process exits only, no new entries')
    parser.add_argument('--scan-watchlist', action='store_true',
                        help='Scan and print watchlist')
    args = parser.parse_args()

    if not POLYGON_API_KEY:
        log.error("POLYGON_API_KEY not set. Export it or add to .env file.")
        sys.exit(1)

    print("""
╔═══════════════════════════════════════════════════════════════════╗
║  KONA v56 - Options Trading Engine                                ║
╠═══════════════════════════════════════════════════════════════════╣
║  P55/P56 Filters Active:                                          ║
║    MA Cross Entry (2/3 MA confirmation)                           ║
║    Earnings Filter (skip if DTE <= 21)                            ║
║    Skip May/July for Puts                                         ║
║    Strict Put Filters (ret_5d, dist_sma50, RSI)                  ║
╚═══════════════════════════════════════════════════════════════════╝
""")

    if args.refresh_earnings:
        log.info("Refreshing earnings cache from Polygon...")
        cache = get_earnings_cache()
        cache.refresh_from_polygon()
        return

    if args.scan_watchlist:
        log.info("Scanning watchlist...")
        cache = get_earnings_cache()
        cache.load()
        # Load universe
        import csv
        tickers = []
        try:
            with open(UNIVERSE_FILE) as f:
                for row in csv.DictReader(f):
                    tickers.append(row['ticker'])
        except:
            log.error(f"Could not load {UNIVERSE_FILE}")
            return
        watchlist = scan_watchlist(tickers[:50])  # First 50 for quick test
        print(f"\n{'Ticker':<8} {'Side':<6} {'Score':>6} {'BB%':>6} {'Stoch':>6} {'MA':<8} {'DTE':>5} {'Status'}")
        print("-" * 70)
        for w in watchlist[:20]:
            status = 'OK' if w.get('earnings_ok', True) and w.get('month_ok', True) and w.get('strict_ok', True) else 'WARN'
            dte = w.get('dte') or '-'
            triggered = 'TRIGGERED' if w.get('triggered') else ''
            print(f"{w['ticker']:<8} {w['side']:<6} {w['score']:>5.0f} {w['bb_pct']*100:>5.0f}% {w['stoch_k']:>5.0f} {w['ma_state']:<8} {dte:>5} {status} {triggered}")
        return

    # Normal trading mode
    log.info("Loading earnings cache...")
    cache = get_earnings_cache()
    cache.load()
    if args.exits_only:
        log.info("Running in exits-only mode")
    log.info("Kona v56 ready. Implement your main trading loop here.")
    log.info("Use p55_entry_check() before each trade entry.")


if __name__ == '__main__':
    main()
