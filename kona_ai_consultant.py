"""
kona_ai_consultant.py — AI integration for Big Island webapp
═══════════════════════════════════════════════════════════════

Embeds Claude as a 24/7 trading consultant inside the Kona system.
Three layers:

  1. CHAT — Interactive Q&A about trades, positions, strategy
  2. MONITOR — Background thread, checks every 15 min, alerts on anomalies
  3. SAFETY — Suggests risk actions, requires human approval

All read-only by default. Phase 3 (safety controls) writes to
kona_controls.json ONLY after explicit Jamie approval via webapp.

Requirements:
  pip install anthropic --break-system-packages

Usage in webapp:
  from kona_ai_consultant import KonaAI
  ai = KonaAI(data_dir='~/kona_aws')
  ai.start_monitor()  # background thread

  @app.route('/api/ai/chat', methods=['POST'])
  def ai_chat():
      msg = request.json['message']
      response = ai.chat(msg)
      return jsonify({'response': response})
"""

import os
import json
import csv
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

import anthropic

log = logging.getLogger('kona.ai')


# ════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — This is what makes the AI understand Kona
# ════════════════════════════════════════════════════════════════

KONA_SYSTEM_PROMPT = """You are the AI consultant embedded in the Kona options trading system.
You have deep knowledge of the strategy and real-time access to all trading data.

STRATEGY OVERVIEW:
- Mean-reversion using Bollinger Bands, Stochastic RSI, and MACD
- Buy options when stocks hit band extremes, sell when they revert
- 30 DTE target (prefer 20-25 DTE), 0.50-0.75 delta (sliding scale: lower delta at longer DTE)
- $20K max position sizing, volume-aware adaptive sizing
- 449-ticker liquid universe, ~1-6 signals per ticker per year

KEY PARAMETERS:
- Entry: Patient escalating limit orders (4 phases over 30 min, then hold at ask)
- Exit: Immediate on trigger day (opposite band, trail stop, calendar)
- Walk-away: Cancel entry if ask > mid × 1.15
- Signal invalidation: Cancel if stock crosses back inside bands
- 3-day minimum hold before stops can trigger
- Risk tiers: Caution at -1.0% SPY (0.70x size), Risk-off at -1.5% SPY (pause entries)

WHAT YOU SHOULD WATCH FOR:
- Correlated positions (multiple names in same sector or direction)
- Execution anomalies (unfilled orders, wide spreads, walk-away blocks)
- Regime changes (consecutive stops suggesting trend not mean-reversion)
- Theta drag on long-held positions approaching expiry
- Position concentration (too much capital in one name)
- System health (WebSocket disconnects, IB Gateway issues)
- Unusual IV levels at entry (high IV = expensive options, lower expected returns)

HOW TO COMMUNICATE:
- Be direct and specific. Jamie is an experienced trader.
- Lead with the actionable insight, then supporting data.
- Flag risk levels: 🟢 normal, 🟡 watch, 🔴 act now
- When suggesting actions, explain the reasoning AND the alternative.
- Don't hedge everything — have an opinion backed by data.
- Use dollar amounts and percentages, not vague terms.

WHAT YOU CANNOT DO:
- You cannot execute trades. You can only suggest.
- You cannot modify kona_controls.json without Jamie's approval.
- You cannot access external market data directly (use what's in the state files).
- You are not a financial advisor. You are a trading system monitor.

CONTEXT STRUCTURE:
Each message includes a <kona_state> block with current system data.
Parse it carefully. The data is real — these are live positions with real money.
"""


# ════════════════════════════════════════════════════════════════
# DATA AGGREGATOR — Collects all Kona state into structured context
# ════════════════════════════════════════════════════════════════

class KonaDataAggregator:
    """
    Reads all Kona state files and produces a structured snapshot
    that fits within Claude's context window efficiently.
    
    Data sources:
      kona_state.json     — positions, equity, fills
      kona_controls.json  — risk level, pause state, overrides
      logs/               — streaming, execution, risk logs
      trade_history.csv   — completed trades with outcomes
      signals_today.json  — today's signal activity
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).expanduser()
    
    def _read_json(self, filename: str) -> dict:
        """Safely read a JSON file, return empty dict on failure."""
        path = self.data_dir / filename
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to read {filename}: {e}")
        return {}
    
    def _read_csv_tail(self, filename: str, n: int = 50) -> List[dict]:
        """Read last N rows of a CSV file."""
        path = self.data_dir / filename
        try:
            if path.exists():
                rows = []
                with open(path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                return rows[-n:]
        except IOError as e:
            log.warning(f"Failed to read {filename}: {e}")
        return []
    
    def _read_log_tail(self, filename: str, n: int = 100) -> str:
        """Read last N lines of a log file."""
        path = self.data_dir / 'logs' / filename
        try:
            if path.exists():
                with open(path) as f:
                    lines = f.readlines()
                return ''.join(lines[-n:])
        except IOError:
            return ''
    
    def get_full_snapshot(self) -> str:
        """
        Build the complete state snapshot for Claude.
        
        Target: 2000-4000 tokens. Enough context to be useful,
        small enough to leave room for multi-turn conversation.
        """
        state = self._read_json('kona_state.json')
        controls = self._read_json('kona_controls.json')
        
        sections = []
        
        # ── System status ──
        sections.append(self._format_system_status(state, controls))
        
        # ── Open positions ──
        sections.append(self._format_positions(state))
        
        # ── Today's activity ──
        sections.append(self._format_today_activity(state))
        
        # ── Recent trade history ──
        sections.append(self._format_recent_trades())
        
        # ── Risk state ──
        sections.append(self._format_risk_state(controls))
        
        # ── Execution state ──
        sections.append(self._format_execution_state(state))
        
        # ── Recent errors/warnings ──
        sections.append(self._format_recent_errors())
        
        return '\n'.join(s for s in sections if s)
    
    def get_positions_only(self) -> str:
        """Lightweight snapshot for monitoring checks."""
        state = self._read_json('kona_state.json')
        controls = self._read_json('kona_controls.json')
        return '\n'.join([
            self._format_system_status(state, controls),
            self._format_positions(state),
            self._format_risk_state(controls),
        ])
    
    def _format_system_status(self, state: dict, controls: dict) -> str:
        lines = [
            "═══ SYSTEM STATUS ═══",
            f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}",
            f"  Engine: {'RUNNING' if state.get('engine_running') else 'STOPPED'}",
            f"  WebSocket: {'CONNECTED' if state.get('ws_connected') else 'DISCONNECTED'}",
            f"  IB Gateway: {'CONNECTED' if state.get('ib_connected') else 'DISCONNECTED'}",
            f"  Capital: ${state.get('available_capital', 0):,.0f} / "
            f"${state.get('total_capital', 100000):,.0f}",
            f"  Open positions: {len(state.get('positions', {}))}",
            f"  Risk mode: {controls.get('risk_mode', 'normal')}",
        ]
        return '\n'.join(lines)
    
    def _format_positions(self, state: dict) -> str:
        positions = state.get('positions', {})
        if not positions:
            return "═══ POSITIONS ═══\n  No open positions."
        
        lines = ["═══ POSITIONS ═══"]
        total_pnl = 0
        for sym, pos in positions.items():
            entry = pos.get('avg_entry', 0)
            current = pos.get('current_price', entry)
            contracts = pos.get('contracts', 0)
            pnl = (current - entry) * contracts * 100
            total_pnl += pnl
            hold_days = (datetime.now() - datetime.fromisoformat(
                pos.get('entry_date', datetime.now().isoformat())
            )).days if pos.get('entry_date') else 0
            
            dte = 0
            if pos.get('expiry'):
                try:
                    dte = (datetime.strptime(pos['expiry'], '%Y-%m-%d') - datetime.now()).days
                except ValueError:
                    pass
            
            lines.append(
                f"  {pos.get('ticker', '?'):6s} {pos.get('side', '?'):5s} "
                f"{contracts:>2d}x @ ${entry:.2f} → ${current:.2f} "
                f"P&L=${pnl:>+,.0f} ({(current-entry)/entry*100 if entry > 0 else 0:>+.1f}%) "
                f"hold={hold_days}d DTE={dte}d "
                f"state={pos.get('state', '?')}"
            )
        
        lines.append(f"  ──────────")
        lines.append(f"  Total unrealized P&L: ${total_pnl:>+,.0f}")
        return '\n'.join(lines)
    
    def _format_today_activity(self, state: dict) -> str:
        today = state.get('today', {})
        signals = today.get('signals_fired', 0)
        entries = today.get('entries_placed', 0)
        exits = today.get('exits_triggered', 0)
        skipped = today.get('signals_skipped', 0)
        walk_aways = today.get('walk_away_blocks', 0)
        
        lines = [
            "═══ TODAY ═══",
            f"  Signals fired: {signals} (entered: {entries}, skipped: {skipped})",
            f"  Exits triggered: {exits}",
            f"  Walk-away blocks: {walk_aways}",
        ]
        
        # List today's signals
        for sig in today.get('signal_log', [])[-10:]:
            lines.append(f"  {sig.get('time', '')} {sig.get('ticker', '')} "
                        f"{sig.get('side', '')} → {sig.get('action', '')}: "
                        f"{sig.get('reason', '')}")
        
        return '\n'.join(lines)
    
    def _format_recent_trades(self) -> str:
        trades = self._read_csv_tail('trade_history.csv', 20)
        if not trades:
            return "═══ RECENT TRADES ═══\n  No trade history yet."
        
        lines = ["═══ RECENT TRADES (last 20) ═══"]
        for t in trades:
            pnl = float(t.get('net_pnl', 0))
            pnl_pct = float(t.get('pnl_pct', 0))
            lines.append(
                f"  {t.get('signal_date', ''):10s} {t.get('ticker', ''):6s} "
                f"{t.get('side', ''):5s} → {t.get('exit_type', ''):20s} "
                f"hold={t.get('hold_cal_days', '?'):>3s}d "
                f"P&L=${pnl:>+,.0f} ({pnl_pct:>+.1f}%)"
            )
        return '\n'.join(lines)
    
    def _format_risk_state(self, controls: dict) -> str:
        lines = [
            "═══ RISK STATE ═══",
            f"  Mode: {controls.get('risk_mode', 'normal')}",
            f"  SPY change: {controls.get('spy_change_pct', 0):+.2f}%",
            f"  QQQ change: {controls.get('qqq_change_pct', 0):+.2f}%",
            f"  Sizing multiplier: {controls.get('size_multiplier', 1.0):.2f}x",
            f"  Entries paused: {controls.get('entries_paused', False)}",
            f"  Manual override: {controls.get('manual_override', None)}",
        ]
        return '\n'.join(lines)
    
    def _format_execution_state(self, state: dict) -> str:
        pending = state.get('pending_entries', {})
        active_exits = state.get('active_exits', {})
        
        lines = ["═══ EXECUTION STATE ═══"]
        
        if pending:
            lines.append(f"  Pending entries: {len(pending)}")
            for sym, entry in pending.items():
                lines.append(
                    f"    {entry.get('ticker', '?')} phase={entry.get('phase', '?')} "
                    f"filled={entry.get('filled', 0)}/{entry.get('target', 0)} "
                    f"limit=${entry.get('limit', 0):.2f} "
                    f"walk_away={'YES' if entry.get('walk_away_blocked') else 'no'}"
                )
        else:
            lines.append("  No pending entries.")
        
        if active_exits:
            lines.append(f"  Active exits: {len(active_exits)}")
            for sym, exit_o in active_exits.items():
                lines.append(
                    f"    {exit_o.get('ticker', '?')} {exit_o.get('exit_type', '?')} "
                    f"sold={exit_o.get('contracts_sold', 0)}/{exit_o.get('contracts', 0)} "
                    f"state={exit_o.get('sell_state', '?')}"
                )
        else:
            lines.append("  No active exits.")
        
        return '\n'.join(lines)
    
    def _format_recent_errors(self) -> str:
        """Pull recent ERROR/WARNING lines from logs."""
        log_text = self._read_log_tail('kona_streaming.log', 200)
        if not log_text:
            return ''
        
        error_lines = [
            line.strip() for line in log_text.split('\n')
            if any(kw in line for kw in ['ERROR', 'WARNING', 'CRITICAL', 
                                          'DISCONNECT', 'TIMEOUT', 'FAILED'])
        ][-10:]  # last 10 errors
        
        if not error_lines:
            return "═══ SYSTEM HEALTH ═══\n  No recent errors."
        
        lines = ["═══ RECENT ERRORS ═══"]
        for err in error_lines:
            lines.append(f"  {err[:120]}")
        return '\n'.join(lines)


# ════════════════════════════════════════════════════════════════
# CHAT INTERFACE — Interactive Q&A with Claude
# ════════════════════════════════════════════════════════════════

class KonaChat:
    """
    Manages conversation with Claude about Kona's state.
    
    Each chat session maintains history so Claude can reference
    earlier questions in the conversation. History is trimmed
    to keep context under control.
    """
    
    MAX_HISTORY = 20  # messages before trimming
    MODEL = 'claude-sonnet-4-6'  # fast + cheap for monitoring
    
    def __init__(self, aggregator: KonaDataAggregator):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY env
        self.aggregator = aggregator
        self.history: List[dict] = []
    
    def chat(self, user_message: str, include_full_state: bool = True) -> str:
        """
        Send a message and get Claude's analysis.
        
        Automatically prepends current Kona state to the first
        message or when state-dependent questions are asked.
        """
        # Build the user message with state context
        if include_full_state or not self.history:
            state_snapshot = self.aggregator.get_full_snapshot()
            augmented_message = (
                f"<kona_state>\n{state_snapshot}\n</kona_state>\n\n"
                f"{user_message}"
            )
        else:
            augmented_message = user_message
        
        self.history.append({
            'role': 'user',
            'content': augmented_message,
        })
        
        # Trim history if too long
        if len(self.history) > self.MAX_HISTORY:
            # Keep first message (has initial state) and last N
            self.history = self.history[:1] + self.history[-(self.MAX_HISTORY - 1):]
        
        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=2000,
                system=KONA_SYSTEM_PROMPT,
                messages=self.history,
            )
            
            assistant_message = response.content[0].text
            
            self.history.append({
                'role': 'assistant',
                'content': assistant_message,
            })
            
            return assistant_message
            
        except anthropic.APIError as e:
            log.error(f"Claude API error: {e}")
            return f"⚠️ AI unavailable: {e}"
    
    def reset(self):
        """Clear conversation history."""
        self.history = []


# ════════════════════════════════════════════════════════════════
# BACKGROUND MONITOR — Passive 24/7 anomaly detection
# ════════════════════════════════════════════════════════════════

class KonaMonitor:
    """
    Background thread that periodically checks for anomalies.
    
    Two-layer detection:
      Layer 1 (rule-based, every check):
        - Position P&L thresholds
        - Execution timeouts
        - System health
        - Correlated positions
        
      Layer 2 (Claude-interpreted, on trigger):
        - Complex pattern recognition
        - Regime change detection
        - Cross-position risk analysis
    
    Alerts are stored in a list and surfaced via the webapp.
    """
    
    CHECK_INTERVAL = 900  # 15 minutes
    
    # Rule-based thresholds
    POSITION_WARNING_PCT = -30.0     # 🟡 position down 30%
    POSITION_CRITICAL_PCT = -50.0    # 🔴 position down 50%
    CONSECUTIVE_STOPS = 3            # 🔴 regime change signal
    MAX_ENTRY_HOURS = 4              # 🟡 entry taking too long
    MAX_CORRELATED = 3               # 🟡 too many same-direction positions
    CAPITAL_DEPLOYED_PCT = 0.70      # 🟡 70%+ capital deployed
    DTE_WARNING = 5                  # 🔴 position approaching expiry
    
    def __init__(self, aggregator: KonaDataAggregator, 
                 enable_claude: bool = True):
        self.aggregator = aggregator
        self.enable_claude = enable_claude
        self.alerts: List[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        if enable_claude:
            self.client = anthropic.Anthropic()
    
    def start(self):
        """Start the background monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True,
            name='kona-ai-monitor'
        )
        self._thread.start()
        log.info("AI monitor started (checking every %ds)", self.CHECK_INTERVAL)
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("AI monitor stopped")
    
    def _monitor_loop(self):
        """Main loop — runs rule checks, escalates to Claude when needed."""
        while self._running:
            try:
                self._run_checks()
            except Exception as e:
                log.error(f"Monitor check failed: {e}", exc_info=True)
            
            # Sleep in small increments so we can stop quickly
            for _ in range(self.CHECK_INTERVAL):
                if not self._running:
                    break
                time.sleep(1)
    
    def _run_checks(self):
        """Execute all rule-based checks."""
        state = self.aggregator._read_json('kona_state.json')
        controls = self.aggregator._read_json('kona_controls.json')
        
        new_alerts = []
        
        # Check 1: Position P&L
        new_alerts.extend(self._check_position_pnl(state))
        
        # Check 2: Execution health
        new_alerts.extend(self._check_execution(state))
        
        # Check 3: System health
        new_alerts.extend(self._check_system_health(state))
        
        # Check 4: Correlated risk
        new_alerts.extend(self._check_correlation(state))
        
        # Check 5: Capital deployment
        new_alerts.extend(self._check_capital(state))
        
        # Check 6: DTE warnings
        new_alerts.extend(self._check_dte(state))
        
        # Check 7: Consecutive stops (regime change)
        new_alerts.extend(self._check_consecutive_stops())
        
        # If we found 🔴 alerts, escalate to Claude for interpretation
        critical_alerts = [a for a in new_alerts if a['level'] == 'critical']
        if critical_alerts and self.enable_claude:
            interpretation = self._escalate_to_claude(
                new_alerts, state, controls
            )
            if interpretation:
                new_alerts.append({
                    'level': 'ai_analysis',
                    'time': datetime.now().isoformat(),
                    'message': interpretation,
                    'source': 'claude',
                })
        
        # Add new alerts (dedup by message)
        existing_messages = {a['message'] for a in self.alerts[-50:]}
        for alert in new_alerts:
            if alert['message'] not in existing_messages:
                self.alerts.append(alert)
                log.info(f"ALERT [{alert['level']}]: {alert['message']}")
        
        # Trim old alerts
        if len(self.alerts) > 200:
            self.alerts = self.alerts[-100:]
    
    def _check_position_pnl(self, state: dict) -> List[dict]:
        alerts = []
        for sym, pos in state.get('positions', {}).items():
            entry = pos.get('avg_entry', 0)
            current = pos.get('current_price', entry)
            if entry <= 0:
                continue
            pnl_pct = (current - entry) / entry * 100
            ticker = pos.get('ticker', sym)
            
            if pnl_pct <= self.POSITION_CRITICAL_PCT:
                alerts.append({
                    'level': 'critical',
                    'time': datetime.now().isoformat(),
                    'message': f"🔴 {ticker} down {pnl_pct:.1f}% "
                              f"(${entry:.2f} → ${current:.2f})",
                    'source': 'position_pnl',
                    'ticker': ticker,
                })
            elif pnl_pct <= self.POSITION_WARNING_PCT:
                alerts.append({
                    'level': 'warning',
                    'time': datetime.now().isoformat(),
                    'message': f"🟡 {ticker} down {pnl_pct:.1f}% "
                              f"(${entry:.2f} → ${current:.2f})",
                    'source': 'position_pnl',
                    'ticker': ticker,
                })
        return alerts
    
    def _check_execution(self, state: dict) -> List[dict]:
        alerts = []
        for sym, entry in state.get('pending_entries', {}).items():
            created = entry.get('created_at')
            if not created:
                continue
            try:
                created_dt = datetime.fromisoformat(created)
            except ValueError:
                continue
            
            hours_pending = (datetime.now() - created_dt).total_seconds() / 3600
            ticker = entry.get('ticker', sym)
            
            if hours_pending > self.MAX_ENTRY_HOURS:
                alerts.append({
                    'level': 'warning',
                    'time': datetime.now().isoformat(),
                    'message': f"🟡 {ticker} entry pending {hours_pending:.1f}h "
                              f"(filled {entry.get('filled', 0)}/{entry.get('target', 0)}, "
                              f"phase={entry.get('phase', '?')})",
                    'source': 'execution',
                    'ticker': ticker,
                })
            
            if entry.get('walk_away_blocked'):
                alerts.append({
                    'level': 'warning',
                    'time': datetime.now().isoformat(),
                    'message': f"🟡 {ticker} entry walk-away blocked "
                              f"(spread too wide, limit capped)",
                    'source': 'execution',
                    'ticker': ticker,
                })
        return alerts
    
    def _check_system_health(self, state: dict) -> List[dict]:
        alerts = []
        
        if not state.get('ws_connected'):
            alerts.append({
                'level': 'critical',
                'time': datetime.now().isoformat(),
                'message': "🔴 WebSocket DISCONNECTED — no market data flowing",
                'source': 'system_health',
            })
        
        if not state.get('ib_connected'):
            alerts.append({
                'level': 'critical',
                'time': datetime.now().isoformat(),
                'message': "🔴 IB Gateway DISCONNECTED — cannot execute trades",
                'source': 'system_health',
            })
        
        return alerts
    
    def _check_correlation(self, state: dict) -> List[dict]:
        """Check for too many same-direction positions."""
        alerts = []
        positions = state.get('positions', {})
        
        longs = [p.get('ticker', '') for p in positions.values() 
                 if p.get('side') == 'long']
        shorts = [p.get('ticker', '') for p in positions.values() 
                  if p.get('side') == 'short']
        
        if len(longs) >= self.MAX_CORRELATED:
            alerts.append({
                'level': 'warning',
                'time': datetime.now().isoformat(),
                'message': f"🟡 {len(longs)} long positions open "
                          f"({', '.join(longs[:5])}) — correlated risk if market drops",
                'source': 'correlation',
            })
        
        if len(shorts) >= self.MAX_CORRELATED:
            alerts.append({
                'level': 'warning',
                'time': datetime.now().isoformat(),
                'message': f"🟡 {len(shorts)} short positions open "
                          f"({', '.join(shorts[:5])}) — correlated risk if market rips",
                'source': 'correlation',
            })
        
        return alerts
    
    def _check_capital(self, state: dict) -> List[dict]:
        alerts = []
        total = state.get('total_capital', 100000)
        available = state.get('available_capital', total)
        
        if total > 0:
            deployed_pct = (total - available) / total
            if deployed_pct >= self.CAPITAL_DEPLOYED_PCT:
                alerts.append({
                    'level': 'warning',
                    'time': datetime.now().isoformat(),
                    'message': f"🟡 {deployed_pct:.0%} capital deployed "
                              f"(${total - available:,.0f} / ${total:,.0f})",
                    'source': 'capital',
                })
        return alerts
    
    def _check_dte(self, state: dict) -> List[dict]:
        alerts = []
        for sym, pos in state.get('positions', {}).items():
            if not pos.get('expiry'):
                continue
            try:
                expiry = datetime.strptime(pos['expiry'], '%Y-%m-%d')
                dte = (expiry - datetime.now()).days
                ticker = pos.get('ticker', sym)
                
                if dte <= self.DTE_WARNING:
                    alerts.append({
                        'level': 'critical',
                        'time': datetime.now().isoformat(),
                        'message': f"🔴 {ticker} expires in {dte} days — "
                                  f"theta accelerating, consider exit",
                        'source': 'dte',
                        'ticker': ticker,
                    })
            except ValueError:
                pass
        return alerts
    
    def _check_consecutive_stops(self) -> List[dict]:
        """Check if recent trades are all stops — regime change signal."""
        alerts = []
        trades = self.aggregator._read_csv_tail('trade_history.csv', 10)
        
        if len(trades) < self.CONSECUTIVE_STOPS:
            return alerts
        
        recent = trades[-self.CONSECUTIVE_STOPS:]
        stop_types = {'stop_above_start', 'stop_below_start', 
                      'early_loss_exit'}
        
        if all(t.get('exit_type') in stop_types for t in recent):
            tickers = [t.get('ticker', '?') for t in recent]
            alerts.append({
                'level': 'critical',
                'time': datetime.now().isoformat(),
                'message': f"🔴 Last {self.CONSECUTIVE_STOPS} trades ALL stopped out "
                          f"({', '.join(tickers)}) — possible regime change, "
                          f"market may be trending not mean-reverting",
                'source': 'regime_change',
            })
        
        return alerts
    
    def _escalate_to_claude(self, alerts: List[dict], 
                            state: dict, controls: dict) -> Optional[str]:
        """
        When critical alerts fire, ask Claude to interpret the
        overall picture and suggest specific actions.
        """
        snapshot = self.aggregator.get_positions_only()
        
        alert_text = '\n'.join(
            f"  [{a['level']}] {a['message']}" for a in alerts
        )
        
        prompt = (
            f"<kona_state>\n{snapshot}\n</kona_state>\n\n"
            f"MONITORING ALERTS TRIGGERED:\n{alert_text}\n\n"
            f"Analyze these alerts together. What's the overall picture? "
            f"Is this a coordinated problem or isolated issues? "
            f"What specific action should Jamie take RIGHT NOW? "
            f"Keep it to 3-4 sentences max."
        )
        
        try:
            response = self.client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=500,
                system=KONA_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response.content[0].text
        except anthropic.APIError as e:
            log.error(f"Claude escalation failed: {e}")
            return None
    
    def get_recent_alerts(self, n: int = 20) -> List[dict]:
        """Return most recent alerts for webapp display."""
        return self.alerts[-n:]
    
    def get_unread_count(self) -> int:
        """Count alerts from the last check cycle."""
        if not self.alerts:
            return 0
        cutoff = datetime.now() - timedelta(seconds=self.CHECK_INTERVAL)
        return sum(
            1 for a in self.alerts
            if datetime.fromisoformat(a['time']) > cutoff
        )


# ════════════════════════════════════════════════════════════════
# SAFETY CONTROLLER — Suggest + Approve risk actions
# ════════════════════════════════════════════════════════════════

class KonaSafety:
    """
    AI-suggested safety actions that require human approval.
    
    Flow:
      1. Monitor or Claude suggests an action (e.g., pause entries)
      2. Suggestion stored with reason and details
      3. Jamie reviews in webapp, clicks Approve or Reject
      4. On approval, writes to kona_controls.json
      5. On rejection, logs the decision
    
    NEVER autonomous. Every action requires a button click.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir).expanduser()
        self.pending_actions: List[dict] = []
        self.action_history: List[dict] = []
    
    def suggest(self, action_type: str, reason: str,
                params: dict = None, source: str = 'monitor') -> dict:
        """
        Create a pending safety action suggestion.
        
        action_type: 'pause_entries', 'reduce_size', 'risk_off',
                     'force_exit', 'resume_normal'
        """
        suggestion = {
            'id': f"action_{int(time.time())}_{len(self.pending_actions)}",
            'type': action_type,
            'reason': reason,
            'params': params or {},
            'source': source,
            'suggested_at': datetime.now().isoformat(),
            'status': 'pending',  # pending → approved → executed | rejected
        }
        
        self.pending_actions.append(suggestion)
        log.info(f"SAFETY SUGGESTION: {action_type} — {reason}")
        return suggestion
    
    def approve(self, action_id: str) -> bool:
        """
        Jamie approves a suggested action. Execute it.
        Returns True if action was found and executed.
        """
        for action in self.pending_actions:
            if action['id'] == action_id and action['status'] == 'pending':
                action['status'] = 'approved'
                action['approved_at'] = datetime.now().isoformat()
                
                success = self._execute_action(action)
                action['executed'] = success
                
                self.action_history.append(action)
                self.pending_actions.remove(action)
                
                log.info(f"SAFETY APPROVED: {action['type']} "
                        f"({'executed' if success else 'FAILED'})")
                return success
        
        return False
    
    def reject(self, action_id: str) -> bool:
        """Jamie rejects a suggested action."""
        for action in self.pending_actions:
            if action['id'] == action_id and action['status'] == 'pending':
                action['status'] = 'rejected'
                action['rejected_at'] = datetime.now().isoformat()
                
                self.action_history.append(action)
                self.pending_actions.remove(action)
                
                log.info(f"SAFETY REJECTED: {action['type']}")
                return True
        return False
    
    def _execute_action(self, action: dict) -> bool:
        """
        Write the approved action to kona_controls.json.
        
        This is the ONLY place the AI system modifies trading state,
        and ONLY after explicit human approval.
        """
        controls_path = self.data_dir / 'kona_controls.json'
        
        try:
            controls = {}
            if controls_path.exists():
                with open(controls_path) as f:
                    controls = json.load(f)
            
            action_type = action['type']
            
            if action_type == 'pause_entries':
                controls['entries_paused'] = True
                controls['pause_reason'] = action['reason']
                controls['paused_at'] = datetime.now().isoformat()
                controls['paused_by'] = 'ai_approved'
                
            elif action_type == 'reduce_size':
                multiplier = action['params'].get('multiplier', 0.5)
                controls['size_multiplier'] = multiplier
                controls['size_reason'] = action['reason']
                
            elif action_type == 'risk_off':
                controls['risk_mode'] = 'risk_off'
                controls['entries_paused'] = True
                controls['risk_off_reason'] = action['reason']
                controls['risk_off_at'] = datetime.now().isoformat()
                
            elif action_type == 'resume_normal':
                controls['risk_mode'] = 'normal'
                controls['entries_paused'] = False
                controls['size_multiplier'] = 1.0
                controls.pop('pause_reason', None)
                controls.pop('risk_off_reason', None)
                
            elif action_type == 'force_exit':
                # Flag a specific position for immediate exit
                ticker = action['params'].get('ticker')
                if ticker:
                    force_exits = controls.get('force_exits', [])
                    force_exits.append({
                        'ticker': ticker,
                        'reason': action['reason'],
                        'requested_at': datetime.now().isoformat(),
                    })
                    controls['force_exits'] = force_exits
            else:
                log.warning(f"Unknown action type: {action_type}")
                return False
            
            # Write atomically
            tmp_path = controls_path.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(controls, f, indent=2)
            tmp_path.rename(controls_path)
            
            return True
            
        except Exception as e:
            log.error(f"Failed to execute safety action: {e}", exc_info=True)
            return False


# ════════════════════════════════════════════════════════════════
# MAIN INTERFACE — What the webapp imports
# ════════════════════════════════════════════════════════════════

class KonaAI:
    """
    Top-level interface. Import this in the webapp.
    
    Usage:
        ai = KonaAI(data_dir='~/kona_aws')
        ai.start_monitor()
        
        # Chat endpoint
        response = ai.chat("What's the riskiest position right now?")
        
        # Get alerts for dashboard
        alerts = ai.get_alerts()
        
        # Safety actions
        pending = ai.get_pending_actions()
        ai.approve_action(action_id)
    """
    
    def __init__(self, data_dir: str = '~/kona_aws'):
        self.aggregator = KonaDataAggregator(data_dir)
        self.chat_session = KonaChat(self.aggregator)
        self.monitor = KonaMonitor(self.aggregator)
        self.safety = KonaSafety(data_dir)
        
        log.info("KonaAI initialized (data_dir=%s)", data_dir)
    
    # ── Chat ──
    def chat(self, message: str) -> str:
        """Send a message to the AI consultant."""
        return self.chat_session.chat(message)
    
    def reset_chat(self):
        """Clear chat history."""
        self.chat_session.reset()
    
    # ── Monitor ──
    def start_monitor(self):
        """Start background monitoring."""
        self.monitor.start()
    
    def stop_monitor(self):
        """Stop background monitoring."""
        self.monitor.stop()
    
    def get_alerts(self, n: int = 20) -> List[dict]:
        """Get recent alerts."""
        return self.monitor.get_recent_alerts(n)
    
    def get_alert_count(self) -> int:
        """Unread alert count for badge."""
        return self.monitor.get_unread_count()
    
    # ── Safety ──
    def suggest_action(self, action_type: str, reason: str,
                       params: dict = None) -> dict:
        """AI suggests a safety action."""
        return self.safety.suggest(action_type, reason, params)
    
    def get_pending_actions(self) -> List[dict]:
        """Get actions awaiting approval."""
        return self.safety.pending_actions
    
    def approve_action(self, action_id: str) -> bool:
        """Approve a safety action."""
        return self.safety.approve(action_id)
    
    def reject_action(self, action_id: str) -> bool:
        """Reject a safety action."""
        return self.safety.reject(action_id)
    
    # ── Quick analysis (no chat history) ──
    def quick_analysis(self) -> str:
        """
        One-shot analysis of current state.
        Good for scheduled reports (daily open, daily close).
        """
        snapshot = self.aggregator.get_full_snapshot()
        
        prompt = (
            f"<kona_state>\n{snapshot}\n</kona_state>\n\n"
            f"Give a brief status report:\n"
            f"1. Overall health (🟢/🟡/🔴)\n"
            f"2. Any positions that need attention\n"
            f"3. Risk exposure assessment\n"
            f"4. One actionable recommendation\n"
            f"Keep it concise — 4-6 sentences total."
        )
        
        try:
            response = self.chat_session.client.messages.create(
                model=KonaChat.MODEL,
                max_tokens=500,
                system=KONA_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': prompt}],
            )
            return response.content[0].text
        except anthropic.APIError as e:
            return f"⚠️ Analysis unavailable: {e}"


# ════════════════════════════════════════════════════════════════
# FLASK ROUTES — Copy these into kona_webapp.py
# ════════════════════════════════════════════════════════════════

FLASK_ROUTES_TEMPLATE = """
# ── Add to kona_webapp.py imports ──
from kona_ai_consultant import KonaAI

# ── Initialize after app = Flask(...) ──
ai = KonaAI(data_dir='~/kona_aws')
ai.start_monitor()

# ── AI Chat endpoint ──
@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    message = request.json.get('message', '')
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    response = ai.chat(message)
    return jsonify({
        'response': response,
        'alerts': ai.get_alert_count(),
    })

# ── AI Alerts endpoint ──
@app.route('/api/ai/alerts')
def ai_alerts():
    return jsonify({
        'alerts': ai.get_alerts(20),
        'unread': ai.get_alert_count(),
    })

# ── AI Safety actions ──
@app.route('/api/ai/actions')
def ai_actions():
    return jsonify({
        'pending': ai.get_pending_actions(),
    })

@app.route('/api/ai/actions/<action_id>/approve', methods=['POST'])
def ai_approve(action_id):
    success = ai.approve_action(action_id)
    return jsonify({'success': success})

@app.route('/api/ai/actions/<action_id>/reject', methods=['POST'])
def ai_reject(action_id):
    success = ai.reject_action(action_id)
    return jsonify({'success': success})

# ── Daily report (call from cron at market close) ──
@app.route('/api/ai/report')
def ai_report():
    report = ai.quick_analysis()
    return jsonify({'report': report})

# ── Reset chat ──
@app.route('/api/ai/reset', methods=['POST'])
def ai_reset():
    ai.reset_chat()
    return jsonify({'success': True})
"""
