#!/usr/bin/env python3
"""
IB Account Sync — pulls comprehensive live data from IB Gateway.

Syncs every 30s to kona_state.json:
  - Account balance, P&L, margin, buying power
  - Open positions with Greeks, current prices, unrealized P&L
  - Market benchmarks (SPY, QQQ, VIX, DIA, IWM)
  - Recent executions/fills
  - Account P&L breakdown (daily, unrealized, realized)

King Kam + Big Island read this file for real-time intelligence.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ib-sync")

DATA_DIR = os.environ.get("KONA_STATE_DIR", "/home/ec2-user/kona_aws/data")
STATE_FILE = os.path.join(DATA_DIR, "kona_state.json")
IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "4001"))
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))
SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "30"))

# Market benchmarks to track
BENCHMARKS = ["SPY", "QQQ", "VIX", "DIA", "IWM"]


def load_existing_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def sync_once() -> bool:
    """Connect to IB, pull everything, write state."""
    from ib_insync import IB, Stock, Index

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
    except Exception as e:
        log.warning(f"IB connect failed: {e}")
        # Write disconnected state
        existing = load_existing_state()
        existing["ib_connected"] = False
        existing["last_run_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        save_state(existing)
        return False

    try:
        # ── Account Summary ──────────────────────────────────────────
        account_values = ib.accountSummary()
        acct = {}
        for av in account_values:
            acct[av.tag] = av.value

        net_liq = float(acct.get("NetLiquidation", 0))
        cash = float(acct.get("TotalCashValue", 0))
        buying_power = float(acct.get("BuyingPower", 0))
        position_value = float(acct.get("GrossPositionValue", 0))
        maintenance_margin = float(acct.get("MaintMarginReq", 0))
        available_funds = float(acct.get("AvailableFunds", 0))
        realized_pnl = float(acct.get("RealizedPnL", 0))
        unrealized_pnl = float(acct.get("UnrealizedPnL", 0))

        # ── Account PnL ─────────────────────────────────────────────
        pnl_data = {}
        try:
            pnl_list = ib.reqPnL(acct.get("AccountCode", ib.managedAccounts()[0]))
            ib.sleep(1)
            if pnl_list:
                pnl_data = {
                    "daily_pnl": float(pnl_list.dailyPnL) if pnl_list.dailyPnL else 0,
                    "unrealized_pnl": float(pnl_list.unrealizedPnL) if pnl_list.unrealizedPnL else 0,
                    "realized_pnl": float(pnl_list.realizedPnL) if pnl_list.realizedPnL else 0,
                }
                ib.cancelPnL(pnl_list)
        except Exception as e:
            log.debug(f"PnL request: {e}")

        # ── Open Positions with Greeks + current prices ──────────────
        positions = ib.positions()
        open_positions = {}

        for pos in positions:
            contract = pos.contract
            ib.qualifyContracts(contract)

            key = f"{contract.symbol}_{contract.lastTradeDateOrExpiry}_{contract.right}{contract.strike}"

            entry_cost = float(pos.avgCost)
            if contract.secType == "OPT":
                entry_per_contract = entry_cost / 100.0
            else:
                entry_per_contract = entry_cost

            position_info = {
                "ticker": contract.symbol,
                "side": "call" if contract.right == "C" else "put" if contract.right == "P" else "stock",
                "contracts": int(abs(pos.position)),
                "avg_entry": round(entry_per_contract, 4),
                "current_price": 0,
                "entry_date": "",
                "expiry": contract.lastTradeDateOrExpiry or "",
                "strike": float(contract.strike) if contract.strike else 0,
                "sec_type": contract.secType,
                "state": "holding",
                "unrealized_pnl": 0,
                "unrealized_pnl_pct": 0,
                "greeks": {},
            }

            # Request market data + model (for Greeks)
            ticker = ib.reqMktData(contract, "100,101,104,106,221,225", False, False)
            ib.sleep(2)

            # Current price
            if ticker.last and ticker.last > 0:
                position_info["current_price"] = round(float(ticker.last), 4)
            elif ticker.close and ticker.close > 0:
                position_info["current_price"] = round(float(ticker.close), 4)
            elif ticker.bid and ticker.ask and ticker.bid > 0:
                position_info["current_price"] = round((float(ticker.bid) + float(ticker.ask)) / 2, 4)

            # Greeks (options only)
            if contract.secType == "OPT" and ticker.modelGreeks:
                g = ticker.modelGreeks
                position_info["greeks"] = {
                    "delta": round(float(g.delta), 4) if g.delta else None,
                    "gamma": round(float(g.gamma), 4) if g.gamma else None,
                    "theta": round(float(g.theta), 4) if g.theta else None,
                    "vega": round(float(g.vega), 4) if g.vega else None,
                    "iv": round(float(g.impliedVol) * 100, 2) if g.impliedVol else None,
                }

            # Bid/ask spread
            if ticker.bid and ticker.ask and ticker.bid > 0:
                position_info["bid"] = round(float(ticker.bid), 4)
                position_info["ask"] = round(float(ticker.ask), 4)
                position_info["spread"] = round(float(ticker.ask) - float(ticker.bid), 4)

            # Volume
            if ticker.volume and ticker.volume > 0:
                position_info["volume"] = int(ticker.volume)

            # Calculate P&L
            cp = position_info["current_price"]
            ep = position_info["avg_entry"]
            qty = position_info["contracts"]
            if cp > 0 and ep > 0:
                multiplier = 100 if contract.secType == "OPT" else 1
                position_info["unrealized_pnl"] = round((cp - ep) * qty * multiplier, 2)
                position_info["unrealized_pnl_pct"] = round(((cp - ep) / ep) * 100, 2)

            ib.cancelMktData(contract)
            open_positions[key] = position_info

        # ── Market Benchmarks ────────────────────────────────────────
        market_data = {}
        benchmark_contracts = []

        for sym in BENCHMARKS:
            if sym == "VIX":
                c = Index("VIX", "CBOE")
            else:
                c = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(c)
            benchmark_contracts.append((sym, c))

        for sym, c in benchmark_contracts:
            t = ib.reqMktData(c, "", False, False)
            benchmark_contracts  # keep ref

        ib.sleep(2)

        for sym, c in benchmark_contracts:
            t = ib.ticker(c)
            if t:
                last = float(t.last) if t.last and t.last > 0 else float(t.close) if t.close and t.close > 0 else 0
                prev_close = float(t.close) if t.close and t.close > 0 else 0
                change = last - prev_close if last and prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0

                market_data[sym] = {
                    "last": round(last, 2),
                    "prev_close": round(prev_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "high": round(float(t.high), 2) if t.high and t.high > 0 else None,
                    "low": round(float(t.low), 2) if t.low and t.low > 0 else None,
                    "volume": int(t.volume) if t.volume and t.volume > 0 else None,
                }
            ib.cancelMktData(c)

        # ── Recent Executions ────────────────────────────────────────
        recent_executions = []
        try:
            fills = ib.fills()
            for fill in fills[-20:]:  # Last 20 fills
                exec_info = fill.execution
                contract = fill.contract
                recent_executions.append({
                    "time": exec_info.time.isoformat() if exec_info.time else "",
                    "ticker": contract.symbol,
                    "side": exec_info.side,  # BOT or SLD
                    "qty": int(exec_info.shares),
                    "price": round(float(exec_info.price), 4),
                    "sec_type": contract.secType,
                    "right": contract.right,
                    "strike": float(contract.strike) if contract.strike else 0,
                    "expiry": contract.lastTradeDateOrExpiry or "",
                    "commission": round(float(fill.commissionReport.commission), 4) if fill.commissionReport and fill.commissionReport.commission else 0,
                    "realized_pnl": round(float(fill.commissionReport.realizedPNL), 2) if fill.commissionReport and fill.commissionReport.realizedPNL and fill.commissionReport.realizedPNL < 1e300 else 0,
                })
        except Exception as e:
            log.debug(f"Executions: {e}")

        # ── Open Orders ──────────────────────────────────────────────
        open_orders = []
        try:
            orders = ib.openOrders()
            for order in orders:
                open_orders.append({
                    "order_id": order.orderId,
                    "action": order.action,
                    "qty": int(order.totalQuantity),
                    "order_type": order.orderType,
                    "limit_price": float(order.lmtPrice) if order.lmtPrice else None,
                    "status": order.orderStatus.status if hasattr(order, 'orderStatus') else "unknown",
                })
        except Exception as e:
            log.debug(f"Orders: {e}")

        # ── Build State ──────────────────────────────────────────────
        existing = load_existing_state()
        accounts = ib.managedAccounts()
        account_id = accounts[0] if accounts else "unknown"

        state = {
            "version": 3,
            "account": account_id,
            "last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ib_connected": True,
            "engine_running": True,

            # Account financials
            "balance": net_liq,
            "cash": cash,
            "buying_power": buying_power,
            "position_value": position_value,
            "maintenance_margin": maintenance_margin,
            "available_funds": available_funds,

            # P&L
            "daily_pnl": pnl_data.get("daily_pnl", 0),
            "unrealized_pnl": pnl_data.get("unrealized_pnl", unrealized_pnl),
            "realized_pnl": pnl_data.get("realized_pnl", realized_pnl),

            # Positions
            "open_positions": open_positions,

            # Market
            "market": market_data,

            # Executions
            "recent_executions": recent_executions,

            # Orders
            "open_orders": open_orders,

            # Preserved from engine
            "closed_trades": existing.get("closed_trades", []),
            "streak_losses": existing.get("streak_losses", 0),
            "cooldown_until_date": existing.get("cooldown_until_date", None),
        }

        save_state(state)

        pos_count = len(open_positions)
        exec_count = len(recent_executions)
        spy = market_data.get("SPY", {})
        spy_str = f"SPY ${spy.get('last',0):,.0f} ({spy.get('change_pct',0):+.1f}%)" if spy.get("last") else "SPY n/a"
        log.info(f"Synced: ${net_liq:,.0f} | {pos_count} pos | {exec_count} fills | {spy_str}")
        return True

    except Exception as e:
        log.error(f"Sync error: {e}", exc_info=True)
        return False
    finally:
        ib.disconnect()


def main():
    log.info(f"IB Sync v3 starting — {IB_HOST}:{IB_PORT} → {STATE_FILE}, every {SYNC_INTERVAL}s")
    log.info(f"Benchmarks: {', '.join(BENCHMARKS)}")

    if sync_once():
        log.info("Initial sync OK")
    else:
        log.warning("Initial sync failed — will retry")

    while True:
        time.sleep(SYNC_INTERVAL)
        try:
            sync_once()
        except Exception as e:
            log.error(f"Sync loop error: {e}")


if __name__ == "__main__":
    main()
