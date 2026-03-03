#!/usr/bin/env python3
"""
IB Account Sync — pulls live data from IB Gateway and writes kona_state.json.

Connects to IB Gateway via ib_insync, reads account balance and open positions,
and atomically updates the state file so King Kam + Big Island have real data.

Runs as a systemd service (ib-sync.service) — loops every 30s during market hours.
"""

import os
import sys
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
IB_CLIENT_ID = int(os.environ.get("IB_CLIENT_ID", "10"))  # Use 10 to avoid conflict with engine (1)
SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "30"))


def load_existing_state() -> dict:
    """Load current state file to preserve fields we don't update."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    """Atomically write state file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def sync_once() -> bool:
    """Connect to IB, pull account data, write state. Returns True on success."""
    from ib_insync import IB

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=10)
    except Exception as e:
        log.warning(f"IB connect failed: {e}")
        return False

    try:
        # Account summary
        account_values = ib.accountSummary()
        balance_map = {}
        for av in account_values:
            if av.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                          "AvailableFunds", "GrossPositionValue"):
                balance_map[av.tag] = float(av.value)

        net_liq = balance_map.get("NetLiquidation", 0)
        cash = balance_map.get("TotalCashValue", 0)
        buying_power = balance_map.get("BuyingPower", 0)
        position_value = balance_map.get("GrossPositionValue", 0)

        # Open positions
        positions = ib.positions()
        open_positions = {}
        for pos in positions:
            contract = pos.contract
            key = f"{contract.symbol}_{contract.lastTradeDateOrExpiry}_{contract.right}{contract.strike}"
            open_positions[key] = {
                "ticker": contract.symbol,
                "side": "call" if contract.right == "C" else "put" if contract.right == "P" else "stock",
                "contracts": int(abs(pos.position)),
                "avg_entry": float(pos.avgCost / (100 if contract.secType == "OPT" else 1)),
                "current_price": 0,  # Updated below if available
                "entry_date": "",
                "expiry": contract.lastTradeDateOrExpiry or "",
                "strike": float(contract.strike) if contract.strike else 0,
                "sec_type": contract.secType,
                "state": "holding",
            }

        # Try to get current prices for positions
        if positions:
            for pos in positions:
                contract = pos.contract
                ib.qualifyContracts(contract)
                ticker = ib.reqMktData(contract, "", False, False)
                ib.sleep(1)
                key = f"{contract.symbol}_{contract.lastTradeDateOrExpiry}_{contract.right}{contract.strike}"
                if key in open_positions and ticker.last and ticker.last > 0:
                    open_positions[key]["current_price"] = float(ticker.last)
                elif key in open_positions and ticker.close and ticker.close > 0:
                    open_positions[key]["current_price"] = float(ticker.close)
                ib.cancelMktData(contract)

        # Load existing state to preserve closed_trades and other fields
        existing = load_existing_state()

        # Account ID
        accounts = ib.managedAccounts()
        account_id = accounts[0] if accounts else "unknown"

        # Build updated state
        state = {
            "version": 2,
            "account": account_id,
            "last_run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "balance": net_liq,
            "cash": cash,
            "buying_power": buying_power,
            "position_value": position_value,
            "open_positions": open_positions,
            "closed_trades": existing.get("closed_trades", []),
            "streak_losses": existing.get("streak_losses", 0),
            "cooldown_until_date": existing.get("cooldown_until_date", None),
            "ib_connected": True,
            "engine_running": True,
        }

        save_state(state)
        log.info(f"Synced: ${net_liq:,.2f} balance, {len(open_positions)} positions, account {account_id}")
        return True

    except Exception as e:
        log.error(f"Sync error: {e}")
        return False
    finally:
        ib.disconnect()


def main():
    log.info(f"IB Sync starting — {IB_HOST}:{IB_PORT}, writing to {STATE_FILE}, every {SYNC_INTERVAL}s")

    # Initial sync
    if sync_once():
        log.info("Initial sync OK")
    else:
        log.warning("Initial sync failed — will retry")

    # Loop
    while True:
        time.sleep(SYNC_INTERVAL)
        try:
            sync_once()
        except Exception as e:
            log.error(f"Sync loop error: {e}")


if __name__ == "__main__":
    main()
