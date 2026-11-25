"""
Monthly live signal + (optional) order generation for the FX Carry + Momentum strategy.

Default behaviour is SAFE: it only computes the signal, saves it to signals/,
and prints the orders it WOULD send. Nothing is sent to any broker unless you
pass both `--broker ib` and `--execute`.

Examples
--------
  # 1) signal only (this is what GitHub Actions runs every month)
  python src/live_signal.py

  # 2) compare with your Interactive Brokers PAPER account, print orders (dry run)
  python src/live_signal.py --broker ib --capital 100000

  # 3) actually send market orders to the PAPER account (port 7497)
  python src/live_signal.py --broker ib --capital 100000 --execute

Requires TWS or IB Gateway running locally with the API enabled
(paper trading port 7497 by default) and `pip install ib_async`.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from strategy import current_targets, load_data

ROOT = Path(__file__).resolve().parents[1]
SIGNAL_DIR = ROOT / "signals"

# IB IDEALPRO pair for each currency, and whether USD is the BASE of the pair.
# Long foreign currency = BUY pair if USD is quote (EURUSD), SELL pair if USD is base (USDJPY).
IB_PAIRS = {
    "EUR": ("EURUSD", False), "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False), "NZD": ("NZDUSD", False),
    "JPY": ("USDJPY", True),  "CHF": ("USDCHF", True),
    "CAD": ("USDCAD", True),  "NOK": ("USDNOK", True),
    "SEK": ("USDSEK", True),
}

PAPER_PORT, LIVE_PORT = 7497, 7496


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------

def compute_and_save(n: int, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot, rates = load_data(verbose=True)
    tgt = current_targets(spot, rates, n=n, mom_lookback=lookback)
    as_of = tgt.attrs["as_of"]

    SIGNAL_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_as_of": as_of,
        "params": {"n_per_leg": n, "mom_lookback_months": lookback},
        "weights": {k: round(float(v), 6) for k, v in tgt["final_weight"].items()},
        "carry_pct": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in tgt["carry_%"].items()},
        "momentum_pct": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in tgt["momentum_%"].items()},
    }
    (SIGNAL_DIR / "latest.json").write_text(json.dumps(payload, indent=2))

    hist_file = SIGNAL_DIR / "history.csv"
    row = pd.DataFrame([{"data_as_of": as_of, **payload["weights"]}])
    if hist_file.exists():
        hist = pd.read_csv(hist_file, dtype={"data_as_of": str})
        hist = pd.concat([hist[hist["data_as_of"] != as_of], row], ignore_index=True)
    else:
        hist = row
    hist.to_csv(hist_file, index=False)

    print(f"\nSignal with data as of {as_of} (positions for the following month):")
    print(tgt.round(3).to_string())
    return tgt, spot


# --------------------------------------------------------------------------
# Broker: Interactive Brokers via ib_async
# --------------------------------------------------------------------------

def target_usd_exposure(weights: pd.Series, capital: float, leverage: float) -> pd.Series:
    """USD value of the foreign-currency exposure we want (+ long, - short)."""
    return weights.fillna(0.0) * capital * leverage


def ib_rebalance(weights: pd.Series, spot: pd.DataFrame, capital: float, leverage: float,
                 host: str, port: int, client_id: int, min_trade_usd: float,
                 execute: bool) -> None:
    from ib_async import IB, Forex, MarketOrder  # imported only when needed

    if execute and port != PAPER_PORT:
        raise SystemExit(
            f"Refusing to execute on port {port}. This script only sends orders to the "
            f"PAPER port {PAPER_PORT}. Change the code consciously if you ever go live."
        )

    ib = IB()
    ib.connect(host, port, clientId=client_id)
    try:
        contracts = {c: Forex(IB_PAIRS[c][0]) for c in weights.index if c in IB_PAIRS}
        ib.qualifyContracts(*contracts.values())

        # prices: live mid if available, else last month-end fixing from FRED
        prices = {}
        for ccy, tk in zip(contracts, ib.reqTickers(*contracts.values())):
            p = tk.midpoint()
            if p is None or (isinstance(p, float) and math.isnan(p)):
                p = tk.close
            if p is None or (isinstance(p, float) and math.isnan(p)) or p <= 0:
                usd_per_ccy = float(spot[ccy].dropna().iloc[-1])
                p = 1 / usd_per_ccy if IB_PAIRS[ccy][1] else usd_per_ccy
            prices[ccy] = float(p)

        # current exposure, in USD value of foreign currency
        current = pd.Series(0.0, index=list(contracts))
        for pos in ib.positions():
            c = pos.contract
            if c.secType != "CASH":
                continue
            pair = c.symbol + c.currency
            for ccy, (p, usd_base) in IB_PAIRS.items():
                if pair == p and ccy in current.index:
                    # EURUSD position is in EUR (base); USDJPY position is in USD (base)
                    current[ccy] = -pos.position if usd_base else pos.position * prices[ccy]

        target = target_usd_exposure(weights.reindex(current.index), capital, leverage)
        delta = target - current

        print("\nCurrency   current_USD    target_USD     delta_USD   order")
        orders = []
        for ccy in current.index:
            d = delta[ccy]
            pair, usd_base = IB_PAIRS[ccy]
            if abs(d) < min_trade_usd:
                desc = "-"
            else:
                if usd_base:   # trade quantity in USD; long foreign = SELL USDxxx
                    action, qty = ("SELL" if d > 0 else "BUY"), abs(d)
                else:          # trade quantity in foreign ccy; long foreign = BUY xxxUSD
                    action, qty = ("BUY" if d > 0 else "SELL"), abs(d) / prices[ccy]
                qty = int(round(qty))
                orders.append((ccy, action, qty))
                desc = f"{action} {qty:,} {pair}"
            print(f"{ccy:<8}{current[ccy]:>13,.0f}{target[ccy]:>14,.0f}{d:>14,.0f}   {desc}")

        if not execute:
            print("\nDRY RUN: no orders sent. Add --execute to send them to the PAPER account.")
            return

        for ccy, action, qty in orders:
            trade = ib.placeOrder(contracts[ccy], MarketOrder(action, qty))
            ib.sleep(2)
            print(f"{ccy}: {action} {qty:,} -> {trade.orderStatus.status}")
    finally:
        ib.disconnect()


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="currencies per leg")
    ap.add_argument("--lookback", type=int, default=3, help="momentum lookback (months)")
    ap.add_argument("--broker", choices=["none", "ib"], default="none")
    ap.add_argument("--capital", type=float, default=100_000, help="USD capital")
    ap.add_argument("--leverage", type=float, default=1.0,
                    help="gross exposure per unit of weight (1.0 -> each 1/3 weight = 33%% of capital)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PAPER_PORT)
    ap.add_argument("--client-id", type=int, default=7)
    ap.add_argument("--min-trade-usd", type=float, default=1_000)
    ap.add_argument("--execute", action="store_true", help="actually send orders (paper only)")
    args = ap.parse_args()

    tgt, spot = compute_and_save(args.n, args.lookback)
    if args.broker == "ib":
        ib_rebalance(tgt["final_weight"], spot, args.capital, args.leverage, args.host,
                     args.port, args.client_id, args.min_trade_usd, args.execute)


if __name__ == "__main__":
    main()
