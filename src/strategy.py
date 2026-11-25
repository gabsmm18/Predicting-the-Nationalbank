"""
FX Carry + Momentum filter — core library.

Universe: G10 currencies vs USD (EUR, GBP, JPY, CHF, AUD, NZD, CAD, NOK, SEK).
Data: FRED (daily FX fixings from the Fed H.10, monthly 3M interbank rates from the OECD).

Conventions
-----------
* Every FX series is expressed as USD per 1 unit of foreign currency (S),
  so a rise in S = the foreign currency appreciates vs the USD.
* Monthly excess return of being LONG foreign currency / SHORT USD in month t+1:
      xr_{t+1} = ln(S_{t+1} / S_t) + (i*_t - i_t) / 12
  i.e. spot change + interest differential earned over the month
  (this is what a rolled 1M FX forward pays, under covered interest parity).
* Signals are computed with information available at end of month t
  and applied to returns of month t+1 (no look-ahead).
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Data definitions
# --------------------------------------------------------------------------

# FRED FX series: (series id, True if quoted as foreign per USD -> must invert)
FX_SERIES = {
    "EUR": ("DEXUSEU", False),  # USD per EUR
    "GBP": ("DEXUSUK", False),  # USD per GBP
    "AUD": ("DEXUSAL", False),  # USD per AUD
    "NZD": ("DEXUSNZ", False),  # USD per NZD
    "JPY": ("DEXJPUS", True),   # JPY per USD -> invert
    "CHF": ("DEXSZUS", True),   # CHF per USD -> invert
    "CAD": ("DEXCAUS", True),   # CAD per USD -> invert
    "NOK": ("DEXNOUS", True),   # NOK per USD -> invert
    "SEK": ("DEXSDUS", True),   # SEK per USD -> invert
}

# OECD 3-month interbank rates (monthly averages, % p.a.) via FRED
RATE_SERIES = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "NOK": "IR3TIB01NOM156N",
    "SEK": "IR3TIB01SEM156N",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_API_URL = ("https://api.stlouisfed.org/fred/series/observations"
                "?series_id={}&api_key={}&file_type=json")

# Free key from https://fredaccount.stlouisfed.org/apikeys
# Set it as an environment variable FRED_API_KEY (or assign it here at runtime).
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def _http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_fred(series_id: str, retries: int = 3) -> pd.Series:
    """
    Download one FRED series as a pandas Series.

    * If FRED_API_KEY is set -> official FRED API (reliable, recommended).
    * Otherwise -> the keyless CSV export, which is often slow or blocked
      from cloud servers such as Google Colab or GitHub Actions.
    Retries with increasing waits if the server does not answer.
    """
    last_err = None
    for attempt in range(retries):
        try:
            if FRED_API_KEY:
                obs = json.loads(_http_get(FRED_API_URL.format(series_id, FRED_API_KEY)))["observations"]
                s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
            else:
                raw = _http_get(FRED_CSV_URL.format(series_id))
                df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True)
                s = df.iloc[:, 0]
            s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna().sort_index()
            s.name = series_id
            return s
        except Exception as e:  # timeout, HTTP error, bad key...
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{last_err}")


def to_monthly_last(s: pd.Series) -> pd.Series:
    """Daily series -> month-end value, indexed by monthly Period."""
    return s.groupby(s.index.to_period("M")).last()


def to_monthly_period(s: pd.Series) -> pd.Series:
    """Monthly FRED series (dated 1st of month) -> monthly Period index."""
    out = s.copy()
    out.index = out.index.to_period("M")
    return out.groupby(level=0).last()


def load_data(fetch=fetch_fred, start: str = "2000-01", max_ffill_months: int = 4,
              drop_incomplete_month: bool = True, verbose: bool = True):
    """
    Returns
    -------
    spot  : DataFrame (monthly, USD per 1 foreign unit), columns = currencies
    rates : DataFrame (monthly, % p.a.), columns = currencies + 'USD'

    `max_ffill_months`: OECD rates are published with a lag of 1-3 months;
    we carry the last value forward for at most this many months
    (the 3M rate is a very persistent variable, so this is a mild assumption).
    `drop_incomplete_month`: drop the current calendar month (its "month-end"
    price would just be today's price, not a real month-end).
    """
    spot, rates = {}, {}
    for ccy, (sid, invert) in FX_SERIES.items():
        try:
            s = fetch(sid)
            s = 1.0 / s if invert else s
            spot[ccy] = to_monthly_last(s)
        except Exception as e:  # keep going if one series fails
            print(f"[warning] FX {ccy} ({sid}) not downloaded: {e}")
    for ccy, sid in RATE_SERIES.items():
        try:
            rates[ccy] = to_monthly_period(fetch(sid))
        except Exception as e:
            print(f"[warning] rate {ccy} ({sid}) not downloaded: {e}")

    if "USD" not in rates or not spot:
        raise RuntimeError(
            "Download failed (USD rate or all FX series missing). If you see timeouts, "
            "set a free FRED API key: https://fredaccount.stlouisfed.org/apikeys")

    spot = pd.DataFrame(spot).sort_index()
    rates = pd.DataFrame(rates).sort_index()

    if verbose:
        print("Last available observation per series:")
        last = pd.concat(
            [spot.apply(lambda c: c.last_valid_index()).rename("FX"),
             rates.apply(lambda c: c.last_valid_index()).rename("3M rate")],
            axis=1,
        )
        print(last.to_string())

    if drop_incomplete_month:
        this_month = pd.Timestamp.today().to_period("M")
        spot = spot.loc[spot.index < this_month]

    idx = spot.index
    rates =rates.reindex(idx.union(rates.index)).ffill(limit=max_ffill_months).reindex(idx)
    start_p = pd.Period(start, "M")
    return spot.loc[spot.index >= start_p], rates.loc[rates.index >= start_p]


# --------------------------------------------------------------------------
# 2. Returns and signals
# --------------------------------------------------------------------------

def compute_carry(rates: pd.DataFrame) -> pd.DataFrame:
    """Interest differential i* - i_USD (% p.a.), known at end of month t."""
    ccys = [c for c in rates.columns if c != "USD"]
    return rates[ccys].sub(rates["USD"], axis=0)


def compute_excess_returns(spot: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """
    xr[t] = return earned DURING month t by being long foreign / short USD,
            using the interest differential known at end of month t-1.
    """
    carry = compute_carry(rates)[spot.columns]
    spot_ret = np.log(spot).diff()
    return spot_ret + carry.shift(1) / 100.0 / 12.0


def compute_momentum(xr: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Cumulative excess return over the last `lookback` months (known at t)."""
    return xr.rolling(lookback, min_periods=lookback).sum()


# --------------------------------------------------------------------------
# 3. Portfolio construction
# --------------------------------------------------------------------------

def carry_weights(carry: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Long the n highest-yielders, short the n lowest. Weights +-1/n per leg."""
    w = pd.DataFrame(0.0, index=carry.index, columns=carry.columns)
    for t, row in carry.iterrows():
        row = row.dropna()
        if len(row) < 2 * n:
            continue
        ranked = row.sort_values()
        w.loc[t, ranked.index[-n:]] = 1.0 / n   # high yielders: LONG
        w.loc[t, ranked.index[:n]] = -1.0 / n   # low yielders (funding): SHORT
    return w


def momentum_filter(w: pd.DataFrame, mom: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a carry position only if recent momentum agrees with it:
      long  & momentum < 0  -> flat
      short & momentum > 0  -> flat
    """
    mom = mom.reindex_like(w)
    keep_long = (w > 0) & (mom > 0)
    keep_short = (w < 0) & (mom < 0)
    return w.where(keep_long | keep_short, 0.0)


def ts_momentum_weights(mom: pd.DataFrame) -> pd.DataFrame:
    """Pure time-series momentum: sign of past return, equal weight."""
    sig = np.sign(mom)
    n_avail = sig.notna().sum(axis=1).replace(0, np.nan)
    return sig.div(n_avail, axis=0).fillna(0.0)


def vol_target(port_ret_gross: pd.Series, target_vol: float = 0.10,
               window: int = 24, max_lev: float = 3.0) -> pd.Series:
    """
    Leverage applied in month t+1 = target / realised vol up to month t.
    Uses only past data (shift(1)).
    """
    realised = port_ret_gross.rolling(window, min_periods=12).std() * np.sqrt(12)
    lev = (target_vol / realised).clip(upper=max_lev).shift(1)
    return lev.fillna(0.0)


def backtest(weights: pd.DataFrame, xr: pd.DataFrame, cost_bps: float = 3.0,
             leverage: pd.Series | None = None) -> pd.DataFrame:
    """
    weights at end of month t are applied to xr of month t+1.
    Transaction cost = cost_bps * turnover (sum of |dw|).
    """
    w = weights.reindex(xr.index).fillna(0.0)
    if leverage is not None:
        w = w.mul(leverage.reindex(w.index).fillna(0.0), axis=0)
    w_lag = w.shift(1).fillna(0.0)
    gross = (w_lag * xr.fillna(0.0)).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0)
    cost = turnover * cost_bps / 10_000
    net = gross - cost
    return pd.DataFrame({"gross": gross, "cost": cost, "net": net,
                         "turnover": turnover})


# --------------------------------------------------------------------------
# 4. Performance metrics
# --------------------------------------------------------------------------

def perf_stats(r: pd.Series) -> pd.Series:
    r = r.dropna()
    r = r[r.ne(0).cummax()]          # drop leading zeros (warm-up period)
    cum = r.cumsum()
    dd = cum - cum.cummax()
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    return pd.Series({
        "Ann. return": ann_ret,
        "Ann. vol": ann_vol,
        "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Skewness": r.skew(),
        "Worst month": r.min(),
        "Hit rate": (r > 0).mean(),
        "Months": len(r),
    })


def drawdown(r: pd.Series) -> pd.Series:
    cum = r.fillna(0).cumsum()
    return cum - cum.cummax()


# --------------------------------------------------------------------------
# 5. Convenience: full pipeline
# --------------------------------------------------------------------------

def run_all(spot, rates, n=3, mom_lookback=3, cost_bps=3.0, target_vol=None):
    carry = compute_carry(rates)[spot.columns]
    xr = compute_excess_returns(spot, rates)
    mom = compute_momentum(xr, mom_lookback)

    w = {
        "Carry": carry_weights(carry, n),
        "Momentum (TS)": ts_momentum_weights(mom),
    }
    w["Carry + Mom filter"] = momentum_filter(w["Carry"], mom)

    results = {}
    for name, wt in w.items():
        lev = None
        if target_vol is not None:
            gross = backtest(wt, xr, cost_bps=0.0)["gross"]
            lev = vol_target(gross, target_vol)
        results[name] = backtest(wt, xr, cost_bps=cost_bps, leverage=lev)
    return {"carry": carry, "xr": xr, "mom": mom, "weights": w, "results": results}


def current_targets(spot, rates, n=3, mom_lookback=3) -> pd.DataFrame:
    """Target weights for NEXT month, from the latest complete month."""
    out = run_all(spot, rates, n=n, mom_lookback=mom_lookback)
    t = out["weights"]["Carry + Mom filter"].index[-1]
    df = pd.DataFrame({
        "carry_%": out["carry"].loc[t],
        "momentum_%": out["mom"].loc[t] * 100,
        "carry_weight": out["weights"]["Carry"].loc[t],
        "final_weight": out["weights"]["Carry + Mom filter"].loc[t],
    })
    df.attrs["as_of"] = str(t)
    return df.sort_values("carry_%", ascending=False)
