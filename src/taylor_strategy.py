"""
Taylor Rule Duration Strategy — core library.

Idea: the Taylor rule gives a benchmark for where the policy rate "should" be.
If the market (2-year Treasury yield) or the Fed (fed funds rate) sits far below
that benchmark, policy is likely to tighten more than priced -> yields rise ->
short duration. If they sit far above it -> yields fall -> long duration.

Conventions
-----------
* All rates in percent per annum.
* Monthly frequency, indexed by pandas Period('M').
* A signal formed at the end of month t is applied to the return of month t+1.
* Macro data are lagged to reflect publication delays (no look-ahead on timing).
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
# 1. Data
# --------------------------------------------------------------------------

SERIES = {
    "core_pce": "PCEPILFE",   # Core PCE price index, monthly (Fed's preferred inflation gauge)
    "unrate":   "UNRATE",     # Unemployment rate, monthly
    "nrou":     "NROU",       # CBO noncyclical rate of unemployment, quarterly
    "fedfunds": "FEDFUNDS",   # Effective federal funds rate, monthly average
    "y2":       "DGS2",       # 2-year Treasury constant-maturity yield, daily
    "y10":      "DGS10",      # 10-year Treasury constant-maturity yield, daily
    "tbill":    "DTB3",       # 3-month Treasury bill rate, daily (funding rate)
}

# Publication lag in months: value for month m is known at the end of month m + lag
PUB_LAG = {"core_pce": 1, "unrate": 1}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_API_URL = ("https://api.stlouisfed.org/fred/series/observations"
                "?series_id={}&api_key={}&file_type=json")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def _http_get(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def fetch_fred(series_id: str, retries: int = 3) -> pd.Series:
    """Download one FRED series (official API if FRED_API_KEY is set, else keyless CSV)."""
    last_err = None
    for attempt in range(retries):
        try:
            if FRED_API_KEY:
                obs = json.loads(_http_get(FRED_API_URL.format(series_id, FRED_API_KEY)))["observations"]
                s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
            else:
                df = pd.read_csv(io.StringIO(_http_get(FRED_CSV_URL.format(series_id))),
                                 index_col=0, parse_dates=True)
                s = df.iloc[:, 0]
            s = pd.to_numeric(s.replace(".", np.nan), errors="coerce").dropna().sort_index()
            s.name = series_id
            return s
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{series_id}: {last_err}")


def _monthly(s: pd.Series, how: str = "last") -> pd.Series:
    g = s.groupby(s.index.to_period("M"))
    return g.last() if how == "last" else g.mean()


def load_data(fetch=fetch_fred, start: str = "1987-01", drop_incomplete_month: bool = True,
              verbose: bool = True) -> pd.DataFrame:
    """
    Monthly DataFrame with:
      core_pce_yoy, unrate, nrou (lagged for publication), fedfunds, y2, y10, tbill (month-end).
    """
    raw = {k: fetch(sid) for k, sid in SERIES.items()}

    df = pd.DataFrame({
        "core_pce_yoy": _monthly(raw["core_pce"]).pct_change(12) * 100,
        "unrate": _monthly(raw["unrate"]),
        "fedfunds": _monthly(raw["fedfunds"]),
        "y2": _monthly(raw["y2"]),
        "y10": _monthly(raw["y10"]),
        "tbill": _monthly(raw["tbill"]),
    })
    # quarterly NROU -> monthly, forward-filled within the quarter
    nrou = raw["nrou"].copy()
    nrou.index = nrou.index.to_period("M")
    df["nrou"] = nrou.reindex(df.index.union(nrou.index)).ffill().reindex(df.index)

    if verbose:
        print("Last available observation per series:")
        print(df.apply(lambda c: c.last_valid_index()).to_string())

    # publication lags: inflation and unemployment of month m are known only at end of m+1
    for col, src in [("core_pce_yoy", "core_pce"), ("unrate", "unrate")]:
        df[col] = df[col].shift(PUB_LAG[src])

    if drop_incomplete_month:
        df = df.loc[df.index < pd.Timestamp.today().to_period("M")]
    # carry the latest macro print forward for up to 2 months at the end of the sample
    df[["core_pce_yoy", "unrate", "nrou"]] = df[["core_pce_yoy", "unrate", "nrou"]].ffill(limit=2)
    return df.loc[df.index >= pd.Period(start, "M")].dropna(subset=["y2", "fedfunds"])


# --------------------------------------------------------------------------
# 2. Taylor rule and signals
# --------------------------------------------------------------------------

def taylor_rate(df: pd.DataFrame, r_star: float = 2.0, pi_star: float = 2.0,
                a_pi: float = 0.5, a_gap: float = 0.5, okun: float = 2.0) -> pd.Series:
    """
    Taylor (1993):  i = r* + pi + a_pi (pi - pi*) + a_gap * output_gap
    Output gap proxied through Okun's law:  gap = okun * (u* - u)
    a_gap = 0.5 is the original rule; a_gap = 1.0 is the 'balanced-approach' rule.
    """
    pi = df["core_pce_yoy"]
    gap = okun * (df["nrou"] - df["unrate"])
    return r_star + pi + a_pi * (pi - pi_star) + a_gap * gap


def signals(df: pd.DataFrame, taylor: pd.Series, tenor: str = "y2") -> pd.DataFrame:
    """
    policy_gap = fed funds - Taylor rate   (>0: Fed tighter than the rule)
    market_gap = Treasury yield - Taylor rate (>0: market prices a path above the rule)
    Positive gap -> rates 'too high' relative to fundamentals -> expect yields to fall -> LONG duration.
    """
    return pd.DataFrame({
        "policy_gap": df["fedfunds"] - taylor,
        "market_gap": df[tenor] - taylor,
    })


# --------------------------------------------------------------------------
# 3. Treasury returns from yields
# --------------------------------------------------------------------------

def par_bond_price(coupon: np.ndarray, yld: np.ndarray, maturity: float) -> np.ndarray:
    """Price (per 1 of face) of a semi-annual bond; yields and coupon in decimals."""
    c, y = np.asarray(coupon, float), np.asarray(yld, float)
    n = 2 * maturity
    h = y / 2
    small = np.abs(h) < 1e-10
    h_safe = np.where(small, 1e-10, h)
    disc = (1 + h_safe) ** (-n)
    annuity = np.where(small, n, (1 - disc) / h_safe)
    return c / 2 * annuity + disc


def bond_excess_returns(df: pd.DataFrame, tenor: str = "y2", maturity: float = 2.0) -> pd.DataFrame:
    """
    Monthly return of a constant-maturity Treasury, approximated as a par bond bought at
    end of month t (coupon = yield_t) and revalued at end of month t+1 at yield_{t+1}:
        R_{t+1} = P(c = y_t, y = y_{t+1}) - 1 + y_t / 12
    Excess return = R_{t+1} - tbill_t / 12   (funding at the 3-month T-bill rate).
    The one-month ageing of the bond (roll-down) is ignored.
    """
    y = df[tenor] / 100
    y_prev = y.shift(1)
    price = pd.Series(par_bond_price(y_prev.values, y.values, maturity), index=df.index)
    total = price - 1 + y_prev / 12
    excess = total - df["tbill"].shift(1) / 100 / 12
    dy = (y - y_prev) * 100
    return pd.DataFrame({"total": total, "excess": excess, "dy_pp": dy})


# --------------------------------------------------------------------------
# 4. Positions and backtest
# --------------------------------------------------------------------------

def position_scaled(signal: pd.Series, scale: float = 2.0) -> pd.Series:
    """Continuous position in [-1, 1]: a 2pp gap (default) -> full long/short."""
    return (signal / scale).clip(-1, 1)


def position_sign(signal: pd.Series, band: float = 0.0) -> pd.Series:
    """+1 / -1 outside a +-band (pp), 0 inside it."""
    pos = np.sign(signal).where(signal.abs() > band, 0.0)
    return pos.where(signal.notna())


def backtest(position: pd.Series, excess: pd.Series, cost_bp: float = 1.0) -> pd.DataFrame:
    """
    position at end of month t applied to excess return of month t+1.
    cost_bp: cost in basis points of notional per unit of position change.
    """
    w = position.reindex(excess.index).fillna(0.0)
    w_lag = w.shift(1).fillna(0.0)
    gross = w_lag * excess.fillna(0.0)
    turnover = (w - w.shift(1).fillna(0.0)).abs().shift(1).fillna(0.0)
    cost = turnover * cost_bp / 10_000
    return pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                         "position": w_lag, "turnover": turnover})


# --------------------------------------------------------------------------
# 5. Statistics
# --------------------------------------------------------------------------

def perf_stats(r: pd.Series) -> pd.Series:
    r = r.dropna()
    r = r[r.ne(0).cummax()]
    cum = r.cumsum()
    dd = cum - cum.cummax()
    ann_ret, ann_vol = r.mean() * 12, r.std() * np.sqrt(12)
    return pd.Series({
        "Ann. return": ann_ret,
        "Ann. vol": ann_vol,
        "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "t-stat": (ann_ret / ann_vol) * np.sqrt(len(r) / 12) if ann_vol > 0 else np.nan,
        "Max drawdown": dd.min(),
        "Skewness": r.skew(),
        "Worst month": r.min(),
        "Hit rate": (r > 0).mean(),
        "Months": len(r),
    })


def drawdown(r: pd.Series) -> pd.Series:
    cum = r.fillna(0).cumsum()
    return cum - cum.cummax()


def predictive_regression(signal: pd.Series, yields: pd.Series, horizon: int = 12):
    """
    Regress the change in yield over the next `horizon` months on today's signal:
        y_{t+h} - y_t = a + b * signal_t + e
    Overlapping observations -> Newey-West (HAC) standard errors with `horizon` lags.
    A NEGATIVE b means: when rates sit above the Taylor benchmark, they subsequently fall.
    """
    import statsmodels.api as sm
    dy = yields.shift(-horizon) - yields
    data = pd.concat([dy.rename("dy"), signal.rename("signal")], axis=1).dropna()
    X = sm.add_constant(data["signal"])
    res = sm.OLS(data["dy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    return pd.Series({"beta": res.params["signal"], "t-stat (NW)": res.tvalues["signal"],
                      "R2": res.rsquared, "N": int(res.nobs)})


# --------------------------------------------------------------------------
# 6. Full pipeline
# --------------------------------------------------------------------------

def run_all(df: pd.DataFrame, tenor: str = "y2", maturity: float = 2.0, r_star: float = 2.0,
            a_gap: float = 0.5, scale: float = 2.0, cost_bp: float = 1.0) -> dict:
    tr = taylor_rate(df, r_star=r_star, a_gap=a_gap)
    sig = signals(df, tr, tenor)
    rets = bond_excess_returns(df, tenor, maturity)
    pos = {
        "Long duration (benchmark)": pd.Series(1.0, index=df.index),
        "Policy gap": position_scaled(sig["policy_gap"], scale),
        "Market gap": position_scaled(sig["market_gap"], scale),
    }
    res = {k: backtest(p, rets["excess"], cost_bp) for k, p in pos.items()}
    return {"taylor": tr, "signals": sig, "returns": rets, "positions": pos, "results": res}


def current_signal(df: pd.DataFrame, **kw) -> pd.Series:
    out = run_all(df, **kw)
    t = df.index[-1]
    return pd.Series({
        "as_of": str(t),
        "fed_funds_%": df.loc[t, "fedfunds"],
        "taylor_rate_%": out["taylor"].loc[t],
        "2y_yield_%": df.loc[t, "y2"],
        "policy_gap_pp": out["signals"].loc[t, "policy_gap"],
        "market_gap_pp": out["signals"].loc[t, "market_gap"],
        "position_policy_gap": out["positions"]["Policy gap"].loc[t],
        "position_market_gap": out["positions"]["Market gap"].loc[t],
    })
