"""
Defending the krone — core library.

Can exchange-market pressure on the Danish krone predict Danmarks Nationalbank's
unilateral policy-rate moves, i.e. changes in its rate that are NOT a response to the ECB?

Data
----
* Statistics Denmark / Danmarks Nationalbank Statbank API (no key):
    - DNRENTD: Nationalbank rate on certificates of deposit (monetary policy deposit rate), daily
    - DNSNB2 : Nationalbank FX interventions, net purchase of foreign exchange (DKK bn), monthly
* FRED (free API key):
    - DEXDNUS (DKK per USD) and DEXUSEU (USD per EUR) -> EUR/DKK, daily
    - ECBMRRFR (ECB main refinancing rate) and ECBDFR (ECB deposit facility rate), daily

Conventions
-----------
* Monthly frequency, pandas Period('M') index, month-end values.
* EUR/DKK deviation from the ERM II central rate (7.46038): negative = strong krone.
* Interventions > 0 = Nationalbank BUYS foreign exchange (sells kroner) = resists APPRECIATION.
* Policy spread = Nationalbank deposit rate - ECB reference rate.
* Signals at end of month t are used to predict events in month t+1.
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

CENTRAL_RATE = 7.46038          # ERM II central rate, DKK per EUR
BAND = 2.25                     # ERM II fluctuation band, +-%
ECB_REF_SWITCH = "2014-06"      # before: compare with ECB MRO rate; from then: deposit facility rate
# FRED's ECBMRRFR covers fixed-rate tenders only, so it is flat during the variable-rate tender
# period (June 2000 - October 2008). Before 9 Oct 2008 the ECB corridor was symmetric
# (MRO = deposit rate + 1pp), so the MRO is reconstructed from the deposit facility rate.
MRO_FIXED_RATE_FROM = "2008-10-15"
MRO_FROM_DFR_UNTIL = "2008-10-08"
MIN_PERSISTENCE_DAYS = 10       # spread changes reversed within this window are timing mismatches
# Technical adjustment of the Nationalbank's instruments (19 March 2021): the CD rate rose
# from -0.60% to -0.50% while the current-account rate fell; not a change in the policy stance.
TECHNICAL_DATES = [("2021-03-15", "2021-03-25")]

STATBANK_URL = "https://api.statbank.dk/v1/data/{table}/CSV?lang=en&valuePresentation=Code&{query}"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
FRED_API_URL = ("https://api.stlouisfed.org/fred/series/observations"
                "?series_id={}&api_key={}&file_type=json")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def _http_get(url: str, timeout: int = 90) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _retry(fn, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


def fetch_fred(series_id: str) -> pd.Series:
    """One FRED series (official API if FRED_API_KEY is set, else keyless CSV)."""
    def go():
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
    return _retry(go)


def _parse_statbank_time(t: pd.Series) -> pd.DatetimeIndex:
    t = t.astype(str)
    if t.str.contains("D").any():                                   # daily: 2015M01D15
        return pd.to_datetime(t.str.replace("M", "-").str.replace("D", "-"), format="%Y-%m-%d")
    return pd.to_datetime(t.str.replace("M", "-") + "-01", format="%Y-%m-%d")   # monthly: 2015M01


def fetch_statbank(table: str, **filters) -> pd.Series:
    """
    One series from the Statistics Denmark Statbank API.
    `filters` select one value per variable, e.g. INSTRUMENT="OIBNAA", OPGOER="E".
    """
    query = "&".join([f"{k}={v}" for k, v in filters.items()] + ["Tid=*"])

    def go():
        raw = _http_get(STATBANK_URL.format(table=table, query=query))
        df = pd.read_csv(io.StringIO(raw), sep=";", dtype=str)
        val = pd.to_numeric(df["INDHOLD"].str.replace(",", "."), errors="coerce")
        s = pd.Series(val.values, index=_parse_statbank_time(df["TID"])).dropna().sort_index()
        s.name = table
        return s
    return _retry(go)


def _month_end(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("M")).last()


def mro_equivalent(ecb_mro: pd.Series, ecb_dfr: pd.Series) -> pd.Series:
    """Daily MRO: deposit rate + 1pp until 8 Oct 2008, FRED fixed-rate MRO from 15 Oct 2008."""
    days = pd.date_range(min(ecb_mro.index.min(), ecb_dfr.index.min()),
                         max(ecb_mro.index.max(), ecb_dfr.index.max()), freq="D")
    dfr_d = ecb_dfr.reindex(days).ffill()
    early = (dfr_d + 1.0).loc[:MRO_FROM_DFR_UNTIL]
    late = ecb_mro.loc[MRO_FIXED_RATE_FROM:]
    return pd.concat([early, late]).sort_index().reindex(days).ffill()


def daily_unilateral_moves(dn: pd.Series, ecb_mro: pd.Series, ecb_dfr: pd.Series) -> pd.Series:
    """
    Unilateral moves measured on DAILY data, then summed by month.

    1. On every calendar day, the unilateral change is the change in the Nationalbank rate
       minus the change in the ECB reference rate (MRO before June 2014, deposit rate after).
       Using each series' own daily change avoids an artificial jump when the reference switches.
    2. Technical adjustments of the Nationalbank's instruments are excluded.
    3. Short-lived deviations that are reversed are removed: when the Nationalbank and the
       ECB implement the same decision a few days apart (e.g. Danish +0.60 on 28 Oct 2022,
       ECB +0.75 on 2 Nov 2022), the temporary gap is removed and only the persistent
       difference (-0.15) is counted. Consecutive moves in the same direction are kept.
    """
    days = pd.date_range(dn.index.min(), max(dn.index.max(), ecb_dfr.index.max()), freq="D")
    dn_d = dn.reindex(days).ffill()
    mro_d = mro_equivalent(ecb_mro, ecb_dfr).reindex(days).ffill()
    dfr_d = ecb_dfr.reindex(days).ffill()
    use_dfr = days >= pd.Timestamp(ECB_REF_SWITCH + "-01")
    d_ecb = pd.Series(np.where(use_dfr, dfr_d.diff(), mro_d.diff()), index=days)
    d_dn = dn_d.diff()
    for a, b in TECHNICAL_DATES:
        d_dn.loc[a:b] = 0.0
    start = max(ecb_mro.index.min(), ecb_dfr.index.min(), dn.index.min())
    u = (d_dn - d_ecb).loc[start:].fillna(0.0).round(4)
    cum = u.cumsum().round(4)

    # Remove short-lived levels that are REVERSED: a level lasting less than
    # MIN_PERSISTENCE_DAYS, entered by a move in one direction and left by a move in the
    # opposite direction, is a timing mismatch with the ECB. Consecutive moves in the
    # same direction (e.g. the four cuts of 19 Jan - 5 Feb 2015) are genuine and kept.
    run_id = (cum != cum.shift()).cumsum()
    runs = cum.groupby(run_id).agg(["first", "size"])
    level, length = runs["first"].values, runs["size"].values
    transient = np.zeros(len(runs), dtype=bool)
    for k in range(1, len(runs) - 1):
        enter, leave = level[k] - level[k - 1], level[k + 1] - level[k]
        transient[k] = length[k] < MIN_PERSISTENCE_DAYS and np.sign(enter) != np.sign(leave)
    keep = pd.Series(~transient, index=runs.index).reindex(run_id.values).values
    clean = cum.where(keep).ffill().fillna(0.0)
    daily = clean.diff().fillna(0.0).round(4)
    return daily.groupby(daily.index.to_period("M")).sum().round(4)

def load_data(fetch_fred=fetch_fred, fetch_statbank=fetch_statbank, start: str = "1999-01",
              drop_incomplete_month: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Monthly DataFrame: eurdkk, dev_pct, dn_rate, ecb_ref, spread, interventions."""
    eurdkk = fetch_fred("DEXDNUS") * fetch_fred("DEXUSEU")          # DKK per USD x USD per EUR
    ecb_mro, ecb_dfr = fetch_fred("ECBMRRFR"), fetch_fred("ECBDFR")
    dn = fetch_statbank("DNRENTD", INSTRUMENT="OIBNAA", OPGOER="E")    # CD rate, daily
    interv = fetch_statbank("DNSNB2", INSTRUMENT="N7A", SPECNAT="I")   # monthly, DKK bn

    df = pd.DataFrame({
        "eurdkk": _month_end(eurdkk.dropna()),
        "dn_rate": _month_end(dn),
        "ecb_mro": _month_end(mro_equivalent(ecb_mro, ecb_dfr)),
        "ecb_dfr": _month_end(ecb_dfr),
    })
    iv = interv.copy()
    iv.index = iv.index.to_period("M")
    df["interventions"] = iv.groupby(level=0).last()

    if verbose:
        print("Last available observation per series:")
        print(df.apply(lambda c: c.last_valid_index()).to_string())

    if drop_incomplete_month:
        df = df.loc[df.index < pd.Timestamp.today().to_period("M")]
    df = df.loc[df.index >= pd.Period(start, "M")]
    df[["dn_rate", "ecb_mro", "ecb_dfr"]] = df[["dn_rate", "ecb_mro", "ecb_dfr"]].ffill()

    df["unilateral"] = daily_unilateral_moves(dn, ecb_mro, ecb_dfr).reindex(df.index).fillna(0.0)

    switch = pd.Period(ECB_REF_SWITCH, "M")
    df["ecb_ref"] = np.where(df.index < switch, df["ecb_mro"], df["ecb_dfr"])
    df["spread"] = df["dn_rate"] - df["ecb_ref"]
    df["dev_pct"] = (df["eurdkk"] / CENTRAL_RATE - 1) * 100
    return df.dropna(subset=["eurdkk", "dn_rate", "ecb_ref"])


# --------------------------------------------------------------------------
# 2. Unilateral moves and exchange-market pressure
# --------------------------------------------------------------------------

def unilateral_moves(df: pd.DataFrame, min_bp: float = 5.0) -> pd.Series:
    """
    Unilateral move in month t, in percentage points: the persistent change in the
    Nationalbank-ECB policy spread during month t (see daily_unilateral_moves).
    Moves smaller than `min_bp` basis points are ignored.
    """
    u = df["unilateral"].round(4)
    return u.where(u.abs() >= min_bp / 100, 0.0)


def _expanding_z(x: pd.Series, min_periods: int = 24) -> pd.Series:
    """Standardise using only past data (expanding mean and std up to t)."""
    mu = x.expanding(min_periods).mean()
    sd = x.expanding(min_periods).std()
    return (x - mu) / sd


def pressure_index(df: pd.DataFrame, min_periods: int = 24) -> pd.DataFrame:
    """
    Exchange-market pressure (EMP), in the spirit of Girton & Roper (1977) and
    Eichengreen, Rose & Wyplosz (1996), adapted to a hard peg:
        EMP_t = z(-deviation_t) + z(interventions_t)
    Positive EMP = APPRECIATION pressure on the krone (strong krone and/or the
    Nationalbank buying foreign exchange). Negative = DEPRECIATION pressure.
    Interest-rate changes are excluded because they are the outcome we want to predict.
    Standardisation is expanding, so the index uses no future information.
    """
    strength = -df["dev_pct"]
    iv = df["interventions"].fillna(0.0)
    out = pd.DataFrame({
        "z_strength": _expanding_z(strength, min_periods),
        "z_interventions": _expanding_z(iv, min_periods),
    })
    out["emp"] = out["z_strength"] + out["z_interventions"]
    return out


def event_table(df: pd.DataFrame, moves: pd.Series, emp: pd.DataFrame) -> pd.DataFrame:
    """One row per unilateral move, with the pressure readings at the end of the previous month."""
    ev = moves[moves != 0]
    rows = []
    for t, m in ev.items():
        prev = t - 1
        rows.append({
            "month": str(t),
            "direction": "hike" if m > 0 else "cut",
            "move_bp": round(m * 100),
            "spread_after_%": df.loc[t, "spread"],
            "EUR/DKK dev. t-1 (%)": df.loc[prev, "dev_pct"] if prev in df.index else np.nan,
            "interventions t-1 (DKK bn)": df.loc[prev, "interventions"] if prev in df.index else np.nan,
            "interventions t (DKK bn)": df.loc[t, "interventions"],
            "EMP t-1": emp.loc[prev, "emp"] if prev in emp.index else np.nan,
        })
    return pd.DataFrame(rows).set_index("month")


# --------------------------------------------------------------------------
# 3. Statistical tests
# --------------------------------------------------------------------------

def event_time_profile(emp: pd.Series, moves: pd.Series, window=(-6, 3)) -> pd.DataFrame:
    """Average EMP in event time around cuts and around hikes."""
    out = {}
    for name, mask in [("cuts", moves < 0), ("hikes", moves > 0)]:
        dates = moves.index[mask]
        prof = {}
        for k in range(window[0], window[1] + 1):
            vals = [emp.get(d + k, np.nan) for d in dates]
            prof[k] = np.nanmean(vals) if len(vals) else np.nan
        out[f"{name} (n={len(dates)})"] = pd.Series(prof)
    out = pd.DataFrame(out)
    out.index.name = "months relative to move"
    return out


def predictive_regression(df: pd.DataFrame, emp: pd.Series, moves: pd.Series, lags: int = 3):
    """
    Next month's unilateral move (pp) on today's EMP and policy spread:
        move_{t+1} = a + b * EMP_t + c * spread_t + e
    Expected: b < 0 (appreciation pressure -> cut), c < 0 (a wide spread is later normalised).
    Newey-West standard errors.
    """
    import statsmodels.api as sm
    data = pd.concat([moves.shift(-1).rename("move_next"), emp.rename("EMP"),
                      df["spread"].rename("spread")], axis=1).dropna()
    X = sm.add_constant(data[["EMP", "spread"]])
    res = sm.OLS(data["move_next"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return pd.DataFrame({"coef": res.params, "t-stat (NW)": res.tvalues}).assign(
        R2=res.rsquared, N=int(res.nobs))


def logit_events(emp: pd.Series, moves: pd.Series, direction: str = "cut"):
    """Probability of a unilateral cut (or hike) next month as a function of EMP today."""
    import statsmodels.api as sm
    y = (moves.shift(-1) < 0) if direction == "cut" else (moves.shift(-1) > 0)
    data = pd.concat([y.astype(float).rename("y"), emp.rename("EMP")], axis=1).dropna()
    data = data.iloc[:-1]                                       # last month has no 'next'
    try:
        res = sm.Logit(data["y"], sm.add_constant(data["EMP"])).fit(disp=0)
        return pd.Series({"coef EMP": res.params["EMP"], "t-stat": res.tvalues["EMP"],
                          "pseudo R2": res.prsquared, "events": int(data["y"].sum()),
                          "N": int(res.nobs)}), res
    except Exception as e:
        print(f"Logit for {direction}s could not be estimated: {e}")
        return pd.Series(dtype=float), None


def classification(emp: pd.Series, moves: pd.Series, thresholds=(1.0, 1.5, 2.0), horizon: int = 3):
    """
    Rule: 'expect a cut within the next `horizon` months' when EMP_t > k
          'expect a hike within the next `horizon` months' when EMP_t < -k.
    Precision = share of signals followed by a move in the predicted direction.
    Recall    = share of moves that were preceded by a signal.
    """
    rows = []
    fut_cut = pd.concat([(moves.shift(-h) < 0) for h in range(1, horizon + 1)], axis=1).any(axis=1)
    fut_hike = pd.concat([(moves.shift(-h) > 0) for h in range(1, horizon + 1)], axis=1).any(axis=1)
    past_sig = lambda sig: pd.concat([sig.shift(h, fill_value=False) for h in range(1, horizon + 1)], axis=1).any(axis=1)
    for k in thresholds:
        for name, sig, fut, mv in [("cut", emp > k, fut_cut, moves < 0), ("hike", emp < -k, fut_hike, moves > 0)]:
            n_sig = int(sig.sum())
            precision = fut[sig].mean() if n_sig else np.nan
            recall = past_sig(sig)[mv].mean() if mv.sum() else np.nan
            rows.append({"threshold": k, "direction": name, "signal months": n_sig,
                         "precision": precision, "recall": recall, "moves": int(mv.sum())})
    return pd.DataFrame(rows).set_index(["threshold", "direction"])


# --------------------------------------------------------------------------
# 4. Stylised trade
# --------------------------------------------------------------------------

def spread_trade(emp: pd.Series, moves: pd.Series, scale: float = 2.0, band: float = 1.0) -> pd.DataFrame:
    """
    Stylised position in the DKK-EUR short-rate spread (e.g. a 1-year DKK vs EUR swap spread):
        position_t = clip(EMP_t / scale, -1, 1) if |EMP_t| > band, else 0
        P&L_{t+1}  = position_t * (-unilateral move_{t+1}) * 100     (basis points, duration = 1)
    Positive position = receive DKK / pay EUR: gains when the Nationalbank cuts unilaterally.
    This measures the predictability of policy moves. It is an upper bound on tradable
    profits, because market prices partly anticipate the moves (see limitations).
    """
    pos = (emp / scale).clip(-1, 1).where(emp.abs() > band, 0.0).fillna(0.0)
    pnl = pos.shift(1) * (-moves) * 100
    return pd.DataFrame({"position": pos, "pnl_bp": pnl.fillna(0.0)})


def current_reading(df: pd.DataFrame, emp: pd.DataFrame) -> pd.Series:
    t = df.index[-1]
    return pd.Series({
        "as_of": str(t),
        "EUR/DKK": round(df.loc[t, "eurdkk"], 4),
        "deviation from central rate (%)": round(df.loc[t, "dev_pct"], 3),
        "interventions this month (DKK bn)": df.loc[t, "interventions"],
        "Nationalbank rate (%)": df.loc[t, "dn_rate"],
        "ECB deposit rate (%)": df.loc[t, "ecb_dfr"],
        "policy spread (pp)": round(df.loc[t, "spread"], 2),
        "EMP": round(emp.loc[t, "emp"], 2),
    })
