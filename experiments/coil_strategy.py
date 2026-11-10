"""
COIL - gold quiet-day breakout with a 1:3 exit (research card, 2026-11-10).

Same rules as experiments/gold_own_ideas.py (where it was tested: dev 2004-17 +24.7R PF 1.22, val 2017-22 +10.5R
PF 1.39, but only 2 of 4 val quarters positive -> NOT passed). Kept in its own small module so the cloud app can
run it without the study's heavy imports; experiments/verify_coil_module.py checks both give identical signals.

  - Coil day D = a completed UTC day whose high-low range is the smallest of the last 7 days (itself included).
  - In the 48 hours after D ends, the FIRST completed 1h candle that closes outside D's range decides: if it agrees
    with the daily trend -> trade it at that 1h close; otherwise no trade for this coil.
  - Stop = the other side of D. Target = 3 x the stop distance. Stop never moved. Out after 120 hours.
"""
import numpy as np
import pandas as pd

import mtf_strategies as mt

HOLD_H = 120.0
WINDOW_H = 48


def coil(m5, f):
    d = mt.tf_bars(f, "1D")
    h1 = mt.tf_bars(f, "1h")
    rng = (d["high"] - d["low"])
    nr7 = (rng <= rng.rolling(7).min() + 1e-12).to_numpy() & np.isfinite(rng.rolling(7).min().to_numpy())
    dh, dl = d["high"].to_numpy(), d["low"].to_numpy()
    dend = (d.index + pd.Timedelta(days=1)).to_numpy()             # the UTC day is over at the next midnight
    t1 = (h1.index + pd.Timedelta(hours=1)).to_numpy()             # a 1h candle is complete at its end
    c1, p1 = h1["close"].to_numpy(), h1["pos"].to_numpy(np.int64)
    bull, bear = f["daily_bull"], f["daily_bear"]
    P, BUY, STOP = [], [], []
    for i in np.flatnonzero(nr7):
        a = int(np.searchsorted(t1, dend[i], side="right"))        # first 1h candle that ENDS after the day is over
        b = int(np.searchsorted(t1, dend[i] + np.timedelta64(WINDOW_H, "h"), side="right"))
        if a >= b:
            continue
        cc = c1[a:b]
        out = np.flatnonzero((cc > dh[i]) | (cc < dl[i]))
        if not len(out):
            continue
        j = a + int(out[0])
        up = c1[j] > dh[i]
        if (up and bull[p1[j]]) or ((not up) and bear[p1[j]]):
            P.append(p1[j]); BUY.append(up); STOP.append(dl[i] if up else dh[i])
    return mt._emit(f, P, BUY, STOP, 3.0, 3.0, HOLD_H, "no_breakeven")


def state(f) -> dict:
    """For the screen: the latest quiet day, its range and whether its 48-hour window is still open."""
    d = mt.tf_bars(f, "1D")
    rng = d["high"] - d["low"]
    mn = rng.rolling(7).min()
    days = d.index[(rng <= mn + 1e-12) & mn.notna()]
    out = {}
    if not len(days):
        return {"quiet day": "none in the last days"}
    last = days[-1]
    end = last + pd.Timedelta(days=1)
    now = f["idx"][-1] + pd.Timedelta(minutes=5)
    out["quiet day"] = f"{last:%a %d %b} (range {float(rng.loc[last]):.2f})"
    out["quiet day range"] = f"{float(d.loc[last, 'low']):.2f} - {float(d.loc[last, 'high']):.2f}"
    if now < end:
        out["window"] = "the quiet day is still running"
    elif now <= end + pd.Timedelta(hours=WINDOW_H):
        out["window"] = f"open until {end + pd.Timedelta(hours=WINDOW_H):%a %H:%M} UTC"
    else:
        out["window"] = "closed - waiting for the next quiet day"
    return out
