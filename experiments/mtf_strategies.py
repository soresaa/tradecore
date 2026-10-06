"""
Timeframe-generic strategy builders for improve_lab.py (round 4).

Signals are decided on COMPLETED bars of a chosen timeframe (15min, 1h, 4h, 1D,
built from the 5-minute data in UTC) and entered at the close of that bar's
last 5-minute candle; the trade is then simulated on 5-minute candles, so the
stop / target order inside a big bar is exact.

Base names (use them as spec["base"] in improve_lab):
  DON:<tf>:<N>        Donchian breakout: close beyond the N-bar high/low, WITH the daily trend
  DONX:<tf>:<N>       same, no trend filter
  TRAP:<tf>:<LB>:<CW> the C05 trap on <tf> bars: hammer/star at a fresh LB-bar extreme
                      AGAINST the daily trend, broken by a close within CW bars -> trade with the trend
  SQZ:<tf>            squeeze: ATR in the quietest fifth of the last 120 bars, then a 12-bar range break
  PB:<tf>             trend pullback: EMA20 > EMA50 (tf), the bar dips to EMA20 and closes back above it
                      in the upper half of its range (mirror for shorts), with the daily trend
  MAX:<tf>            EMA20 / EMA50 cross on tf closes, with the daily trend

Default exits (all can be overridden by a spec's "exits"):
  DON / SQZ / MAX : stop 2.0 x ATR(tf), 2R -> stop to entry, 4R, hold 30 bars
  TRAP            : stop beyond the trap candle (>= 1.2 ATR, skip > 3.5 ATR), 1R -> entry, 2.5R, hold 9 bars
  PB              : stop 1.5 x ATR(tf), 1R -> entry, 2.5R, hold 20 bars
Completion rule: a tf bar counts when it has at least half of its 5-minute slots.
Its signal bar is its last 5-minute candle. (A bar cut short by a session break
is closed at its last candle, as a chart shows it.)
"""
import numpy as np
import pandas as pd

import trap_candidates as tc
from deep_search_study import sig_frame

TF_HOURS = {"15min": 0.25, "1h": 1.0, "4h": 4.0, "1D": 24.0}


def tf_bars(f, tf):
    key = f"_tf_{tf}"
    if key in f:
        return f[key]
    idx = f["idx"]
    k = idx.floor(tf)
    df = pd.DataFrame({"open": f["open"], "high": f["high"], "low": f["low"], "close": f["close"],
                       "pos": np.arange(len(idx))}, index=idx)
    g = df.groupby(k)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "n": g["pos"].size(), "last": g["pos"].last()})
    slots = int(round(TF_HOURS[tf] * 12))
    b = b[b["n"] >= max(1, slots // 2)].copy()
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    b["A"] = tr.rolling(14).mean()                       # ATR(14) of completed tf bars, incl. this one
    b["pos"] = b["last"].astype(np.int64)
    b["e20"] = b["close"].ewm(span=20, adjust=False).mean()
    b["e50"] = b["close"].ewm(span=50, adjust=False).mean()
    f[key] = b
    return b


def _emit(f, pos, buy, stop, r1, r2, hold, rule):
    idx, c = f["idx"], f["close"]
    pos = np.asarray(pos, np.int64)
    buy = np.asarray(buy, bool)
    stop = np.asarray(stop, float)
    e = c[pos]
    risk = np.abs(e - stop)
    sg = np.where(buy, 1.0, -1.0)
    ok = np.isfinite(risk) & (risk > 0)
    pos, buy, stop, e, risk, sg = pos[ok], buy[ok], stop[ok], e[ok], risk[ok], sg[ok]
    # two signals on one bar: same direction keeps the first, opposite drops both (as tc.emit)
    return tc.emit(f, pos, buy, stop, e + sg * r1 * risk, e + sg * r2 * risk, hold, rule)


def don(tf, N, trend=True):
    def build(m5, f):
        b = tf_bars(f, tf)
        hi = b["high"].shift(1).rolling(N).max().to_numpy()
        lo = b["low"].shift(1).rolling(N).min().to_numpy()
        c, A, p = b["close"].to_numpy(), b["A"].to_numpy(), b["pos"].to_numpy(np.int64)
        with np.errstate(invalid="ignore"):
            up, dn = c > hi, c < lo
            if trend:
                up &= f["daily_bull"][p]
                dn &= f["daily_bear"][p]
        rows = np.flatnonzero((up | dn) & np.isfinite(A))
        buy = up[rows]
        e = c[rows]
        stop = e - np.where(buy, 1.0, -1.0) * 2.0 * A[rows]
        return _emit(f, p[rows], buy, stop, 2.0, 4.0, 30 * TF_HOURS[tf], "breakeven")
    return build


def trap(tf, LB=12, CW=4):
    def build(m5, f):
        b = tf_bars(f, tf)
        o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
        A, p = b["A"].to_numpy(), b["pos"].to_numpy(np.int64)
        n = len(b)
        rng = h - l
        plo = pd.Series(l).shift(1).rolling(LB - 1).min().to_numpy()
        phi = pd.Series(h).shift(1).rolling(LB - 1).max().to_numpy()
        with np.errstate(invalid="ignore"):
            bull = (l <= plo) & (c > o) & (c >= l + 0.67 * rng) & (rng >= A) & f["daily_bear"][p]
            bear = (h >= phi) & (c < o) & (c <= h - 0.67 * rng) & (rng >= A) & f["daily_bull"][p]
        P, BUY, STOP = [], [], []
        for setups, short in ((np.flatnonzero(bull), True), (np.flatnonzero(bear), False)):
            X = np.full(len(setups), -1, np.int64)
            for kk in range(CW, 0, -1):
                x = setups + kk
                okx = x < n
                xc = np.where(okx, x, 0)
                with np.errstate(invalid="ignore"):
                    cond = okx & ((c[xc] < l[setups]) if short else (c[xc] > h[setups]))
                X = np.where(cond, x, X)
            m = X >= 0
            E, Xm = setups[m], X[m]
            a, e = A[Xm], c[Xm]
            st = np.maximum(h[E] + 0.2 * a, e + 1.2 * a) if short else np.minimum(l[E] - 0.2 * a, e - 1.2 * a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - st) <= 3.5 * a
            P.append(p[Xm][ok]); BUY.append(np.full(ok.sum(), not short)); STOP.append(st[ok])
        return _emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), 1.0, 2.5,
                     9 * TF_HOURS[tf], "breakeven")
    return build


def sqz(tf):
    def build(m5, f):
        b = tf_bars(f, tf)
        c, A, p = b["close"].to_numpy(), b["A"], b["pos"].to_numpy(np.int64)
        q20 = A.shift(1).rolling(120, min_periods=60).quantile(0.2).to_numpy()
        A = A.to_numpy()
        hi = b["high"].shift(1).rolling(12).max().to_numpy()
        lo = b["low"].shift(1).rolling(12).min().to_numpy()
        pc = b["close"].shift(1).to_numpy()
        with np.errstate(invalid="ignore"):
            quiet = pd.Series(A).shift(1).to_numpy() <= q20
            up = quiet & (c > hi) & (pc <= hi)
            dn = quiet & (c < lo) & (pc >= lo)
        rows = np.flatnonzero((up | dn) & np.isfinite(A))
        buy = up[rows]
        e = c[rows]
        stop = e - np.where(buy, 1.0, -1.0) * 2.0 * A[rows]
        return _emit(f, p[rows], buy, stop, 2.0, 4.0, 30 * TF_HOURS[tf], "breakeven")
    return build


def pb(tf):
    def build(m5, f):
        b = tf_bars(f, tf)
        o, h, l, c = (b[x].to_numpy() for x in ("open", "high", "low", "close"))
        A, p = b["A"].to_numpy(), b["pos"].to_numpy(np.int64)
        e20, e50 = b["e20"].to_numpy(), b["e50"].to_numpy()
        with np.errstate(invalid="ignore"):
            up = (e20 > e50) & (l <= e20) & (c > e20) & (c >= l + 0.5 * (h - l)) & f["daily_bull"][p]
            dn = (e20 < e50) & (h >= e20) & (c < e20) & (c <= h - 0.5 * (h - l)) & f["daily_bear"][p]
        rows = np.flatnonzero((up | dn) & np.isfinite(A))
        buy = up[rows]
        stop = c[rows] - np.where(buy, 1.0, -1.0) * 1.5 * A[rows]
        return _emit(f, p[rows], buy, stop, 1.0, 2.5, 20 * TF_HOURS[tf], "breakeven")
    return build


def macross(tf):
    def build(m5, f):
        b = tf_bars(f, tf)
        c, A, p = b["close"].to_numpy(), b["A"].to_numpy(), b["pos"].to_numpy(np.int64)
        d = (b["e20"] - b["e50"]).to_numpy()
        dp = np.concatenate([[np.nan], d[:-1]])
        with np.errstate(invalid="ignore"):
            up = (d > 0) & (dp <= 0) & f["daily_bull"][p]
            dn = (d < 0) & (dp >= 0) & f["daily_bear"][p]
        rows = np.flatnonzero((up | dn) & np.isfinite(A))
        buy = up[rows]
        stop = c[rows] - np.where(buy, 1.0, -1.0) * 2.0 * A[rows]
        return _emit(f, p[rows], buy, stop, 2.0, 4.0, 30 * TF_HOURS[tf], "breakeven")
    return build


def from_name(name):
    """'DON:4h:20' -> builder; None if the name is not a timeframe base."""
    parts = name.split(":")
    kind = parts[0]
    if kind not in ("DON", "DONX", "TRAP", "SQZ", "PB", "MAX"):
        import trend_bases                   # round 5: ST, MACD, ADX, PSAR, ICHI, KC, HMA, AROON, CONS
        return trend_bases.from_name(name)
    tf = parts[1]
    if tf not in TF_HOURS:
        raise ValueError(f"timeframe must be one of {list(TF_HOURS)}")
    if kind in ("DON", "DONX"):
        return don(tf, int(parts[2]) if len(parts) > 2 else 20, trend=(kind == "DON"))
    if kind == "TRAP":
        return trap(tf, int(parts[2]) if len(parts) > 2 else 12, int(parts[3]) if len(parts) > 3 else 4)
    if kind == "SQZ":
        return sqz(tf)
    if kind == "PB":
        return pb(tf)
    return macross(tf)
