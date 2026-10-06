"""
Pre-registered "trap" candidates C01-C14 (TRAP_RESEARCH_BRIEF.md + the
pre-registration relayed with it), as signal builders for trap_study.py.

Every builder is build(m5, f) -> deep_search_study.sig_frame rows (each signal
carries its own stop, targets, max hold in hours and exit rule), plus optional
diagnostic columns ('tag', 'baseline', ...) that trap_study merges back onto
the simulated trades.

Global conventions (G1-G4), applied everywhere below:
  * 5m BID bars, naive UTC stamps at bar OPEN. 'Bar hh:mm NY' = the bar whose
    New York local open is hh:mm (DST-aware via the tz database).
  * entry = close of the signal bar; the simulator starts on the next bar.
  * Hourly bars: idx.floor('1h'); an hour exists only if it has >= 10 of 12
    bars AND its minute-55 bar, and is known from that bar's close on. Every
    hourly lookback is over existing (valid) hourly rows and excludes the
    current hour. 15m bars: floor('15min'), 3 of 3 bars, complete at the
    minute%15 == 10 bar.
  * Pivots are used only after confirmation, session levels only from
    completed sessions, higher-timeframe values only from completed bars
    (best_per_market_study.map_to_5m shifts one bar).

Layout:
  CELLS        cell_id -> {candidate, params, markets, controls, factory}
  CANDIDATES   cell_id -> {market_label: builder}         (primary markets)
  CONTROLS     ctrl_id -> {candidate, kind, markets, factory}
  CONFIRM      candidate -> {primary market: [confirmation markets]}
  LEVELS       candidate -> factory(market) -> fn(m5, f) -> (up_level, dn_level)
  builder_for(cell_or_ctrl, market) builds for any market (confirmation too).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from deep_search_study import sig_frame, daily_atr, local_minutes
from best_per_market_study import map_to_5m

NY = "America/New_York"
HOUR = pd.Timedelta(hours=1)
FIVE = pd.Timedelta(minutes=5)
NO_TARGET = 10.0            # "effectively no target" multiple of risk (C02 benchmark)

DATA_FILES = {
    "GOLD": "xauusd_m5_dukascopy_23y.csv",
    "GOLD (broker)": "real_xauusd_5y.csv",
    "SILVER": "xagusd_m5_dukascopy_23y.csv",
    "BTC": "btcusd_m5_dukascopy_23y.csv",
    "BTC (Binance)": "btcusd_m5_binance.csv",
    "ETH (Binance)": "ethusd_m5_binance.csv",
    "NAS100": "nas100_m5_dukascopy_23y.csv",
    "US500": "us500_m5_dukascopy_23y.csv",
    "US30": "us30_m5_dukascopy_23y.csv",
}
KIND = {"GOLD": "metal", "GOLD (broker)": "metal", "SILVER": "metal",
        "NAS100": "index", "US30": "index", "US500": "index",
        "BTC": "crypto", "BTC (Binance)": "crypto", "ETH (Binance)": "crypto"}
# C14 round-number step. GOLD/BTC/NAS100/US30 are pre-registered; the
# confirmation-market steps are fixed here, before any result, at a similar
# share of today's price.
ROUND_STEP = {"GOLD": 10.0, "GOLD (broker)": 10.0, "SILVER": 0.25, "BTC": 1000.0,
              "BTC (Binance)": 1000.0, "ETH (Binance)": 50.0, "NAS100": 100.0,
              "US30": 250.0, "US500": 25.0}
# C08 partner, same vendor (Dukascopy with Dukascopy, Binance with Binance)
PARTNER = {"NAS100": "US500", "US30": "US500", "GOLD": "SILVER", "BTC (Binance)": "ETH (Binance)",
           "US500": "NAS100", "SILVER": "GOLD", "ETH (Binance)": "BTC (Binance)"}


# =============================================================== shared helpers
def emit(f, pos, buy, stop, tp1, tp2, hold, rule, extras=None):
    """Per-signal arrays -> sig_frame. Two signals on one bar: same direction
    keeps the first, opposite directions drop both. Rows whose stop/targets are
    on the wrong side of entry, or whose hold is not a positive number, drop."""
    idx, c = f["idx"], f["close"]
    n = len(c)
    pos = np.asarray(pos, dtype=np.int64).ravel()
    k = len(pos)

    def arr(x, dtype=float):
        a = np.asarray(x, dtype=dtype)
        return np.full(k, a.item(), dtype=dtype) if a.ndim == 0 else a.ravel()

    buy, stop, tp1, tp2, hold = arr(buy, bool), arr(stop), arr(tp1), arr(tp2), arr(hold)
    extras = {key: (np.asarray(v).ravel() if np.ndim(v) else np.full(k, v, dtype=object))
              for key, v in (extras or {}).items()}
    if k:
        order = np.argsort(pos, kind="stable")
        pos, buy, stop, tp1, tp2, hold = pos[order], buy[order], stop[order], tp1[order], tp2[order], hold[order]
        extras = {key: v[order] for key, v in extras.items()}
        d = pd.DataFrame({"pos": pos, "buy": buy})
        conflict = (d.groupby("pos")["buy"].transform("nunique") > 1).to_numpy()
        keep = ~conflict & ~d.duplicated("pos", keep="first").to_numpy()
        e = c[pos]
        sgn = np.where(buy, 1.0, -1.0)
        with np.errstate(invalid="ignore"):
            keep &= (sgn * (e - stop) > 0) & (sgn * (tp1 - e) > 0) & (sgn * (tp2 - tp1) >= 0)
            keep &= np.isfinite(tp1) & np.isfinite(tp2) & np.isfinite(hold) & (hold > 0)
        pos, buy, stop, tp1, tp2, hold = pos[keep], buy[keep], stop[keep], tp1[keep], tp2[keep], hold[keep]
        extras = {key: v[keep] for key, v in extras.items()}
    B = np.zeros(n, bool)
    S, T1, T2, HH = (np.full(n, np.nan) for _ in range(4))
    B[pos], S[pos], T1[pos], T2[pos], HH[pos] = buy, stop, tp1, tp2, hold
    out = sig_frame(idx, pos, B, c, S, T1, T2, HH, rule)
    for key, v in extras.items():
        full = np.empty(n, dtype=object)
        full[pos] = v
        out[key] = full[out["pos"].to_numpy()]
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def ny(f):
    """(NY minutes of day, NY local date as naive midnight, NY weekday)."""
    if "_ny" not in f:
        mins, local = local_minutes(f["idx"], NY)
        dates = pd.DatetimeIndex(local.normalize().tz_localize(None))
        f["_ny"] = (mins, dates, np.asarray(local.dayofweek))
    return f["_ny"]


def at_ny(f, hh, mm):
    """Series: NY weekday date -> position of the bar whose NY open is hh:mm."""
    mins, dates, dow = ny(f)
    sel = np.flatnonzero((mins == hh * 60 + mm) & (dow < 5))
    return pd.Series(sel, index=dates[sel]).groupby(level=0).first()


def ny_to_utc(dates, hh, mm):
    """NY local date + hh:mm -> naive UTC DatetimeIndex (DST-aware)."""
    loc = (pd.DatetimeIndex(dates) + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(
        NY, ambiguous="NaT", nonexistent="NaT")
    return loc.tz_convert("UTC").tz_localize(None)


def hours_to(f, pos, target_utc):
    """Hold in hours from the signal bar's CLOSE to target_utc (the simulator
    marks a timeout at the close of the first bar opening at/after it)."""
    t0 = f["idx"][np.asarray(pos, np.int64)] + FIVE
    return np.asarray((pd.DatetimeIndex(target_utc) - t0).total_seconds(), float) / 3600.0


def prev_cash_close(f, dates, max_days=4):
    """For each NY date: position of the 15:55 NY bar on the most recent EARLIER
    NY weekday that has it, if within max_days calendar days; else -1."""
    cc = at_ny(f, 15, 55)
    cd = cc.index
    k = cd.searchsorted(pd.DatetimeIndex(dates), side="left") - 1
    out = np.full(len(dates), -1, np.int64)
    ok = k >= 0
    kk = np.where(ok, k, 0)
    gap = (pd.DatetimeIndex(dates) - cd[kk]).days
    ok &= np.asarray(gap <= max_days)
    out[ok] = cc.to_numpy()[kk[ok]]
    return out


def get_atrd(m5, f):
    if "_atrd" not in f:
        f["_atrd"] = daily_atr(m5, f["idx"])
    return f["_atrd"]


def hourly_g4(g):
    """Valid hourly rows of an f-like dict (idx/open/high/low/close/atr1h):
    open/high/low/close, pos55 (position of the minute-55 bar), A = atr1h there."""
    idx = g["idx"]
    n = len(idx)
    hk = idx.floor("1h")
    df = pd.DataFrame({"open": g["open"], "high": g["high"], "low": g["low"], "close": g["close"],
                       "pos": np.arange(n)}, index=idx)
    gr = df.groupby(hk)
    h = pd.DataFrame({"open": gr["open"].first(), "high": gr["high"].max(), "low": gr["low"].min(),
                      "close": gr["close"].last(), "n": gr["pos"].size(), "last": gr["pos"].last()})
    last = h["last"].to_numpy(np.int64)
    ok = (h["n"].to_numpy() >= 10) & (np.asarray(idx[last].minute) == 55)
    h = h[ok].copy()
    h["pos55"] = h["last"].astype(np.int64)
    h["A"] = np.asarray(g["atr1h"], float)[h["pos55"].to_numpy()]
    return h.drop(columns=["last"])


def hourly(f):
    if "_h1" not in f:
        f["_h1"] = hourly_g4(f)
    return f["_h1"]


def prior_hour(hidx, vals, idx):
    """Per 5m bar: value of the last valid hourly row that STARTED before the
    bar's own hour (i.e. completed before the current hour began)."""
    k = hidx.searchsorted(idx.floor("1h"), side="left") - 1
    out = np.full(len(idx), np.nan)
    ok = k >= 0
    out[ok] = np.asarray(vals, float)[k[ok]]
    return out


def own_hour_row(hidx, idx):
    """Per 5m bar: row of its own hour in the valid-hour table, -1 if none."""
    return hidx.get_indexer(idx.floor("1h"))


def h20(f, n=20):
    """(H20, L20) per 5m bar: max high / min low of the n valid hours completed
    before the current hour (C11, C14 U4)."""
    key = f"_h20_{n}"
    if key not in f:
        h = hourly(f)
        hi = h["high"].rolling(n).max().to_numpy()
        lo = h["low"].rolling(n).min().to_numpy()
        f[key] = (prior_hour(h.index, hi, f["idx"]), prior_hour(h.index, lo, f["idx"]))
    return f[key]


def swings(f):
    """Hourly 3-3 fractals over valid rows: (is_swing_high, is_swing_low) per row.
    A swing at row k is KNOWN only once row k+3 has completed."""
    if "_sw" not in f:
        h = hourly(f)
        hh, hl = h["high"].to_numpy(), h["low"].to_numpy()

        def shifted(x, o):
            out = np.full(len(x), np.nan)
            if o > 0:
                out[o:] = x[:-o]
            else:
                out[:o] = x[-o:]
            return out
        lmax = np.maximum.reduce([shifted(hh, o) for o in (1, 2, 3)])
        rmax = np.maximum.reduce([shifted(hh, o) for o in (-1, -2, -3)])
        lmin = np.minimum.reduce([shifted(hl, o) for o in (1, 2, 3)])
        rmin = np.minimum.reduce([shifted(hl, o) for o in (-1, -2, -3)])
        with np.errstate(invalid="ignore"):
            f["_sw"] = ((hh > lmax) & (hh > rmax), (hl < lmin) & (hl < rmin))
    return f["_sw"]


def rollover_block(f, market):
    """True on bars in the first 30 minutes after the 17:00 NY rollover (gold/indices)."""
    if KIND.get(market) == "crypto":
        return np.zeros(len(f["idx"]), bool)
    mins, _, _ = ny(f)
    return (mins >= 17 * 60) & (mins < 17 * 60 + 30)


def fwd_first(cond):
    """Index of the first True along axis 1, -1 if none."""
    has = cond.any(axis=1)
    return np.where(has, cond.argmax(axis=1), -1)


# =============================================================== C01 IDXDIP4
def c01(stop_mode, T, control=False):
    """Cash-close 4-day-low flush bought inside a daily uptrend (control: every
    daily_bull session at 15:50 NY without the L3 condition)."""
    def build(m5, f):
        c, A, idx = f["close"], f["atr1h"], f["idx"]
        atrd = get_atrd(m5, f)
        s = at_ny(f, 15, 50)
        pos, dates = s.to_numpy(np.int64), s.index
        C = pd.Series(c[pos], index=dates)
        L3 = C.shift(1).rolling(3).min().to_numpy()
        ok = f["daily_bull"][pos] & np.isfinite(A[pos]) & np.isfinite(atrd[pos])
        if not control:
            with np.errstate(invalid="ignore"):
                ok &= c[pos] < L3
        pos, dates = pos[ok], dates[ok]
        a = A[pos]
        S = 3.0 * a if stop_mode == "3.0atr1h" else 1.5 * atrd[pos]
        entry = c[pos]
        exit_d = pd.DatetimeIndex(np.busday_offset(dates.values.astype("datetime64[D]"), 2, roll="forward"))
        hold = hours_to(f, pos, ny_to_utc(exit_d, 15, 55))
        # diagnostic: vol_ratio = ATRd / 100-day mean of daily TR (completed days)
        from backtest import resample
        d1 = resample(m5, "1D")
        prev = d1["close"].shift(1)
        tr = pd.concat([d1["high"] - d1["low"], (d1["high"] - prev).abs(), (d1["low"] - prev).abs()],
                       axis=1).max(axis=1)
        tr100 = map_to_5m(tr.rolling(100).mean(), idx, "1D")[pos]
        with np.errstate(invalid="ignore", divide="ignore"):
            vr = atrd[pos] / tr100
        tag = np.where(vr >= 1.2, "vol_hi", np.where(vr < 0.8, "vol_lo", "vol_mid"))
        return emit(f, pos, True, entry - S, entry + T * a, entry + T * a, hold, "no_breakeven",
                    {"baseline": S / (S + T * a), "tag": tag})
    return build


# =============================================================== C02 FUNDFADE-BTC
def c02(k, hold_h, hours=(0, 8, 16)):
    """Funding-snapshot unwind; hours=(4, 12, 20) is the off-funding placebo."""
    def build(m5, f):
        idx, o, h, l, c, A = f["idx"], f["open"], f["high"], f["low"], f["close"], f["atr1h"]
        days = pd.date_range(idx[0].normalize(), idx[-1].normalize(), freq="D")
        T = pd.DatetimeIndex(np.sort(np.concatenate([days + pd.Timedelta(hours=x) for x in hours])))
        ps = idx.get_indexer(T)
        p1 = idx.get_indexer(T - FIVE)
        p0 = idx.get_indexer(T - pd.Timedelta(hours=4) - FIVE)
        ok = (ps >= 0) & (p1 >= 0) & (p0 >= 0)
        T, ps, p1, p0 = T[ok], ps[ok], p1[ok], p0[ok]
        M = c[p1] - c[p0]
        a = A[ps]
        with np.errstate(invalid="ignore"):
            sell = (M >= k * a) & (c[ps] < o[ps])
            buy = (M <= -k * a) & (c[ps] > o[ps])
        keep = np.flatnonzero((sell | buy) & np.isfinite(a))
        lo = idx.searchsorted(T[keep] - pd.Timedelta(hours=4), side="left")
        out = {"pos": [], "buy": [], "stop": [], "tp1": [], "tp2": []}
        for n_, i in enumerate(keep):
            p = ps[i]
            XH, XL = h[lo[n_]:p + 1].max(), l[lo[n_]:p + 1].min()
            entry, ai, b = c[p], a[i], bool(buy[i])
            stop = XL - 0.5 * ai if b else XH + 0.5 * ai
            risk = abs(entry - stop)
            if risk < 0.5 * ai or risk > 4.0 * ai:
                continue
            tp2 = c[p0[i]] + 0.5 * M[i]
            tp1 = (entry + tp2) / 2.0
            if (b and entry >= tp1) or (not b and entry <= tp1):
                continue
            for key, v in zip(("pos", "buy", "stop", "tp1", "tp2"), (p, b, stop, tp1, tp2)):
                out[key].append(v)
        return emit(f, out["pos"], out["buy"], out["stop"], out["tp1"], out["tp2"], float(hold_h),
                    "partial_tp1", {"tag": np.where(np.asarray(out["buy"], bool), "buy", "sell")})
    return build


def c02_denicola(m5, f):
    """C02 benchmark: 2h bars on even UTC hours; last completed 2h log return
    beyond +/-3 sigma (trailing 30 days, prior bars only) -> fade for 2h."""
    idx, c = f["idx"], f["close"]
    key = idx.floor("2h")
    df = pd.DataFrame({"c": c, "pos": np.arange(len(c))}, index=idx)
    g = df.groupby(key)
    b = pd.DataFrame({"c": g["c"].last(), "n": g["pos"].size(), "last": g["pos"].last()})
    last = b["last"].to_numpy(np.int64)
    ok = (b["n"].to_numpy() >= 20) & (idx[last] == b.index + pd.Timedelta(minutes=115))
    b = b[ok]
    r = np.log(b["c"]).diff()
    r[~(b.index.to_series().diff() == pd.Timedelta(hours=2)).to_numpy()] = np.nan
    sigma = r.rolling("30D", min_periods=100).std().shift(1)
    rv, sv = r.to_numpy(), sigma.to_numpy()
    with np.errstate(invalid="ignore"):
        sell, buy = rv >= 3 * sv, rv <= -3 * sv
    sel = np.flatnonzero(sell | buy)
    pos = b["last"].to_numpy(np.int64)[sel]
    entry = c[pos]
    risk = 3.0 * sv[sel] * entry
    bb = buy[sel]
    sgn = np.where(bb, 1.0, -1.0)
    tp = entry + sgn * NO_TARGET * risk
    return emit(f, pos, bb, entry - sgn * risk, tp, tp, 2.0, "no_breakeven")


# =============================================================== C03 GAPHALF
def c03(fill, m):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        s = at_ny(f, 9, 30)
        pos, dates = s.to_numpy(np.int64), s.index
        pc = prev_cash_close(f, dates)
        ok = pc >= 0
        pos, dates, pc = pos[ok], dates[ok], pc[ok]
        G = c[pos] - c[pc]
        a = A[pos]
        with np.errstate(invalid="ignore"):
            ok = (np.abs(G) >= 0.75 * a) & (np.abs(G) <= 3.0 * a) & (G != 0)
        pos, dates, G = pos[ok], dates[ok], G[ok]
        sg = np.sign(G)
        entry = c[pos]
        stop = entry + sg * m * np.abs(G)
        tp = entry - sg * fill * np.abs(G)
        hold = hours_to(f, pos, ny_to_utc(dates, 15, 55))
        return emit(f, pos, G < 0, stop, tp, tp, hold, "no_breakeven",
                    {"baseline": np.full(len(pos), m / (m + fill)),
                     "tag": np.where(G > 0, "gap_up", "gap_down")})
    return build


# =============================================================== C04 GAPFILL
def c04(gmin, b):
    def build(m5, f):
        o, h, l, c, A = f["open"], f["high"], f["low"], f["close"], f["atr1h"]
        atrd = get_atrd(m5, f)
        s0, s1 = at_ny(f, 9, 30), at_ny(f, 9, 35)
        both = s0.index.intersection(s1.index)
        p0, p1 = s0[both].to_numpy(np.int64), s1[both].to_numpy(np.int64)
        pc = prev_cash_close(f, both)
        ok = pc >= 0
        p0, p1, pc, dates = p0[ok], p1[ok], pc[ok], both[ok]
        PC = c[pc]
        O = o[p0]
        G = O - PC
        a = A[p1]
        hi2, lo2 = np.maximum(h[p0], h[p1]), np.minimum(l[p0], l[p1])
        e = c[p1]
        with np.errstate(invalid="ignore"):
            ok = (np.abs(G) >= gmin * a) & (np.abs(G) <= 2.0 * a)
            up, dn = G > 0, G < 0
            ok &= (up & (lo2 > PC)) | (dn & (hi2 < PC))
            ok &= (up & (e < O)) | (dn & (e > O))
        stop = np.where(up, np.maximum(hi2 + b * a, e + 0.75 * a), np.minimum(lo2 - b * a, e - 0.75 * a))
        risk = np.abs(e - stop)
        with np.errstate(invalid="ignore"):
            ok &= (risk <= 3.0 * a) & (np.abs(PC - e) >= 0.25 * risk)
        sel = np.flatnonzero(ok)
        p1s, dates = p1[sel], dates[sel]
        tp2 = PC[sel]
        tp1 = e[sel] + 0.5 * (PC[sel] - e[sel])
        hold = hours_to(f, p1s, ny_to_utc(dates, 15, 55))
        with np.errstate(invalid="ignore", divide="ignore"):
            gd = np.abs(G[sel]) / atrd[p1s]
        size = np.where(gd < 0.3, "gapD<0.3", np.where(gd <= 0.7, "gapD0.3-0.7", "gapD>0.7"))
        side = np.where(G[sel] > 0, "up", "down")
        return emit(f, p1s, dn[sel], stop[sel], tp1, tp2, hold, "partial_tp1",
                    {"tag": np.char.add(np.char.add(side.astype(str), "|"), size.astype(str))})
    return build


# =============================================================== C05 ENGF
def _c05_setups(f, LB, CW):
    h = hourly(f)
    ho, hh, hl, hc, A_E = (h[x].to_numpy() for x in ("open", "high", "low", "close", "A"))
    p55 = h["pos55"].to_numpy(np.int64)
    n = len(h)
    rng = hh - hl
    prev_lo = pd.Series(hl).shift(1).rolling(LB - 1).min().to_numpy()
    prev_hi = pd.Series(hh).shift(1).rolling(LB - 1).max().to_numpy()
    with np.errstate(invalid="ignore"):
        bull = ((hl <= prev_lo) & (hc > ho) & (hc >= hl + 0.67 * rng) & (rng >= A_E)
                & f["daily_bear"][p55])
        bear = ((hh >= prev_hi) & (hc < ho) & (hc <= hh - 0.67 * rng) & (rng >= A_E)
                & f["daily_bull"][p55])
    out = []
    for setups, short in ((np.flatnonzero(bull), True), (np.flatnonzero(bear), False)):
        X = np.full(len(setups), -1, np.int64)
        for kk in range(CW, 0, -1):                     # smallest kk written last = first X
            x = setups + kk
            okx = x < n
            xc = np.where(okx, x, 0)
            with np.errstate(invalid="ignore"):
                cond = okx & ((hc[xc] < hl[setups]) if short else (hc[xc] > hh[setups]))
            X = np.where(cond, x, X)
        m = X >= 0
        out.append((setups[m], X[m], short))
    return h, out


def c05(LB, CW):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        h, parts = _c05_setups(f, LB, CW)
        hh, hl, p55 = h["high"].to_numpy(), h["low"].to_numpy(), h["pos55"].to_numpy(np.int64)
        P, BUY, STOP = [], [], []
        for E, X, short in parts:
            pos = p55[X]
            a, e = A[pos], c[pos]
            stop = np.maximum(hh[E] + 0.2 * a, e + 1.2 * a) if short else np.minimum(hl[E] - 0.2 * a, e - 1.2 * a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 3.5 * a
            P.append(pos[ok]); BUY.append(np.full(ok.sum(), not short)); STOP.append(stop[ok])
        return _rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), 1.0, 2.5, 36.0,
                        "breakeven")
    return build


def c05_control(LB):
    """With-trend hourly continuation: close beyond the previous LB hours' extreme."""
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        h = hourly(f)
        hh, hl, hc = h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy()
        p55 = h["pos55"].to_numpy(np.int64)
        plo = pd.Series(hl).shift(1).rolling(LB).min().to_numpy()
        phi = pd.Series(hh).shift(1).rolling(LB).max().to_numpy()
        with np.errstate(invalid="ignore"):
            sh = np.flatnonzero(f["daily_bear"][p55] & (hc < plo))
            lg = np.flatnonzero(f["daily_bull"][p55] & (hc > phi))
        P, BUY, STOP = [], [], []
        for rows, short in ((sh, True), (lg, False)):
            pos = p55[rows]
            a, e = A[pos], c[pos]
            stop = np.maximum(hh[rows] + 0.2 * a, e + 1.2 * a) if short else np.minimum(hl[rows] - 0.2 * a, e - 1.2 * a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 3.5 * a
            P.append(pos[ok]); BUY.append(np.full(ok.sum(), not short)); STOP.append(stop[ok])
        return _rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), 1.0, 2.5, 36.0,
                        "breakeven")
    return build


def _rr_emit(f, pos, buy, stop, r1, r2, hold, rule, extras=None):
    e = f["close"][pos]
    risk = np.abs(e - stop)
    sg = np.where(buy, 1.0, -1.0)
    return emit(f, pos, buy, stop, e + sg * r1 * risk, e + sg * r2 * risk, hold, rule, extras)


# =============================================================== C06 ASIA
def asia_days(f, market):
    """Valid Asia boxes: (dates, ws, we, AH, AL, RA). Window = bars 07:00-11:55 UTC."""
    if "_asia" not in f:
        idx, h, l, A = f["idx"], f["high"], f["low"], f["atr1h"]
        d = idx.normalize().unique()
        if KIND.get(market) != "crypto":
            d = d[d.dayofweek < 5]
        a0 = idx.searchsorted(d)
        ws = idx.searchsorted(d + pd.Timedelta(hours=7))
        we = idx.searchsorted(d + pd.Timedelta(hours=12))
        p655 = idx.get_indexer(d + pd.Timedelta(hours=6, minutes=55))
        ok = ((ws - a0) >= 60) & (p655 >= 0)
        rows = []
        for i in np.flatnonzero(ok):
            AH, AL = h[a0[i]:ws[i]].max(), l[a0[i]:ws[i]].min()
            A0 = A[p655[i]]
            RA = AH - AL
            if np.isfinite(A0) and 0.5 * A0 <= RA <= 3.0 * A0 and we[i] > ws[i]:
                rows.append((d[i], ws[i], we[i], AH, AL, RA))
        f["_asia"] = rows
    return f["_asia"]


def _c06_trade(f, D, j, ext, short, AH, AL, RA):
    A, c = f["atr1h"], f["close"]
    a, e = A[j], c[j]
    if short:
        stop = max(ext + 0.25 * a, e + 1.0 * a)
        risk = stop - e
        tp1 = AL if (e - AL) >= 0.6 * risk else e - risk
        tp2 = min(AL - 0.5 * RA, tp1 - 0.5 * risk)
    else:
        stop = min(ext - 0.25 * a, e - 1.0 * a)
        risk = e - stop
        tp1 = AH if (AH - e) >= 0.6 * risk else e + risk
        tp2 = max(AH + 0.5 * RA, tp1 + 0.5 * risk)
    if not np.isfinite(a) or risk > 3.0 * a:
        return None
    return j, not short, stop, tp1, tp2, D


def c06(K, filt, placebo=False, market="GOLD"):
    def build(m5, f):
        c, h, l = f["close"], f["high"], f["low"]
        rows = []
        for D, ws, we, AH, AL, RA in asia_days(f, market):
            cw, hw, lw = c[ws:we], h[ws:we], l[ws:we]
            cands = []
            if not placebo:
                br = np.flatnonzero(cw > AH)
                if len(br):
                    k = br[0]
                    ext = np.maximum.accumulate(hw[k:])
                    cond = cw[k:] < AH - K * RA
                    cond[0] = False
                    if filt:
                        cond &= (ext - AH) <= 0.10 * RA
                    if cond.any():
                        jj = int(cond.argmax())
                        cands.append((k + jj, ext[jj], True))
                br = np.flatnonzero(cw < AL)
                if len(br):
                    k = br[0]
                    ext = np.minimum.accumulate(lw[k:])
                    cond = cw[k:] > AL + K * RA
                    cond[0] = False
                    if filt:
                        cond &= (AL - ext) <= 0.10 * RA
                    if cond.any():
                        jj = int(cond.argmax())
                        cands.append((k + jj, ext[jj], False))
            else:
                prevc = c[ws - 1:we - 1]
                lvl = AH - K * RA
                prior_above = np.concatenate([[False], np.logical_or.accumulate(cw > AH)[:-1]])
                cross = (prevc >= lvl) & (cw < lvl) & ~prior_above
                if cross.any():
                    jj = int(cross.argmax())
                    cands.append((jj, hw[:jj + 1].max(), True))
                lvl2 = AL + K * RA
                prior_below = np.concatenate([[False], np.logical_or.accumulate(cw < AL)[:-1]])
                cross = (prevc <= lvl2) & (cw > lvl2) & ~prior_below
                if cross.any():
                    jj = int(cross.argmax())
                    cands.append((jj, lw[:jj + 1].min(), False))
            if not cands:
                continue
            cands.sort(key=lambda t: t[0])
            if len(cands) > 1 and cands[0][0] == cands[1][0]:
                continue
            jj, ext, short = cands[0]
            t = _c06_trade(f, D, ws + jj, ext, short, AH, AL, RA)
            if t:
                rows.append(t)
        if not rows:
            return emit(f, [], [], [], [], [], [], "partial_tp1")
        pos, buy, stop, tp1, tp2, D = map(np.asarray, zip(*rows))
        hold = hours_to(f, pos, pd.DatetimeIndex(D) + pd.Timedelta(hours=20))
        return emit(f, pos.astype(np.int64), buy.astype(bool), stop, tp1, tp2, hold, "partial_tp1",
                    {"tag": np.where(buy.astype(bool), "long", "short")})
    return build


def c06_levels(market):
    def levels(m5, f):
        n = len(f["idx"])
        up, dn = np.full(n, np.nan), np.full(n, np.nan)
        for D, ws, we, AH, AL, RA in asia_days(f, market):
            up[ws:we], dn[ws:we] = AH, AL
        return up, dn
    return levels


# =============================================================== C07 COMEXTRAP-XAU
def c07(k, b):
    def build(m5, f):
        o, h, l, c, A, idx = f["open"], f["high"], f["low"], f["close"], f["atr1h"], f["idx"]
        s03, s815, s820 = at_ny(f, 3, 0), at_ny(f, 8, 15), at_ny(f, 8, 20)
        days = s03.index.intersection(s815.index).intersection(s820.index)
        wend = idx.searchsorted(ny_to_utc(days, 10, 0), side="left")      # bars opening < 10:00 NY
        rows = []
        for i, d in enumerate(days):
            p03, p815, p820 = s03[d], s815[d], s820[d]
            if not (p03 < p815 < p820 < wend[i]):
                continue
            O3 = o[p03]
            L = c[p815] - O3
            LH, LL = h[p03:p815 + 1].max(), l[p03:p815 + 1].min()
            N0 = o[p820]
            cw, aw = c[p820:wend[i]], A[p820:wend[i]]
            WH = np.maximum.accumulate(h[p820:wend[i]])
            WL = np.minimum.accumulate(l[p820:wend[i]])
            with np.errstate(invalid="ignore"):
                sell = (L >= k * aw) & (WH > LH) & (cw < N0)
                buy = (L <= -k * aw) & (WL < LL) & (cw > N0)
            either = sell | buy
            if not either.any():
                continue
            jj = int(either.argmax())
            j, a, e = p820 + jj, aw[jj], cw[jj]
            is_buy = bool(buy[jj])
            stop = WL[jj] - b * a if is_buy else WH[jj] + b * a
            risk = abs(e - stop)
            if not (0.5 * a <= risk <= 4.0 * a):
                continue
            tp1, tp2 = O3 + 0.5 * L, O3
            if (is_buy and e >= tp1) or (not is_buy and e <= tp1):
                continue
            rows.append((j, is_buy, stop, tp1, tp2, d))
        if not rows:
            return emit(f, [], [], [], [], [], [], "partial_tp1")
        pos, buy, stop, tp1, tp2, D = map(np.asarray, zip(*rows))
        pos = pos.astype(np.int64)
        hold = hours_to(f, pos, ny_to_utc(pd.DatetimeIndex(D), 13, 30))
        return emit(f, pos, buy.astype(bool), stop, tp1, tp2, hold, "partial_tp1")
    return build


# =============================================================== C08 SMT1
_PARTNER_CACHE = {}


def load_partner(label):
    if label not in _PARTNER_CACHE:
        from backtest import load_real_data
        _PARTNER_CACHE.clear()                    # keep at most one partner in memory
        _PARTNER_CACHE[label] = load_real_data(os.path.join(PROJ, DATA_FILES[label]))
    return _PARTNER_CACHE[label]


def smt_hours(m5, f, partner, L):
    """Joined valid hours of P (traded) and Q (partner): exact timestamp inner
    join, NO forward fill of Q (pre-registered rule 1)."""
    key = f"_smt_{partner}_{L}"
    if key not in f:
        q = load_partner(partner)
        joined = m5.index.intersection(q.index)
        P = m5.loc[joined]
        Q = q.loc[joined]
        opos = m5.index.get_indexer(joined)
        gp = {"idx": joined, "open": P["open"].to_numpy(float), "high": P["high"].to_numpy(float),
              "low": P["low"].to_numpy(float), "close": P["close"].to_numpy(float),
              "atr1h": f["atr1h"][opos]}
        gq = {"idx": joined, "open": Q["open"].to_numpy(float), "high": Q["high"].to_numpy(float),
              "low": Q["low"].to_numpy(float), "close": Q["close"].to_numpy(float),
              "atr1h": np.zeros(len(joined))}
        hp, hq = hourly_g4(gp), hourly_g4(gq)
        common = hp.index.intersection(hq.index)
        hp, hq = hp.loc[common], hq.loc[common]
        d = pd.DataFrame(index=common)
        for tag, hx in (("P", hp), ("Q", hq)):
            d[f"hi{tag}"], d[f"lo{tag}"], d[f"cl{tag}"] = hx["high"], hx["low"], hx["close"]
            d[f"HI{tag}"] = hx["high"].shift(1).rolling(L).max()
            d[f"LO{tag}"] = hx["low"].shift(1).rolling(L).min()
        d["pos"] = opos[hp["pos55"].to_numpy(np.int64)]
        f[key] = d
    return f[key]


def c08(leg, L, market, control=False):
    """leg 'A': fade our market's lone failed break; 'B': trade our market when
    the partner's lone break failed. control=True: leg A without the partner."""
    partner = PARTNER[market]

    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        d = smt_hours(m5, f, partner, L)
        g = {k: d[k].to_numpy(float) for k in d.columns if k != "pos"}
        with np.errstate(invalid="ignore"):
            if leg == "A":
                bear = (g["hiP"] > g["HIP"]) & (g["clP"] < g["HIP"])
                bull = (g["loP"] < g["LOP"]) & (g["clP"] > g["LOP"])
                if not control:
                    bear &= g["hiQ"] <= g["HIQ"]
                    bull &= g["loQ"] >= g["LOQ"]
            else:
                bear = (g["hiQ"] > g["HIQ"]) & (g["clQ"] < g["HIQ"]) & (g["hiP"] <= g["HIP"])
                bull = (g["loQ"] < g["LOQ"]) & (g["clQ"] > g["LOQ"]) & (g["loP"] >= g["LOP"])
        one = bear ^ bull
        rows = np.flatnonzero(one)
        pos = d["pos"].to_numpy(np.int64)[rows]
        buy = bull[rows]
        a, e = A[pos], c[pos]
        stop = np.where(buy, np.minimum(g["loP"][rows] - 0.3 * a, e - 1.0 * a),
                        np.maximum(g["hiP"][rows] + 0.3 * a, e + 1.0 * a))
        with np.errstate(invalid="ignore"):
            ok = np.abs(e - stop) <= 3.0 * a
        return _rr_emit(f, pos[ok], buy[ok], stop[ok], 1.0, 2.0, 24.0, "partial_tp1",
                        {"tag": np.where(buy[ok], "bull", "bear")})
    return build


def c08_levels(market, L):
    partner = PARTNER[market]

    def levels(m5, f):
        d = smt_hours(m5, f, partner, L)
        up = d["hiP"].rolling(L).max().to_numpy()
        dn = d["loP"].rolling(L).min().to_numpy()
        return prior_hour(d.index, up, f["idx"]), prior_hour(d.index, dn, f["idx"])
    return levels


# =============================================================== C09 OPENDRIVE
def c09(dmin, r):
    def build(m5, f):
        o, h, l, c, A = f["open"], f["high"], f["low"], f["close"], f["atr1h"]
        s0, s5 = at_ny(f, 9, 30), at_ny(f, 9, 55)
        both = s0.index.intersection(s5.index)
        p0, p5 = s0[both].to_numpy(np.int64), s5[both].to_numpy(np.int64)
        ok = p5 > p0
        p0, p5 = p0[ok], p5[ok]
        O = o[p0]
        D = c[p5] - O
        DH = np.array([h[x:y + 1].max() for x, y in zip(p0, p5)])
        DL = np.array([l[x:y + 1].min() for x, y in zip(p0, p5)])
        a, e = A[p5], c[p5]
        with np.errstate(invalid="ignore"):
            up, dn = D > 0, D < 0
            ok = (np.abs(D) >= dmin * a) & ((up & (c[p5] < o[p5])) | (dn & (c[p5] > o[p5])))
        stop = np.where(up, np.maximum(DH + 0.5 * a, e + 0.75 * a), np.minimum(DL - 0.5 * a, e - 0.75 * a))
        risk = np.abs(e - stop)
        tp = O + (1.0 - r) * D
        with np.errstate(invalid="ignore"):
            beyond = np.where(up, e <= tp, e >= tp)
            ok &= (risk <= 3.0 * a) & ~beyond & (np.abs(tp - e) >= 0.3 * risk)
        sel = np.flatnonzero(ok)
        return emit(f, p5[sel], dn[sel], stop[sel], tp[sel], tp[sel], 2.5, "no_breakeven",
                    {"tag": np.where(up[sel], "drive_up", "drive_down")})
    return build


# =============================================================== C10 FBR
def _c10_boxes(m5, f, cc):
    """Sequential box state machine; returns (signals, fake-break records)."""
    key = f"_c10_{cc}"
    if key in f:
        return f[key]
    idx, h, l, c = f["idx"], f["high"], f["low"], f["close"]
    atrd = get_atrd(m5, f)
    hr = hourly(f)
    hs = hr.index
    p55 = hr["pos55"].to_numpy(np.int64)
    R12 = (hr["high"].rolling(12).max() - hr["low"].rolling(12).min()).to_numpy()
    BTs, BBs = hr["high"].rolling(12).max().to_numpy(), hr["low"].rolling(12).min().to_numpy()
    contiguous = np.zeros(len(hr), bool)
    if len(hr) > 11:
        contiguous[11:] = (hs[11:] - hs[:-11]) == pd.Timedelta(hours=11)
    with np.errstate(invalid="ignore"):
        arm = contiguous & (R12 <= cc * atrd[p55])
    rows_arm = np.flatnonzero(arm)
    t_arm = idx[p55[rows_arm]]
    sigs, fakes = [], []
    t_free = idx[0]
    while True:
        i = t_arm.searchsorted(t_free, side="left")
        if i >= len(rows_arm):
            break
        e = rows_arm[i]
        a = p55[e]
        ta = idx[a]
        BT, BB = BTs[e], BBs[e]
        lim = idx.searchsorted(ta + pd.Timedelta(hours=24), side="right")
        seg = c[a + 1:lim]
        brk = (seg > BT) | (seg < BB)
        if not brk.any():
            t_free = ta + pd.Timedelta(hours=36)            # expired at +24h, then 12h cooldown
            continue
        fk = a + 1 + int(brk.argmax())
        upfake = c[fk] > BT
        fakes.append((fk, upfake, BT, BB))
        lim2 = idx.searchsorted(idx[fk] + pd.Timedelta(hours=12), side="right")
        seg2 = c[fk + 1:lim2]
        cond = (seg2 < BB) if upfake else (seg2 > BT)
        if cond.any():
            si = fk + 1 + int(cond.argmax())
            ext = h[fk:si + 1].max() if upfake else l[fk:si + 1].min()
            sigs.append((si, not upfake, BT, BB, ext))
            end = idx[si]
        else:
            end = idx[fk] + pd.Timedelta(hours=12)
        t_free = end + pd.Timedelta(hours=12)
    f[key] = (sigs, fakes)
    return f[key]


def c10(cc, mode, control=False):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        sigs, fakes = _c10_boxes(m5, f, cc)
        P, BUY, STOP = [], [], []
        if not control:
            for si, is_buy, BT, BB, ext in sigs:
                a, e = A[si], c[si]
                if mode == "edge":
                    stop = BT - 1.0 * a if is_buy else BB + 1.0 * a
                else:
                    stop = min(ext - 0.25 * a, e - a) if is_buy else max(ext + 0.25 * a, e + a)
                P.append(si); BUY.append(is_buy); STOP.append(stop)
        else:
            # first-break continuation on the same boxes; stop geometry mirrored
            for fk, upfake, BT, BB in fakes:
                a, e = A[fk], c[fk]
                is_buy = bool(upfake)
                if mode == "edge":
                    stop = BT - 1.0 * a if is_buy else BB + 1.0 * a
                else:
                    stop = min(BB - 0.25 * a, e - a) if is_buy else max(BT + 0.25 * a, e + a)
                P.append(fk); BUY.append(is_buy); STOP.append(stop)
        P, BUY, STOP = np.asarray(P, np.int64), np.asarray(BUY, bool), np.asarray(STOP, float)
        if len(P):
            a = A[P]
            risk = np.abs(c[P] - STOP)
            with np.errstate(invalid="ignore"):
                ok = (risk <= 3.0 * a) & (risk >= 0.75 * a)
            P, BUY, STOP = P[ok], BUY[ok], STOP[ok]
        return _rr_emit(f, P, BUY, STOP, 1.0, 2.0, 24.0, "partial_tp1")
    return build


# =============================================================== C11 FB20
def _c11_side(f, W, ermax, up):
    c, h, l, A, er = f["close"], f["high"], f["low"], f["atr1h"], f["er_d"]
    n = len(c)
    H20, L20 = h20(f)
    lvl_arr = H20 if up else L20
    prev_c = np.concatenate([[np.nan], c[:-1]])
    prev_l = np.concatenate([[np.nan], lvl_arr[:-1]])
    with np.errstate(invalid="ignore"):
        ev = np.flatnonzero((c > lvl_arr) & (prev_c <= prev_l)) if up else \
             np.flatnonzero((c < lvl_arr) & (prev_c >= prev_l))
    if not len(ev):
        return np.zeros(0, np.int64), np.zeros(0)
    nxt = np.append(ev[1:], n)
    out_p, out_x = [], []
    for s in range(0, len(ev), 20000):                     # chunked: bounded memory
        e, nx = ev[s:s + 20000], nxt[s:s + 20000]
        offs = np.arange(W + 1)
        M = e[:, None] + offs[None, :]
        valid = (M < n) & ((M < nx[:, None]) | (offs[None, :] == 0))
        Mc = np.minimum(M, n - 1)
        hi, lo, cj, Aj, erj = h[Mc], l[Mc], c[Mc], A[Mc], er[Mc]
        lvl = lvl_arr[e][:, None]
        if up:
            ext = np.maximum.accumulate(hi, axis=1)
            tcl = np.empty_like(lo)
            run, cur = np.full(len(e), -np.inf), np.full(len(e), np.nan)
            for col in range(W + 1):
                better = hi[:, col] >= run                    # latest bar on a tie
                run = np.maximum(run, hi[:, col])
                cur = np.where(better, lo[:, col], cur)
                tcl[:, col] = cur
            with np.errstate(invalid="ignore"):
                cond = (valid & (ext - lvl >= 0.20 * Aj) & (cj < lvl - 0.25 * Aj) & (cj < tcl)
                        & (erj < ermax))
        else:
            ext = np.minimum.accumulate(lo, axis=1)
            tch = np.empty_like(hi)
            run, cur = np.full(len(e), np.inf), np.full(len(e), np.nan)
            for col in range(W + 1):
                better = lo[:, col] <= run
                run = np.minimum(run, lo[:, col])
                cur = np.where(better, hi[:, col], cur)
                tch[:, col] = cur
            with np.errstate(invalid="ignore"):
                cond = (valid & (lvl - ext >= 0.20 * Aj) & (cj > lvl + 0.25 * Aj) & (cj > tch)
                        & (erj < ermax))
        cond[:, 0] = False
        first = fwd_first(cond)
        m = first >= 0
        rows = np.flatnonzero(m)
        out_p.append(M[rows, first[m]])
        out_x.append(ext[rows, first[m]])
    return np.concatenate(out_p), np.concatenate(out_x)


def c11(W, ermax):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        P, BUY, STOP = [], [], []
        for up in (True, False):
            p, x = _c11_side(f, W, ermax, up)
            a, e = A[p], c[p]
            short = up
            stop = np.maximum(x + 0.30 * a, e + a) if short else np.minimum(x - 0.30 * a, e - a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 3.0 * a
            P.append(p[ok]); BUY.append(np.full(ok.sum(), not short)); STOP.append(stop[ok])
        return _rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), 1.0, 2.0, 12.0,
                        "breakeven")
    return build


def c11_levels(market):
    def levels(m5, f):
        return h20(f)
    return levels


# =============================================================== C12 IFVG
def m15_g4(f):
    if "_m15" not in f:
        idx = f["idx"]
        n = len(idx)
        qk = idx.floor("15min")
        df = pd.DataFrame({"open": f["open"], "high": f["high"], "low": f["low"], "close": f["close"],
                           "pos": np.arange(n)}, index=idx)
        g = df.groupby(qk)
        q = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                          "close": g["close"].last(), "n": g["pos"].size(), "last": g["pos"].last()})
        last = q["last"].to_numpy(np.int64)
        ok = (q["n"].to_numpy() == 3) & (np.asarray(idx[last].minute) % 15 == 10)
        q = q[ok].copy()
        q["A"] = f["atr1h"][q["last"].to_numpy(np.int64)]
        f["_m15"] = q
    return f["_m15"]


def _c12_zones(f, G, gap, bull):
    """Zones from 3 consecutive (calendar-contiguous) complete 15m bars.
    gap=True: real FVG of >= G*A; gap=False: control windows with NO gap."""
    q = m15_g4(f)
    o, h, l, c, A = (q[x].to_numpy() for x in ("open", "high", "low", "close", "A"))
    last = q["last"].to_numpy(np.int64)
    t = q.index
    k = np.arange(2, len(q))
    contig = ((t[k] - t[k - 1]) == pd.Timedelta(minutes=15)) & ((t[k - 1] - t[k - 2]) == pd.Timedelta(minutes=15))
    k = k[np.asarray(contig)]
    a = A[k]
    body = np.abs(c[k - 1] - o[k - 1])
    with np.errstate(invalid="ignore"):
        if bull:
            disp = (c[k - 1] > o[k - 1]) & (body >= 0.8 * a)
            g_ = l[k] - h[k - 2]
            has = (g_ >= G * a) & (g_ > 0) if gap else (l[k] <= h[k - 2])
            near = np.where(gap, l[k], np.maximum(l[k], h[k - 2]))           # top
            far = np.where(gap, h[k - 2], np.minimum(l[k], h[k - 2]))        # bot
            cancel = np.maximum(h[k - 1], h[k])                              # disp_hi
        else:
            disp = (c[k - 1] < o[k - 1]) & (body >= 0.8 * a)
            g_ = l[k - 2] - h[k]
            has = (g_ >= G * a) & (g_ > 0) if gap else (h[k] >= l[k - 2])
            near = np.where(gap, h[k], np.minimum(h[k], l[k - 2]))
            far = np.where(gap, l[k - 2], np.maximum(h[k], l[k - 2]))
            cancel = np.minimum(l[k - 1], l[k])                              # disp_lo
    sel = np.flatnonzero(disp & has & np.isfinite(a))
    return last[k[sel]], near[sel], far[sel], o[k - 1][sel], cancel[sel]


def _c12_scan(f, market, zones, bull, mode):
    """First qualifying failure per zone within 8h -> signal positions + near edge."""
    idx, c, h, l, A = f["idx"], f["close"], f["high"], f["low"], f["atr1h"]
    n = len(c)
    p, near, far, dopen, cancel = zones
    if not len(p):
        return np.zeros(0, np.int64), np.zeros(0)
    ls = pd.Series(l)
    hs_ = pd.Series(h)
    prev3_lo = ls.shift(1).rolling(3).min().to_numpy()
    prev3_hi = hs_.shift(1).rolling(3).max().to_numpy()
    regime_block = f["daily_bull"] if bull else f["daily_bear"]
    roll = rollover_block(f, market)
    Wb, CH = 96, 4000                                   # 8h of 5m bars; chunked for memory
    out_p, out_n = [], []
    for s in range(0, len(p), CH):
        pp = p[s:s + CH]
        nr, fr, dop, cn = near[s:s + CH], far[s:s + CH], dopen[s:s + CH], cancel[s:s + CH]
        offs = np.arange(1, Wb + 1)
        M = pp[:, None] + offs[None, :]
        inb = M < n
        Mc = np.minimum(M, n - 1)
        within = inb & (idx[Mc.ravel()].to_numpy().reshape(Mc.shape)
                        <= (idx[pp] + pd.Timedelta(hours=8)).to_numpy()[:, None])
        cj, hj, lj, Aj = c[Mc], h[Mc], l[Mc], A[Mc]
        with np.errstate(invalid="ignore"):
            if bull:
                cancel_hit = within & (cj > cn[:, None])
                touch = within & (lj <= nr[:, None])
                if mode == "F1":
                    fail = (cj < fr[:, None] - 0.20 * Aj) & (cj < prev3_lo[Mc])
                else:
                    fail = cj < np.minimum(dop, fr)[:, None]
            else:
                cancel_hit = within & (cj < cn[:, None])
                touch = within & (hj >= nr[:, None])
                if mode == "F1":
                    fail = (cj > fr[:, None] + 0.20 * Aj) & (cj > prev3_hi[Mc])
                else:
                    fail = cj > np.maximum(dop, fr)[:, None]
        tcol = fwd_first(touch)
        ccol = fwd_first(cancel_hit)
        col = np.arange(Wb)[None, :]
        ok = (within & fail & ~regime_block[Mc] & ~roll[Mc]
              & (tcol[:, None] >= 0) & (col >= tcol[:, None])
              & ((ccol[:, None] < 0) | (col < ccol[:, None])))
        first = fwd_first(ok)
        m = first >= 0
        rows = np.flatnonzero(m)
        out_p.append(M[rows, first[m]])
        out_n.append(nr[rows])
    return np.concatenate(out_p), np.concatenate(out_n)


def c12(G, mode, market, control=False):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        P, BUY, STOP = [], [], []
        for bull in (True, False):                   # bull FVG -> SHORT, bear FVG -> LONG
            z = _c12_zones(f, G, not control, bull)
            p, nr = _c12_scan(f, market, z, bull, mode)
            a, e = A[p], c[p]
            stop = np.maximum(nr + 0.30 * a, e + a) if bull else np.minimum(nr - 0.30 * a, e - a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 2.5 * a
            P.append(p[ok]); BUY.append(np.full(ok.sum(), not bull)); STOP.append(stop[ok])
        return _rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), 1.0, 2.0, 12.0,
                        "breakeven")
    return build


# =============================================================== C13 EXH3
def _c13_events(f, N):
    """Per swing level: the hour its Nth distinct sweep completes.
    Returns (records, level_rows_hi, level_rows_lo) where records are
    (row, S, pool_extreme, is_sell)."""
    key = f"_c13_{N}"
    if key in f:
        return f[key]
    h = hourly(f)
    hh, hl, hc, A = (h[x].to_numpy() for x in ("high", "low", "close", "A"))
    hs = h.index
    n = len(h)
    sw_hi, sw_lo = swings(f)
    recs = []
    lvl_hi, lvl_lo = np.full(n, np.nan), np.full(n, np.nan)
    ends = hs.searchsorted(hs + pd.Timedelta(hours=120), side="left")   # rows with start < k+120h
    for is_sell, sw in ((True, sw_hi), (False, sw_lo)):
        for k in np.flatnonzero(sw):
            j0, j1 = k + 4, ends[k]
            if j0 >= j1:
                continue
            S = hh[k] if is_sell else hl[k]
            sh, sl, sc, sa = hh[j0:j1], hl[j0:j1], hc[j0:j1], A[j0:j1]
            with np.errstate(invalid="ignore"):
                inval = (sc > S + 0.25 * sa) if is_sell else (sc < S - 0.25 * sa)
            stop_at = int(inval.argmax()) if inval.any() else len(sc)
            lv = lvl_hi if is_sell else lvl_lo
            lv[j0:j0 + min(stop_at + 1, len(sc))] = S       # G9 level: active incl. the invalidating hour
            if stop_at == 0:
                continue
            sh, sl, sc = sh[:stop_at], sl[:stop_at], sc[:stop_at]
            sweep = ((sh > S) & (sc < S)) if is_sell else ((sl < S) & (sc > S))
            new = sweep & ~np.concatenate([[False], sweep[:-1]])
            cnt = np.cumsum(new)
            hit = np.flatnonzero(new & (cnt == N))
            if not len(hit):
                continue
            r = hit[0]
            pool = sh[:r + 1][sweep[:r + 1]].max() if is_sell else sl[:r + 1][sweep[:r + 1]].min()
            recs.append((j0 + r, S, pool, is_sell))
    f[key] = (recs, lvl_hi, lvl_lo)
    return f[key]


def c13(N, target_R):
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        h = hourly(f)
        p55 = h["pos55"].to_numpy(np.int64)
        recs, _, _ = _c13_events(f, N)
        if not recs:
            return emit(f, [], [], [], [], [], [], "no_breakeven")
        d = pd.DataFrame(recs, columns=["row", "S", "pool", "sell"])
        sells = d[d["sell"]].sort_values("S", ascending=False).groupby("row").first()
        buys = d[~d["sell"]].sort_values("S", ascending=True).groupby("row").first()
        both = sells.index.intersection(buys.index)
        sells, buys = sells.drop(both), buys.drop(both)
        P, BUY, STOP = [], [], []
        for tab, is_sell in ((sells, True), (buys, False)):
            pos = p55[tab.index.to_numpy(np.int64)]
            a, e = A[pos], c[pos]
            pool = tab["pool"].to_numpy()
            stop = np.maximum(pool + 0.5 * a, e + a) if is_sell else np.minimum(pool - 0.5 * a, e - a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 3.0 * a
            P.append(pos[ok]); BUY.append(np.full(ok.sum(), not is_sell)); STOP.append(stop[ok])
        return _rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), target_R, target_R,
                        24.0, "no_breakeven")
    return build


def c13_levels(market):
    def levels(m5, f):
        h = hourly(f)
        _, lh, ll = _c13_events(f, 1)       # level activity does not depend on N
        row = own_hour_row(h.index, f["idx"])
        up, dn = np.full(len(row), np.nan), np.full(len(row), np.nan)
        ok = row >= 0
        up[ok], dn[ok] = lh[row[ok]], ll[row[ok]]
        return up, dn
    return levels


# =============================================================== C14 TDC
def _c14_levels(m5, f, market):
    """Up/down level types U1..U6 / D1..D6 per bar, each known at that bar's close."""
    if "_c14" in f:
        return f["_c14"]
    from backtest import resample
    idx, h, l, c = f["idx"], f["high"], f["low"], f["close"]
    n = len(c)
    d1 = resample(m5, "1D")
    U1, D1 = map_to_5m(d1["high"], idx, "1D"), map_to_5m(d1["low"], idx, "1D")
    wk = idx.normalize() - pd.to_timedelta(np.asarray(idx.dayofweek), unit="D")
    gw = pd.DataFrame({"h": h, "l": l}).groupby(np.asarray(wk))
    wh, wl = gw["h"].max(), gw["l"].min()
    U2 = wh.shift(1).reindex(wk).to_numpy(float)
    D2 = wl.shift(1).reindex(wk).to_numpy(float)
    day = idx.normalize()
    asia = np.asarray(idx.hour < 7)
    ah = pd.Series(h[asia]).groupby(np.asarray(day[asia])).max()
    al = pd.Series(l[asia]).groupby(np.asarray(day[asia])).min()
    after = np.asarray(idx.hour >= 7)
    U3, D3 = np.full(n, np.nan), np.full(n, np.nan)
    U3[after] = ah.reindex(day[after]).to_numpy(float)
    D3[after] = al.reindex(day[after]).to_numpy(float)
    U4, D4 = h20(f)
    hr = hourly(f)
    sw_hi, sw_lo = swings(f)
    hh, hl = hr["high"].to_numpy(), hr["low"].to_numpy()
    # swing at row k is known after row k+3 completes -> value at row k+3, carried forward
    shi = pd.Series(np.where(sw_hi, hh, np.nan)).shift(3).ffill().to_numpy()
    slo = pd.Series(np.where(sw_lo, hl, np.nan)).shift(3).ffill().to_numpy()
    U5, D5 = prior_hour(hr.index, shi, idx), prior_hour(hr.index, slo, idx)
    step = ROUND_STEP[market]
    U6 = step * (np.floor(c / step) + 1.0)
    D6 = step * (np.ceil(c / step) - 1.0)
    f["_c14"] = (np.vstack([U1, U2, U3, U4, U5, U6]), np.vstack([D1, D2, D3, D4, D5, D6]))
    return f["_c14"]


def _throttle(pos, times, gap=pd.Timedelta(hours=4)):
    keep, last = [], None
    for p, t in zip(pos, times):
        if last is None or t - last >= gap:
            keep.append(p)
            last = t
    return np.asarray(keep, np.int64)


def c14(K, hold_H, market, control=False):
    def build(m5, f):
        c, A, idx = f["close"], f["atr1h"], f["idx"]
        U, D = _c14_levels(m5, f, market)
        # cluster measured at bar i-1 (views, no copies), signal at bar i
        Up, Dp, cp, Ap, ci = U[:, :-1], D[:, :-1], c[:-1], A[:-1], c[1:]
        with np.errstate(invalid="ignore"):
            inU = (Up > cp) & (Up <= cp + 0.5 * Ap)
            inD = (Dp < cp) & (Dp >= cp - 0.5 * Ap)
            nU, nD = inU.sum(0), inD.sum(0)
            maxU = np.where(inU, Up, -np.inf).max(0)
            minD = np.where(inD, Dp, np.inf).min(0)
            if control:
                bull = inU[3] & (nU == 1) & (ci > maxU)
                bear = inD[3] & (nD == 1) & (ci < minD)
            else:
                bull = (nU >= K) & (ci > maxU)
                bear = (nD >= K) & (ci < minD)
        bull = np.concatenate([[False], bull])
        bear = np.concatenate([[False], bear])
        ok = np.isfinite(A) & ~rollover_block(f, market)
        pb = _throttle(np.flatnonzero(bull & ok), idx[bull & ok])
        ps = _throttle(np.flatnonzero(bear & ok), idx[bear & ok])
        pos = np.concatenate([pb, ps])
        buy = np.concatenate([np.ones(len(pb), bool), np.zeros(len(ps), bool)])
        e = c[pos]
        stop = np.where(buy, e - A[pos], e + A[pos])
        return _rr_emit(f, pos, buy, stop, 3.0, 3.0, float(hold_H), "no_breakeven")
    return build


# =============================================================== registry
PRIMARY = {
    "C01-IDXDIP4": ["NAS100", "US30", "GOLD"],
    "C02-FUNDFADE-BTC": ["BTC"],
    "C03-GAPHALF": ["NAS100", "US30"],
    "C04-GAPFILL": ["NAS100", "US30"],
    "C05-ENGF": ["GOLD", "NAS100", "US30", "BTC"],
    "C06-ASIA": ["GOLD", "NAS100", "US30", "BTC"],
    "C07-COMEXTRAP-XAU": ["GOLD"],
    "C08-SMT1": ["NAS100", "US30", "GOLD", "BTC (Binance)"],
    "C09-OPENDRIVE": ["NAS100", "US30"],
    "C10-FBR": ["GOLD", "NAS100", "US30", "BTC"],
    "C11-FB20": ["GOLD", "NAS100", "US30", "BTC"],
    "C12-IFVG": ["GOLD", "NAS100", "US30", "BTC"],
    "C13-EXH3": ["GOLD", "NAS100", "US30", "BTC"],
    "C14-TDC": ["GOLD", "NAS100", "US30", "BTC"],
}
_GENERIC_CONFIRM = {"GOLD": ["SILVER"], "NAS100": ["US500"], "US30": ["US500"],
                    "BTC": ["BTC (Binance)", "ETH (Binance)"]}
CONFIRM = {cand: {m: list(_GENERIC_CONFIRM.get(m, [])) for m in mk} for cand, mk in PRIMARY.items()}
CONFIRM["C08-SMT1"] = {"NAS100": ["US500"], "US30": ["US500"], "GOLD": ["SILVER"],
                       "BTC (Binance)": ["ETH (Binance)"]}
# the "US500 traded with partner NAS100" confirmation: PARTNER["US500"] = "NAS100"

LOW_FREQ = {"C01-IDXDIP4", "C07-COMEXTRAP-XAU", "C09-OPENDRIVE", "C10-FBR", "C13-EXH3"}

CELLS, CONTROLS = {}, {}


def _cell(cand, params, factory, controls=()):
    cid = f"{cand}[" + ",".join(f"{k}={v}" for k, v in params.items()) + "]"
    CELLS[cid] = {"candidate": cand, "params": params, "markets": PRIMARY[cand],
                  "controls": list(controls), "factory": factory}
    return cid


def _ctrl(cand, name, kind, factory):
    cid = f"{cand}:{name}"
    if cid not in CONTROLS:
        CONTROLS[cid] = {"candidate": cand, "kind": kind, "markets": PRIMARY[cand], "factory": factory}
    return cid


def _m(fn):
    """Factory for builders that do not depend on the market."""
    return lambda market: fn


for S_ in ("3.0atr1h", "1.5atrD"):
    for T_ in (0.75, 1.25):
        ctl = _ctrl("C01-IDXDIP4", f"CONTROL[S={S_},T={T_}]", "control", _m(c01(S_, T_, control=True)))
        _cell("C01-IDXDIP4", {"S": S_, "T": T_}, _m(c01(S_, T_)), [ctl])

_dn = _ctrl("C02-FUNDFADE-BTC", "BENCH-DENICOLA", "benchmark", _m(c02_denicola))
for k_ in (1.5, 2.5):
    for hh_ in (4, 8):
        ctl = _ctrl("C02-FUNDFADE-BTC", f"PLACEBO-OFFFUNDING[k={k_},hold={hh_}]", "control",
                    _m(c02(k_, hh_, hours=(4, 12, 20))))
        _cell("C02-FUNDFADE-BTC", {"k": k_, "hold": hh_}, _m(c02(k_, hh_)), [ctl, _dn])

for fill_ in (0.5, 0.75):
    for m_ in (1.0, 1.5):
        _cell("C03-GAPHALF", {"fill": fill_, "m": m_}, _m(c03(fill_, m_)))

for g_ in (0.3, 0.6):
    for b_ in (0.5, 1.0):
        _cell("C04-GAPFILL", {"gmin": g_, "b": b_}, _m(c04(g_, b_)))          # control = C03 cells

for LB_ in (12, 24):
    ctl = _ctrl("C05-ENGF", f"CONTROL[LB={LB_}]", "control", _m(c05_control(LB_)))
    for CW_ in (4, 8):
        _cell("C05-ENGF", {"LB": LB_, "CW": CW_}, _m(c05(LB_, CW_)), [ctl])

for K_ in (0.5, 0.25):
    ctl = _ctrl("C06-ASIA", f"PLACEBO[K={K_}]", "placebo_c06",
                (lambda K: lambda market: c06(K, False, placebo=True, market=market))(K_))
    for filt_ in ("off", "on"):
        _cell("C06-ASIA", {"K": K_, "filter": filt_},
              (lambda K, fl: lambda market: c06(K, fl == "on", market=market))(K_, filt_), [ctl])

for k_ in (1.5, 2.5):
    for b_ in (0.5, 1.0):
        _cell("C07-COMEXTRAP-XAU", {"k": k_, "b": b_}, _m(c07(k_, b_)))

for L_ in (24, 48):
    ctl = _ctrl("C08-SMT1", f"CONTROL-NOPARTNER[L={L_}]", "control",
                (lambda L: lambda market: c08("A", L, market, control=True))(L_))
    for leg_ in ("A", "B"):
        _cell("C08-SMT1", {"L": L_, "leg": leg_},
              (lambda L, lg: lambda market: c08(lg, L, market))(L_, leg_), [ctl] if leg_ == "A" else [])

for d_ in (1.0, 1.5):
    for r_ in (0.5, 0.75):
        _cell("C09-OPENDRIVE", {"d": d_, "r": r_}, _m(c09(d_, r_)))

for c_ in (0.5, 0.7):
    for mode_ in ("edge", "extreme"):
        ctl = _ctrl("C10-FBR", f"CONTROL-FIRSTBREAK[c={c_},stop={mode_}]", "control",
                    _m(c10(c_, mode_, control=True)))
        _cell("C10-FBR", {"c": c_, "stop": mode_}, _m(c10(c_, mode_)), [ctl])

for W_ in (12, 24):
    diag = _ctrl("C11-FB20", f"DIAG-NOREGIME[W={W_}]", "diagnostic", _m(c11(W_, np.inf)))
    for er_ in (0.20, 0.35):
        _cell("C11-FB20", {"W": W_, "ERMAX": er_}, _m(c11(W_, er_)), [diag])

for mode_ in ("F1", "F2"):
    ctl = _ctrl("C12-IFVG", f"CONTROL-NOGAP[mode={mode_}]", "control",
                (lambda md: lambda market: c12(0.0, md, market, control=True))(mode_))
    for G_ in (0.25, 0.5):
        _cell("C12-IFVG", {"G": G_, "mode": mode_},
              (lambda G, md: lambda market: c12(G, md, market))(G_, mode_), [ctl])

for R_ in (1.0, 2.0):
    ctl = _ctrl("C13-EXH3", f"CONTROL-N1[R={R_}]", "control", _m(c13(1, R_)))
    for N_ in (2, 3):
        _cell("C13-EXH3", {"N": N_, "R": R_}, _m(c13(N_, R_)), [ctl])

for H_ in (2, 6):
    ctl = _ctrl("C14-TDC", f"CONTROL-SINGLE-U4[hold={H_}]", "control",
                (lambda H: lambda market: c14(1, H, market, control=True))(H_))
    for K_ in (3, 4):
        _cell("C14-TDC", {"K": K_, "hold": H_},
              (lambda K, H: lambda market: c14(K, H, market))(K_, H_), [ctl])

CANDIDATES = {cid: {m: meta["factory"](m) for m in meta["markets"]} for cid, meta in CELLS.items()}

# G9 level placebo (mandatory for C06, C08, C11, C13). C08 uses its L=24 level.
LEVELS = {"C06-ASIA": c06_levels, "C08-SMT1": lambda m: c08_levels(m, 24),
          "C11-FB20": c11_levels, "C13-EXH3": c13_levels}


def builder_for(cid, market):
    """Builder for a cell or control on any market (primary or confirmation)."""
    meta = CELLS.get(cid) or CONTROLS[cid]
    return meta["factory"](market)


# =============================================================== G9 level placebo
def level_placebo(f, up, dn, mask=None, horizon=12, eps=0.05, bucket=0.1):
    """EVENT: first 5m close beyond the level by >= eps*A (first per constant-level
    run). STAT: P(a close back on the other side within `horizon` bars).
    PLACEBO: same statistic on all bars whose 3-bar move has the same size (in A
    units, `bucket` buckets), with the 'level' = the close 3 bars earlier.
    Returns a dict with rates in % and the difference in points."""
    c, A = f["close"], f["atr1h"]
    n = len(c)
    mask = np.ones(n, bool) if mask is None else mask
    s = pd.Series(c)
    fmin = s[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1).to_numpy()
    fmax = s[::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1).to_numpy()
    c3 = s.shift(3).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mv = (c - c3) / A
    nb = 60
    bk = np.clip(np.floor(np.abs(mv) / bucket), 0, nb - 1)
    good = np.isfinite(mv) & np.isfinite(fmin) & np.isfinite(fmax) & mask
    res = {}
    ev_all, pl_all, pd_all = [], [], []
    for side, lvl in (("up", up), ("dn", dn)):
        lv = np.asarray(lvl, float)
        with np.errstate(invalid="ignore"):
            beyond = ((c - lv) >= eps * A) if side == "up" else ((lv - c) >= eps * A)
            run = np.cumsum(np.concatenate([[True], ~((lv[1:] == lv[:-1]) & np.isfinite(lv[1:]))]))
        prev_c = np.concatenate([[np.nan], c[:-1]])
        with np.errstate(invalid="ignore"):
            prev_beyond = ((prev_c - lv) >= eps * A) if side == "up" else ((lv - prev_c) >= eps * A)
        bs = pd.Series(beyond & ~prev_beyond & np.isfinite(lv))
        first = bs & (bs.astype(int).groupby(run).cumsum() == 1)
        with np.errstate(invalid="ignore"):
            sgn_ok = (mv > 0) if side == "up" else (mv < 0)
        evm = first.to_numpy() & good & sgn_ok
        with np.errstate(invalid="ignore"):
            ev_hit = (fmin < lv) if side == "up" else (fmax > lv)
            pl_hit = (fmin < c3) if side == "up" else (fmax > c3)
        pm = good & sgn_ok
        bi = bk[pm].astype(int)
        tot = np.bincount(bi, minlength=nb)
        hit = np.bincount(bi, weights=pl_hit[pm].astype(float), minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(tot > 0, hit / tot, np.nan)
        eb = bk[evm].astype(int)
        ev_all.append(ev_hit[evm].astype(float))
        pl_all.append(rate[eb])
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = ((c - lv) if side == "up" else (lv - c)) / A
        ebd = np.clip(np.floor(dist[evm] / bucket), 0, nb - 1).astype(int)
        pd_all.append(rate[ebd])
        res[f"n_{side}"] = int(evm.sum())
    ev = np.concatenate(ev_all)
    pl = np.concatenate(pl_all)
    pdm = np.concatenate(pd_all)
    ok = np.isfinite(pl)
    if not ok.any():
        return {**res, "n": 0, "event_pct": float("nan"), "placebo_pct": float("nan"), "diff_pts": float("nan")}
    e_pct, p_pct = 100 * ev[ok].mean(), 100 * pl[ok].mean()
    return {**res, "n": int(ok.sum()), "event_pct": round(float(e_pct), 2),
            "placebo_pct": round(float(p_pct), 2), "diff_pts": round(float(e_pct - p_pct), 2),
            # DIAGNOSTIC only (not the pre-registered rule): placebo matched on the
            # event's distance beyond its level instead of its 3-bar move size
            "diff_pts_dist_matched": round(float(e_pct - 100 * np.nanmean(pdm[ok])), 2)}
