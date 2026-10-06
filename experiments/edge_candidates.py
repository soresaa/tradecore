"""
Section 3 candidates of PREREGISTRATION_EDGES.md, exactly as registered.

Each builder takes (m5, f) and returns deep_search_study.sig_frame rows: every
signal carries its own stop, targets, holding time and exit rule. Times are
New York local (America/New_York, DST handled by the tz database); every
decision uses completed 5-minute bars only.
"""
import numpy as np
import pandas as pd

from deep_search_study import sig_frame

NO_TARGET = 100.0            # time-exit strategies: a target that is never reached


def ny_clock(idx):
    loc = idx.tz_localize("UTC").tz_convert("America/New_York")
    return (loc.hour * 60 + loc.minute).to_numpy(), loc.normalize().tz_localize(None)


def bars_at(minutes, dates, hhmm):
    """Series: NY date -> position of the 5-minute bar STARTING at hh:mm New York."""
    h, m = hhmm
    sel = np.flatnonzero(minutes == h * 60 + m)
    return pd.Series(sel, index=pd.DatetimeIndex(dates[sel])).groupby(level=0).first()


def time_exit_frame(f, pos, buy, exit_pos, stop_mult):
    """Time-exit trades: disaster stop at stop_mult x 1H ATR (= 1R), no target.
    The simulator starts on the bar after the signal and exits at the close of the
    first bar opening at/after (that bar's open + hold), so
    hold = exit bar open - (signal bar open + 5 minutes)."""
    idx, c, atr = f["idx"], f["close"], f["atr1h"]
    pos = np.asarray(pos, int)
    exit_pos = np.asarray(exit_pos, int)
    buy_arr = np.zeros(len(c), bool)
    buy_arr[pos] = np.asarray(buy, bool)
    risk = stop_mult * atr
    sign = np.where(buy_arr, 1.0, -1.0)
    stop = c - sign * risk
    tp = c + sign * NO_TARGET * risk
    hold = np.full(len(c), np.nan)
    if len(pos):
        hold[pos] = (idx[exit_pos] - (idx[pos] + pd.Timedelta(minutes=5))).total_seconds() / 3600.0
    ok = np.isfinite(hold[pos]) & (hold[pos] > 0) if len(pos) else np.zeros(0, bool)
    return sig_frame(idx, pos[ok], buy_arr, c, stop, tp, tp, hold, "no_breakeven")


def c1_intraday_momentum(signal_hhmm, ref_hhmm, exit_hhmm):
    """C1 (N1, Baltussen et al. 2021): r_ROD = close of the signal bar / close of the
    reference bar on the previous NY trading date - 1; trade its sign until the
    exit bar closes."""
    def build(m5, f):
        c = f["close"]
        minutes, dates = ny_clock(f["idx"])
        sig_at = bars_at(minutes, dates, signal_hhmm)
        ref_at = bars_at(minutes, dates, ref_hhmm)
        ex_at = bars_at(minutes, dates, exit_hhmm)
        pos, buy, ex = [], [], []
        ref_dates = ref_at.index
        for d, p in sig_at.items():
            k = ref_dates.searchsorted(d) - 1          # previous NY date that has a close
            if k < 0 or d not in ex_at.index or ex_at[d] <= p:
                continue
            r = c[p] / c[ref_at.iloc[k]] - 1.0
            if not np.isfinite(r) or r == 0:
                continue
            pos.append(p)
            buy.append(r > 0)
            ex.append(ex_at[d])
        return time_exit_frame(f, pos, buy, ex, 2.0)
    return build


def c2_overnight(m5, f):
    """C2 (N2): long at the 16:00 NY close, out at the 09:30 NY open next trading day."""
    minutes, dates = ny_clock(f["idx"])
    close_at = bars_at(minutes, dates, (15, 55))       # the bar that closes at 16:00
    open_at = bars_at(minutes, dates, (9, 25))         # the bar that closes at 09:30
    pos, ex = [], []
    for d, p in close_at.items():
        k = open_at.index.searchsorted(d, side="right")
        if k >= len(open_at):
            continue
        pos.append(p)
        ex.append(open_at.iloc[k])
    return time_exit_frame(f, pos, [True] * len(pos), ex, 2.0)


def c3_turn_of_month(m5, f):
    """C3 (N4, McConnell & Xu): long from the 16:00 close of the second-to-last trading
    day of a month to the 16:00 close of the 3rd trading day of the next month."""
    minutes, dates = ny_clock(f["idx"])
    close_at = bars_at(minutes, dates, (15, 55))
    days = close_at.index
    groups = {}
    for dday in days:
        groups.setdefault((dday.year, dday.month), []).append(dday)
    keys = sorted(groups)
    pos, ex = [], []
    for a, b in zip(keys[:-1], keys[1:]):
        this, nxt = groups[a], groups[b]
        if len(this) < 2 or len(nxt) < 3:
            continue
        pos.append(close_at[this[-2]])
        ex.append(close_at[nxt[2]])
    return time_exit_frame(f, pos, [True] * len(pos), ex, 2.0)


ROUND_STEPS = {
    "GOLD": lambda p: np.where(p >= 1000, 50.0, 10.0),
    "GOLD (broker)": lambda p: np.where(p >= 1000, 50.0, 10.0),
    "SILVER": lambda p: np.where(p >= 10, 1.0, 0.5),
    "NAS100": lambda p: np.full_like(p, 100.0),
    "US500": lambda p: np.full_like(p, 50.0),
    "BTC": lambda p: np.where(p >= 10000, 1000.0, 100.0),
    "BTC (Binance)": lambda p: np.where(p >= 10000, 1000.0, 100.0),
    "ETH (Binance)": lambda p: np.where(p >= 1000, 100.0, 10.0),
}


def c4_round_cascade(market):
    """C4 (N7, Osler): a 5-minute close beyond a round level the previous close was on
    the other side of -> trade the cross direction for 2 hours; stop 1.5 x 1H ATR."""
    def build(m5, f):
        idx, c, atr = f["idx"], f["close"], f["atr1h"]
        prev = np.concatenate([[np.nan], c[:-1]])
        step = ROUND_STEPS[market](np.nan_to_num(prev, nan=1.0))
        up = np.floor(c / step) > np.floor(prev / step)
        dn = np.floor(c / step) < np.floor(prev / step)
        pos = np.flatnonzero((up | dn) & np.isfinite(atr) & np.isfinite(prev))
        risk = 1.5 * atr
        sign = np.where(up, 1.0, -1.0)
        stop = c - sign * risk
        tp = c + sign * NO_TARGET * risk
        return sig_frame(idx, pos, up, c, stop, tp, tp, 115.0 / 60.0, "no_breakeven")
    return build


def c5_liquidity_sweep(m5, f):
    """C5 (N8, ICT): first sweep per side per NY day of the previous NY day's high/low
    (trades beyond, closes back inside) -> fade; stop 0.1 x 1H ATR beyond the sweep
    extreme; 2R target; 24-hour limit."""
    idx, c, h, l, atr = f["idx"], f["close"], f["high"], f["low"], f["atr1h"]
    _, dates = ny_clock(idx)
    di = pd.DatetimeIndex(dates)
    dh = pd.Series(h, index=di).groupby(level=0).max()
    dl = pd.Series(l, index=di).groupby(level=0).min()
    prev_h = dh.shift(1).reindex(di).to_numpy(float)     # the previous NY day, completed
    prev_l = dl.shift(1).reindex(di).to_numpy(float)
    swept_hi = (h > prev_h) & (c < prev_h)
    swept_lo = (l < prev_l) & (c > prev_l)
    first_hi = pd.Series(swept_hi.astype(int), index=di).groupby(level=0).cumsum().to_numpy() == 1
    first_lo = pd.Series(swept_lo.astype(int), index=di).groupby(level=0).cumsum().to_numpy() == 1
    sell = swept_hi & first_hi & np.isfinite(atr)
    buy = swept_lo & first_lo & np.isfinite(atr) & ~sell
    pos = np.flatnonzero(sell | buy)
    stop = np.where(buy, l - 0.1 * atr, h + 0.1 * atr)
    risk = np.abs(c - stop)
    tp = np.where(buy, c + 2.0 * risk, c - 2.0 * risk)
    return sig_frame(idx, pos, buy, c, stop, tp, tp, 24.0, "no_breakeven")


CANDIDATES = {
    "C1": {"GOLD": c1_intraday_momentum((12, 55), (13, 25), (13, 25)),
           "NAS100": c1_intraday_momentum((15, 25), (15, 55), (15, 55))},
    "C2": {"NAS100": c2_overnight},
    "C3": {"NAS100": c3_turn_of_month},
    "C4": {m: c4_round_cascade(m) for m in ("GOLD", "NAS100", "BTC")},
    "C5": {m: c5_liquidity_sweep for m in ("GOLD", "NAS100", "BTC")},
}
CONFIRM = {"NAS100": ["US500"], "BTC": ["ETH (Binance)"], "GOLD": ["GOLD (broker)", "SILVER"]}
BASE_OF = {"US500": "NAS100", "ETH (Binance)": "BTC", "GOLD (broker)": "GOLD", "SILVER": "GOLD"}


def builder_for(cand, market):
    table = CANDIDATES[cand]
    if market in table:
        return table[market]
    if cand == "C4":
        return c4_round_cascade(market)
    return table[BASE_OF[market]]
