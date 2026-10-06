"""
Runs PREREGISTRATION_BEST_PER_MARKET.md: six entry ideas x seven markets.

  python experiments/best_per_market_study.py [workers]

Every candidate shares the live strategy's trade management (1.5 x 1H ATR stop,
TP1 -> breakeven, TP2 = 2.35R, 24h limit, one position at a time) and is
simulated by backtest.simulate_trade with that market's own measured costs.
Only the entry idea differs.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from frequency_study import features, stats
from markets_study import MARKETS, cost_model, WARMUP_DAYS, max_concurrent

ATR_MULT, RR2, MAX_HOLD = 1.5, 2.35, 24.0
CANDIDATES = ["A BREAK", "B DONCHIAN", "C MA TREND", "D BB FADE", "E ORB", "F SQUEEZE"]


# ------------------------------------------------------------------ helpers
def hourly_frame(m5: pd.DataFrame):
    from backtest import resample
    return resample(m5, "1h")


def map_to_5m(series: pd.Series, idx: pd.DatetimeIndex, floor: str) -> np.ndarray:
    """A completed higher-timeframe value, known to every 5-minute candle inside
    the NEXT bar of that timeframe (shift(1) before mapping = no lookahead)."""
    return series.shift(1).reindex(idx.floor(floor)).to_numpy(float)


def pack(idx, pos, buy, entry, risk, minute_mask=None) -> pd.DataFrame:
    keep = np.isfinite(risk[pos]) & (risk[pos] > 0)
    pos = pos[keep]
    return pd.DataFrame({
        "pos": pos, "time": idx[pos],
        "direction": np.where(buy[pos], "buy", "sell"),
        "entry": entry[pos], "risk": risk[pos],
    })


# ------------------------------------------------------------------ candidates
def sig_break(m5, f):
    """A: the live gold rules, gates included."""
    idx = f["idx"]
    hourly = (idx.minute == 0)
    ok = (f["long"] | f["short"]) & hourly & (f["er_h1"] >= 0.25) & (f["er_d"] >= 0.20)
    pos = np.flatnonzero(ok)
    risk = ATR_MULT * f["atr1h"]
    return pack(idx, pos, f["long"], f["close"], risk)


def sig_donchian(m5, f):
    """B: break of the 20-bar 1H Donchian channel, with the daily trend."""
    idx, c = f["idx"], f["close"]
    h1 = hourly_frame(m5)
    hi20 = map_to_5m(h1["high"].rolling(20).max(), idx, "1h")
    lo20 = map_to_5m(h1["low"].rolling(20).min(), idx, "1h")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = (c > hi20) & (prev_c <= hi20) & f["daily_bull"]
    dn = (c < lo20) & (prev_c >= lo20) & f["daily_bear"]
    hourly = (idx.minute == 0)
    pos = np.flatnonzero((up | dn) & hourly & np.isfinite(f["atr1h"]))
    return pack(idx, pos, up, c, ATR_MULT * f["atr1h"])


def sig_ma_trend(m5, f):
    """C: 1H EMA50 vs EMA200 regime, entry on the first close back through EMA20."""
    idx, c = f["idx"], f["close"]
    h1 = hourly_frame(m5)
    hc = h1["close"]
    e20 = map_to_5m(hc.ewm(span=20, adjust=False).mean(), idx, "1h")
    e50 = map_to_5m(hc.ewm(span=50, adjust=False).mean(), idx, "1h")
    e200 = map_to_5m(hc.ewm(span=200, adjust=False).mean(), idx, "1h")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = (e50 > e200) & (prev_c <= e20) & (c > e20)
    dn = (e50 < e200) & (prev_c >= e20) & (c < e20)
    hourly = (idx.minute == 0)
    pos = np.flatnonzero((up | dn) & hourly & np.isfinite(f["atr1h"]))
    return pack(idx, pos, up, c, ATR_MULT * f["atr1h"])


def sig_bb_fade(m5, f):
    """D: mean reversion — 15-minute close outside the 2-sigma band, in chop."""
    from backtest import resample
    from market_strategies import bb_fade_levels
    idx, c = f["idx"], f["close"]
    m15 = resample(m5, "15min")
    up_s, lo_s = bb_fade_levels(m15["close"])       # the live definition, shared
    upper = map_to_5m(up_s, idx, "15min")
    lower = map_to_5m(lo_s, idx, "15min")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    chop = f["er_d"] < 0.20
    dn = (prev_c > upper) & (c <= upper) & chop        # fade back down
    up = (prev_c < lower) & (c >= lower) & chop        # fade back up
    hourly = (idx.minute == 0)
    pos = np.flatnonzero((up | dn) & hourly & np.isfinite(f["atr1h"]))
    return pack(idx, pos, up, c, ATR_MULT * f["atr1h"])


def sig_orb(m5, f):
    """E: first-hour range of the London and New York sessions, then the break."""
    idx, c = f["idx"], f["close"]
    utc = idx.tz_localize("UTC")
    out = []
    for tz, start_min, win_min in (("Europe/London", 8 * 60, 12 * 60),
                                   ("America/New_York", 9 * 60 + 30, 13 * 60 + 30)):
        local = utc.tz_convert(tz)
        mins = local.hour * 60 + local.minute
        day = pd.Series(local.normalize().tz_localize(None), index=idx)
        in_range = (mins >= start_min) & (mins < start_min + 60)
        g = m5[in_range].groupby(day[in_range])
        rng = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(), "n": g.size()})
        rng = rng[rng["n"] >= 6]
        in_win = (mins >= start_min + 60) & (mins < win_min)
        w = pd.DataFrame({"close": c[in_win], "day": day[in_win].to_numpy(),
                          "pos": np.flatnonzero(in_win)})
        w = w.join(rng, on="day", how="inner")
        w = w[(w["close"] > w["hi"]) | (w["close"] < w["lo"])]
        first = w.groupby("day", sort=True).head(1)
        out.append(pd.DataFrame({"pos": first["pos"].to_numpy(),
                                 "buy": (first["close"] > first["hi"]).to_numpy()}))
    both = pd.concat(out).sort_values("pos")
    pos = both["pos"].to_numpy()
    buy = np.zeros(len(c), bool)
    buy[pos] = both["buy"].to_numpy()
    pos = pos[np.isfinite(f["atr1h"][pos])]
    return pack(idx, pos, buy, c, ATR_MULT * f["atr1h"])


def sig_squeeze(m5, f):
    """F: 1H ATR in the bottom fifth of the last 20 days, then a 12-hour range break."""
    from market_strategies import squeeze_state
    idx, c = f["idx"], f["close"]
    h1 = hourly_frame(m5)
    quiet, hi12, lo12 = squeeze_state(h1)           # the live definition, shared
    q = map_to_5m(quiet.astype(float), idx, "1h") > 0.5
    hi = map_to_5m(hi12, idx, "1h")
    lo = map_to_5m(lo12, idx, "1h")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = q & (c > hi) & (prev_c <= hi)
    dn = q & (c < lo) & (prev_c >= lo)
    hourly = (idx.minute == 0)
    pos = np.flatnonzero((up | dn) & hourly & np.isfinite(f["atr1h"]))
    return pack(idx, pos, up, c, ATR_MULT * f["atr1h"])


BUILDERS = {"A BREAK": sig_break, "B DONCHIAN": sig_donchian, "C MA TREND": sig_ma_trend,
            "D BB FADE": sig_bb_fade, "E ORB": sig_orb, "F SQUEEZE": sig_squeeze}


# ------------------------------------------------------------------ simulation
def run_signals(m5, sig, costs, start, end):
    """One position at a time, the live management, this market's costs."""
    from backtest import simulate_trade
    from engine import TradePlan
    n = len(m5)
    trades, busy_until = [], None
    for r in sig.itertuples(index=False):
        if r.time < start or r.time >= end or r.pos + 1 >= n:
            continue
        if busy_until is not None and r.time < busy_until:
            continue
        sign = 1 if r.direction == "buy" else -1
        plan = TradePlan(r.direction, r.entry, r.entry, r.entry - sign * r.risk,
                         r.entry + sign * r.risk, r.entry + sign * RR2 * r.risk, 1.0, RR2)
        res = simulate_trade(m5, r.pos + 1, plan, costs=costs, exit_rule="breakeven",
                             max_hold_hours=MAX_HOLD)
        res.update(time=r.time, direction=r.direction, risk=round(float(r.risk), 5))
        trades.append(res)
        busy_until = res["exit_time"]
    return pd.DataFrame(trades)


def run_one(job):
    label, cand = job
    from backtest import load_real_data
    t0 = time.time()
    file, key = MARKETS[label]
    m5 = load_real_data(os.path.join(PROJ, file))
    costs = cost_model(key)
    price_now = float(m5["close"].tail(8640).median())
    m5 = m5.copy()
    m5["spread"] = costs.spread_price * m5["close"] / price_now     # the agreed cost basis
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=WARMUP_DAYS)
    mid = start + (last - start) / 2
    f = features(m5)
    f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
    sig = BUILDERS[cand](m5, f)
    tr = run_signals(m5, sig, costs, start, last)
    if not tr.empty:
        tr["market"], tr["candidate"] = label, cand
    days = pd.Series(m5.index.normalize().unique())
    meta = {"start": str(start), "mid": str(mid), "last": str(last),
            "train_days": int(((days >= start) & (days < mid) & (days.dt.dayofweek < 5)).sum()),
            "test_days": int(((days >= mid) & (days <= last) & (days.dt.dayofweek < 5)).sum()),
            "train_years": (mid - start).days / 365.25, "test_years": (last - mid).days / 365.25,
            "median_stop": float(tr["risk"].median()) if not tr.empty else float("nan")}
    print(f"  {label:<16} {cand:<12} {len(tr):>5} trades ({time.time()-t0:.0f}s)", flush=True)
    return label, cand, tr, meta


def daily_dir(m5, f):
    """Daily EMA20/50 direction, as the live strategy defines bull and bear."""
    from backtest import resample
    idx, c = f["idx"], f["close"]
    d1 = resample(m5, "1D")
    dc = d1["close"]
    e20 = map_to_5m(dc.ewm(span=20, adjust=False).mean(), idx, "1D")
    e50 = map_to_5m(dc.ewm(span=50, adjust=False).mean(), idx, "1D")
    a20, a50 = 2.0 / 21.0, 2.0 / 51.0
    e20 = e20 + a20 * (c - e20)
    e50 = e50 + a50 * (c - e50)
    return (c > e50) & (e20 > e50), (c < e50) & (e20 < e50)


# ------------------------------------------------------------------ main
def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    labels = [k for k in MARKETS if os.path.exists(os.path.join(PROJ, MARKETS[k][0]))]
    jobs = [(l, c) for l in labels for c in CANDIDATES]
    print(f"{len(jobs)} combinations ({len(labels)} markets x {len(CANDIDATES)} strategies), "
          f"{workers} workers\n", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(run_one, jobs))

    rows = [tr for _, _, tr, _ in out if tr is not None and not tr.empty]
    allt = pd.concat(rows, ignore_index=True)
    allt.to_csv(os.path.join(HERE, "best_per_market_trades.csv"), index=False)

    table = {}
    print("\n" + "=" * 130)
    print("SIX ENTRY IDEAS ON EVERY MARKET — same stop, targets and management; each market's own costs")
    print("=" * 130)
    for label in labels:
        print(f"\n{label}")
        print(f"  {'candidate':<12}{'split':<7}{'trades':>7}{'/day':>6}{'win%':>6}{'net R':>9}"
              f"{'R/year':>8}{'R/trade':>9}{'PF':>7}{'maxDD':>8}{'P(no edge)':>12}")
        table[label] = {}
        for _, cand, tr, meta in [o for o in out if o[0] == label]:
            mid = pd.Timestamp(meta["mid"])
            table[label][cand] = {"meta": meta}
            for split, (a, b, nd, yrs) in (
                    ("TRAIN", (pd.Timestamp(meta["start"]), mid, meta["train_days"], meta["train_years"])),
                    ("TEST", (mid, pd.Timestamp(meta["last"]) + pd.Timedelta(days=1),
                              meta["test_days"], meta["test_years"]))):
                part = tr[(tr["time"] >= a) & (tr["time"] < b)] if not tr.empty else tr
                s = stats(part, max(nd, 1), max(yrs, 0.1))
                table[label][cand][split] = s
                print(f"  {cand:<12}{split:<7}{s['n']:>7}{s['per_day']:>6}{s['win']:>6}{s['R']:>9}"
                      f"{s['R_year']:>8}{s['per_trade']:>9}{s['PF']:>7}{s['maxDD']:>8}{s['p_no_edge']:>11}%")

    print("\n" + "-" * 130)
    print(f"SELECTION RULE (fixed in advance) — {len(jobs)} combinations searched, "
          f"so ~{len(jobs) * 0.01:.1f} false passes are expected at the 1% bar")
    winners = {}
    for label in labels:
        qual = [c for c in CANDIDATES
                if table[label][c]["TRAIN"]["R"] > 0 and table[label][c]["TRAIN"]["PF"] >= 1.10
                and table[label][c]["TRAIN"]["n"] >= 100]
        if not qual:
            print(f"  {label:<16} no candidate qualifies on TRAIN")
            continue
        best = max(qual, key=lambda c: table[label][c]["TRAIN"]["R"])
        s = table[label][best]["TEST"]
        checks = {"net R > 0": s["R"] > 0, "PF >= 1.15": s["PF"] >= 1.15,
                  "P(no edge) <= 1%": s["p_no_edge"] <= 1.0, ">= 50 trades": s["n"] >= 50}
        ok = all(checks.values())
        print(f"  {label:<16} best on TRAIN: {best:<12} -> TEST " +
              "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()) +
              f"  => {'CONFIRMED' if ok else 'not confirmed'}")
        if ok:
            winners[label] = {"candidate": best, "train": table[label][best]["TRAIN"],
                              "test": table[label][best]["TEST"]}

    print("\nCONFIRMED STRATEGIES:", ", ".join(f"{m} = {w['candidate']}" for m, w in winners.items())
          or "none — no market gets a strategy from this search")
    with open(os.path.join(HERE, "best_per_market_decision.json"), "w") as fh:
        json.dump({"winners": winners, "table": table, "combinations": len(jobs)}, fh,
                  indent=2, default=str)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
