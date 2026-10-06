"""
Runs PREREGISTRATION_DEEP.md: nine strategy families, each with its own stop,
target and holding time, on the four non-gold markets (gold as reference).

  python experiments/deep_search_study.py search    # search window only
  python experiments/deep_search_study.py holdout   # opens the holdout, once

The holdout is a separate command on purpose: the search table has to be looked
at, and locked, before the untouched 40% of history is spent.
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
from markets_study import MARKETS, cost_model, WARMUP_DAYS
from best_per_market_study import daily_dir, hourly_frame, map_to_5m

SPLIT = 0.60                      # search on the first 60%, holdout is the rest
DECISION = os.path.join(HERE, "deep_search_decision.json")
GOLD_BAR = {"pf": 1.25, "r_year": 4.0, "r_trade": 0.08, "p_no_edge": 5.0}
QUALIFY = {"pf": 1.15, "trades": 80}

FAMILIES = ["T24", "TSW", "DDN", "BBM", "BB1", "LON", "OVN", "SQ3", "RS2"]
# The four markets the user asked about, plus gold as the yardstick. Silver and
# oil are out: every idea already tested lost on them (RESULTS_BEST_PER_MARKET.md).
MARKET_SUBSET = ["XAU/USD gold", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD"]


# ----------------------------------------------------------------- utilities
def sig_frame(idx, pos, buy, entry, stop, tp1, tp2, hold, rule):
    """One row per signal, carrying its own management."""
    pos = np.asarray(pos, int)
    ok = np.isfinite(entry[pos]) & np.isfinite(stop[pos]) & (np.abs(entry[pos] - stop[pos]) > 0)
    pos = pos[ok]
    risk = np.abs(entry[pos] - stop[pos])
    return pd.DataFrame({
        "pos": pos, "time": idx[pos], "direction": np.where(buy[pos], "buy", "sell"),
        "entry": entry[pos], "stop": stop[pos], "risk": risk,
        "tp1": tp1[pos], "tp2": tp2[pos],
        "hold": hold[pos] if isinstance(hold, np.ndarray) else np.full(len(pos), float(hold)),
        "rule": rule,
    })


def rr_targets(entry, buy, risk, r1, r2):
    sign = np.where(buy, 1.0, -1.0)
    return entry + sign * r1 * risk, entry + sign * r2 * risk


def daily_atr(m5, idx, length=14):
    from backtest import resample
    d1 = resample(m5, "1D")
    prev = d1["close"].shift(1)
    tr = pd.concat([d1["high"] - d1["low"], (d1["high"] - prev).abs(),
                    (d1["low"] - prev).abs()], axis=1).max(axis=1)
    return map_to_5m(tr.rolling(length).mean(), idx, "1D")


def hourly_rsi(m5, idx, length=2):
    h1 = hourly_frame(m5)
    d = h1["close"].diff()
    up = d.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return map_to_5m(100 - 100 / (1 + rs), idx, "1h")


def local_minutes(idx, tz):
    local = idx.tz_localize("UTC").tz_convert(tz)
    return (local.hour * 60 + local.minute).to_numpy(), local


# ----------------------------------------------------------------- families
def fam_T24(m5, f):
    idx, c = f["idx"], f["close"]
    ok = (f["long"] | f["short"]) & (idx.minute == 0) & (f["er_h1"] >= 0.25) & (f["er_d"] >= 0.20)
    pos = np.flatnonzero(ok & np.isfinite(f["atr1h"]))
    buy, risk = f["long"], 1.5 * f["atr1h"]
    stop = c - np.where(buy, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, buy, risk, 1.0, 2.35)
    return sig_frame(idx, pos, buy, c, stop, tp1, tp2, 24.0, "breakeven")


def _donchian_break(m5, f, bars=20):
    idx, c = f["idx"], f["close"]
    h1 = hourly_frame(m5)
    hi = map_to_5m(h1["high"].rolling(bars).max(), idx, "1h")
    lo = map_to_5m(h1["low"].rolling(bars).min(), idx, "1h")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = (c > hi) & (prev_c <= hi) & f["daily_bull"]
    dn = (c < lo) & (prev_c >= lo) & f["daily_bear"]
    return up, dn


def fam_TSW(m5, f):
    idx, c = f["idx"], f["close"]
    up, dn = _donchian_break(m5, f)
    pos = np.flatnonzero((up | dn) & (idx.minute == 0) & np.isfinite(f["atr1h"]))
    risk = 2.0 * f["atr1h"]
    stop = c - np.where(up, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, up, risk, 2.0, 4.0)
    return sig_frame(idx, pos, up, c, stop, tp1, tp2, 120.0, "breakeven")


def fam_DDN(m5, f):
    """20-DAY Donchian break, decided on the first candle of a new day."""
    from backtest import resample
    idx, c = f["idx"], f["close"]
    d1 = resample(m5, "1D")
    hi = map_to_5m(d1["high"].rolling(20).max(), idx, "1D")
    lo = map_to_5m(d1["low"].rolling(20).min(), idx, "1D")
    atr_d = daily_atr(m5, idx)
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = (c > hi) & (prev_c <= hi)
    dn = (c < lo) & (prev_c >= lo)
    pos = np.flatnonzero((up | dn) & (idx.minute == 0) & np.isfinite(atr_d))
    risk = 2.0 * atr_d
    stop = c - np.where(up, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, up, risk, 2.0, 4.0)
    return sig_frame(idx, pos, up, c, stop, tp1, tp2, 240.0, "trail_1r")


def _bb_entries(m5, f):
    from backtest import resample
    from market_strategies import bb_fade_levels
    idx, c = f["idx"], f["close"]
    m15 = resample(m5, "15min")
    up_s, lo_s = bb_fade_levels(m15["close"])
    mid_s = m15["close"].rolling(20).mean()
    upper = map_to_5m(up_s, idx, "15min")
    lower = map_to_5m(lo_s, idx, "15min")
    mid = map_to_5m(mid_s, idx, "15min")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    chop = f["er_d"] < 0.20
    sell = (prev_c > upper) & (c <= upper) & chop
    buy = (prev_c < lower) & (c >= lower) & chop
    return buy, sell, mid


def fam_BBM(m5, f):
    """Fade the band, target the middle band (what mean reversion is for)."""
    idx, c = f["idx"], f["close"]
    buy, sell, mid = _bb_entries(m5, f)
    risk = 1.0 * f["atr1h"]
    stop = c - np.where(buy, 1.0, -1.0) * risk
    pos = np.flatnonzero((buy | sell) & (idx.minute == 0) & np.isfinite(risk) & np.isfinite(mid))
    return sig_frame(idx, pos, buy, c, stop, mid, mid, 12.0, "no_breakeven")


def fam_BB1(m5, f):
    idx, c = f["idx"], f["close"]
    buy, sell, _ = _bb_entries(m5, f)
    risk = 1.0 * f["atr1h"]
    stop = c - np.where(buy, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, buy, risk, 1.0, 1.0)
    pos = np.flatnonzero((buy | sell) & (idx.minute == 0) & np.isfinite(risk))
    return sig_frame(idx, pos, buy, c, stop, tp1, tp2, 12.0, "no_breakeven")


def fam_LON(m5, f):
    """London-open momentum: the direction of the last 4 hours, held to 16:00 London."""
    idx, c = f["idx"], f["close"]
    mins, local = local_minutes(idx, "Europe/London")
    h1 = hourly_frame(m5)
    move = map_to_5m(h1["close"] - h1["close"].shift(4), idx, "1h")
    atr = f["atr1h"]
    at_open = (mins == 8 * 60)
    strong = np.abs(move) > 0.5 * atr
    buy = move > 0
    pos = np.flatnonzero(at_open & strong & np.isfinite(atr))
    risk = 1.0 * atr
    stop = c - np.where(buy, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, buy, risk, 5.0, 5.0)
    hold = np.full(len(c), 8.0)          # 08:00 -> 16:00 London
    return sig_frame(idx, pos, buy, c, stop, tp1, tp2, hold, "no_breakeven")


def fam_OVN(m5, f):
    """Overnight: enter 21:00 UTC with the daily trend, out 10 hours later."""
    idx, c = f["idx"], f["close"]
    at = (idx.hour == 21) & (idx.minute == 0)
    buy = f["daily_bull"]
    take = at & (f["daily_bull"] | f["daily_bear"]) & np.isfinite(f["atr1h"])
    pos = np.flatnonzero(take)
    risk = 1.5 * f["atr1h"]
    stop = c - np.where(buy, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, buy, risk, 5.0, 5.0)
    return sig_frame(idx, pos, buy, c, stop, tp1, tp2, 10.0, "no_breakeven")


def fam_SQ3(m5, f):
    from market_strategies import squeeze_state
    idx, c = f["idx"], f["close"]
    h1 = hourly_frame(m5)
    quiet, hi12, lo12 = squeeze_state(h1)
    q = map_to_5m(quiet.astype(float), idx, "1h") > 0.5
    hi = map_to_5m(hi12, idx, "1h")
    lo = map_to_5m(lo12, idx, "1h")
    prev_c = np.concatenate([[np.nan], c[:-1]])
    up = q & (c > hi) & (prev_c <= hi)
    dn = q & (c < lo) & (prev_c >= lo)
    pos = np.flatnonzero((up | dn) & (idx.minute == 0) & np.isfinite(f["atr1h"]))
    risk = 1.5 * f["atr1h"]
    stop = c - np.where(up, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, up, risk, 2.0, 4.0)
    return sig_frame(idx, pos, up, c, stop, tp1, tp2, 72.0, "trail_1r")


def fam_RS2(m5, f):
    """RSI(2) extreme on the 1-hour chart, only with the daily trend."""
    idx, c = f["idx"], f["close"]
    rsi = hourly_rsi(m5, idx)
    buy = (rsi < 5) & f["daily_bull"]
    sell = (rsi > 95) & f["daily_bear"]
    pos = np.flatnonzero((buy | sell) & (idx.minute == 0) & np.isfinite(f["atr1h"]))
    risk = 1.5 * f["atr1h"]
    stop = c - np.where(buy, 1.0, -1.0) * risk
    tp1, tp2 = rr_targets(c, buy, risk, 1.5, 1.5)
    return sig_frame(idx, pos, buy, c, stop, tp1, tp2, 48.0, "no_breakeven")


BUILDERS = {"T24": fam_T24, "TSW": fam_TSW, "DDN": fam_DDN, "BBM": fam_BBM, "BB1": fam_BB1,
            "LON": fam_LON, "OVN": fam_OVN, "SQ3": fam_SQ3, "RS2": fam_RS2}


# ----------------------------------------------------------------- simulation
def simulate_family(m5, sig, costs, start, end, swap_scale_to=None):
    """`swap_scale_to` = today's price: when given, each trade's swap is scaled
    by entry price / today's price, like the spread. Off by default so every
    result recorded before 2026-09-21 reproduces exactly; PREREGISTRATION_MASTER
    (amendment) explains why NAS100 needs it."""
    import dataclasses
    from backtest import simulate_trade
    from engine import TradePlan
    n = len(m5)
    trades, busy_until = [], None
    for r in sig.itertuples(index=False):
        if r.time < start or r.time >= end or r.pos + 1 >= n:
            continue
        if busy_until is not None and r.time < busy_until:
            continue
        plan = TradePlan(r.direction, r.entry, r.entry, r.stop, r.tp1, r.tp2,
                         abs(r.tp1 - r.entry) / r.risk, abs(r.tp2 - r.entry) / r.risk)
        c = costs
        if swap_scale_to:
            k = r.entry / swap_scale_to
            c = dataclasses.replace(costs, swap_long_per_night=costs.swap_long_per_night * k,
                                    swap_short_per_night=costs.swap_short_per_night * k)
        res = simulate_trade(m5, r.pos + 1, plan, costs=c, exit_rule=r.rule,
                             max_hold_hours=float(r.hold))
        res.update(time=r.time, direction=r.direction, risk=round(float(r.risk), 6))
        trades.append(res)
        busy_until = res["exit_time"]
    return pd.DataFrame(trades)


def window(m5):
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=WARMUP_DAYS)
    cut = start + (last - start) * SPLIT
    return start, cut, last


def run_one(job):
    label, fam, phase = job[:3]
    scale_swap = len(job) > 3 and job[3]
    from backtest import load_real_data
    t0 = time.time()
    file, key = MARKETS[label]
    m5 = load_real_data(os.path.join(PROJ, file))
    costs = cost_model(key)
    price_now = float(m5["close"].tail(8640).median())
    m5 = m5.copy()
    m5["spread"] = costs.spread_price * m5["close"] / price_now
    start, cut, last = window(m5)
    f = features(m5)
    f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
    sig = BUILDERS[fam](m5, f)
    a, b = (start, cut) if phase == "search" else (cut, last + pd.Timedelta(days=1))
    tr = simulate_family(m5, sig, costs, a, b, swap_scale_to=price_now if scale_swap else None)
    days = pd.Series(m5.index.normalize().unique())
    nd = int(((days >= a) & (days < b) & (days.dt.dayofweek < 5)).sum())
    yrs = (b - a).days / 365.25
    s = stats(tr, max(nd, 1), max(yrs, 0.1))
    print(f"  {label:<16}{fam:<5}{phase:<8}{s['n']:>6} trades  netR {s['R']:>8.1f}  "
          f"PF {s['PF']:>5}  R/yr {s['R_year']:>6.1f}  ({time.time()-t0:.0f}s)", flush=True)
    return label, fam, phase, s, tr


def phase_run(phase, jobs, workers=6):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_one, jobs))


def cmd_search():
    labels = [k for k in MARKET_SUBSET if os.path.exists(os.path.join(PROJ, MARKETS[k][0]))]
    jobs = [(l, fam, "search") for l in labels for fam in FAMILIES]
    print(f"SEARCH: {len(jobs)} combinations on the first {int(SPLIT*100)}% of each market\n", flush=True)
    t0 = time.time()
    out = phase_run("search", jobs)
    table, trades = {}, []
    for label, fam, _, s, tr in out:
        table.setdefault(label, {})[fam] = s
        if not tr.empty:
            tr["market"], tr["family"] = label, fam
            trades.append(tr)
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(os.path.join(HERE, "deep_search_trades.csv"),
                                                    index=False)

    print("\n" + "=" * 120)
    print(f"SEARCH WINDOW — first {int(SPLIT*100)}% of each market's history, net of its own costs")
    print("=" * 120)
    winners = {}
    for label in labels:
        print(f"\n{label}")
        print(f"  {'fam':<5}{'trades':>7}{'/day':>6}{'win%':>6}{'net R':>9}{'R/year':>8}"
              f"{'R/trade':>9}{'PF':>7}{'maxDD':>8}{'P(no edge)':>12}")
        for fam in FAMILIES:
            s = table[label][fam]
            print(f"  {fam:<5}{s['n']:>7}{s['per_day']:>6}{s['win']:>6}{s['R']:>9}{s['R_year']:>8}"
                  f"{s['per_trade']:>9}{s['PF']:>7}{s['maxDD']:>8}{s['p_no_edge']:>11}%")
        qual = [fam for fam in FAMILIES
                if table[label][fam]["R"] > 0 and table[label][fam]["PF"] >= QUALIFY["pf"]
                and table[label][fam]["n"] >= QUALIFY["trades"]]
        if qual:
            best = max(qual, key=lambda fm: table[label][fm]["R_year"])
            winners[label] = best
            print(f"  -> candidate: {best} (qualifies: {', '.join(qual)})")
        else:
            print("  -> no family qualifies")
    with open(DECISION, "w") as fh:
        json.dump({"phase": "search", "split": SPLIT, "winners": winners, "search": table,
                   "locked_at": pd.Timestamp.now().isoformat()}, fh, indent=2, default=str)
    print(f"\nlocked {len(winners)} candidates into {os.path.basename(DECISION)} "
          f"— run `holdout` to spend the untouched 40%  ({time.time()-t0:.0f}s)")


def cmd_holdout():
    with open(DECISION) as fh:
        dec = json.load(fh)
    winners = dec.get("winners", {})
    if not winners:
        print("no candidates were locked in — nothing to test")
        return
    if dec.get("holdout"):
        print("the holdout has already been used for these candidates:")
        print(json.dumps(dec["holdout"], indent=2))
        return
    jobs = [(l, fam, "holdout") for l, fam in winners.items()]
    print(f"HOLDOUT: {len(jobs)} candidates, one test each, on history the search never saw\n",
          flush=True)
    out = phase_run("holdout", jobs)
    res = {}
    print("\n" + "=" * 120)
    print("HOLDOUT — the last 40% of each market, opened once")
    print("=" * 120)
    print(f"{'market':<16}{'fam':<5}{'trades':>7}{'/day':>6}{'win%':>6}{'net R':>9}{'R/year':>8}"
          f"{'R/trade':>9}{'PF':>7}{'maxDD':>8}{'P(no edge)':>12}")
    for label, fam, _, s, tr in out:
        res[label] = {"family": fam, **s}
        print(f"{label:<16}{fam:<5}{s['n']:>7}{s['per_day']:>6}{s['win']:>6}{s['R']:>9}"
              f"{s['R_year']:>8}{s['per_trade']:>9}{s['PF']:>7}{s['maxDD']:>8}{s['p_no_edge']:>11}%")

    print("\n" + "-" * 120)
    print("GOLD-GRADE BAR (fixed in advance): PF >= 1.25, R/year >= +4.0, R/trade >= +0.08, "
          "P(no edge) <= 5%")
    for label, s in res.items():
        checks = {"PF>=1.25": s["PF"] >= GOLD_BAR["pf"], "R/year>=4": s["R_year"] >= GOLD_BAR["r_year"],
                  "R/trade>=0.08": s["per_trade"] >= GOLD_BAR["r_trade"],
                  "P(no edge)<=5%": s["p_no_edge"] <= GOLD_BAR["p_no_edge"]}
        s["gold_grade"] = all(checks.values())
        print(f"  {label:<16} {s['family']:<5} " +
              "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()) +
              f"  => {'GOLD-GRADE' if s['gold_grade'] else 'did not reach gold'}")
    dec["holdout"] = res
    dec["holdout_at"] = pd.Timestamp.now().isoformat()
    with open(DECISION, "w") as fh:
        json.dump(dec, fh, indent=2, default=str)
    passed = [l for l, s in res.items() if s["gold_grade"]]
    print("\nVERDICT:", ", ".join(passed) + " reached gold" if passed
          else "no market reached gold-grade profit")


if __name__ == "__main__":
    {"search": cmd_search, "holdout": cmd_holdout}[sys.argv[1] if len(sys.argv) > 1 else "search"]()
