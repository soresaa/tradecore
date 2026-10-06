"""
Runs PREREGISTRATION_ONE_PER_DAY.md: can the live strategy trade ~1x a day?

  python experiments/frequency_study.py check   # reproduction check only
  python experiments/frequency_study.py run     # the full study

The live entry rules (indie_signal_port.generate_signal_indie) are evaluated
once per hour on a 7-day rolling window, which is far too slow to re-run at
5-minute cadence over 23 years. They are reproduced here in vectorized form and
the reproduction is CHECKED against the existing filter_study_trades.csv result
before any candidate is reported. Every accepted signal is then simulated by
backtest.simulate_trade -- the same validated simulator, costs and management as
every other result in this project.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)

import numpy as np
import pandas as pd

DATA = os.path.join(PROJ, "xauusd_m5_dukascopy_23y.csv")
COSTS = os.path.join(PROJ, "broker_costs.json")
START, END = pd.Timestamp("2004-09-16"), pd.Timestamp("2026-09-16")
TRAIN = (START, pd.Timestamp("2015-01-01"))
TEST = (pd.Timestamp("2015-01-01"), END)
REPRO = (START, pd.Timestamp("2021-09-16"))       # filter_study window: 712 trades, +60.71R
RR2, ATR_MULT = 2.35, 1.5
A20, A50 = 2.0 / 21.0, 2.0 / 51.0

# id: (cadence minutes, 1H efficiency gate, daily filter on, daily gate, max hold hours)
VARIANTS = {
    "D0 live (hourly, ER .25, filter)": (60, 0.25, True, 0.20, 24.0),
    "D1 every 15m":                     (15, 0.25, True, 0.20, 24.0),
    "D2 every 5m":                      (5, 0.25, True, 0.20, 24.0),
    "D3 every 5m, ER .15":              (5, 0.15, True, 0.20, 24.0),
    "D4 every 5m, ER .15, no filter":   (5, 0.15, False, 0.20, 24.0),
    "D5 every 5m, no ER, no filter":    (5, 0.0, False, 0.20, 24.0),
    "D6 D4 + 8h max hold":              (5, 0.15, False, 0.20, 8.0),
    "D7 D2 + 8h max hold":              (5, 0.25, True, 0.20, 8.0),
}
BASE = "D0 live (hourly, ER .25, filter)"


# --------------------------------------------------------------- vectorized signals
def features(m5: pd.DataFrame) -> dict:
    """Per-5-minute-candle values of everything the rules look at, using only
    information available at that candle's close. `signals()` picks the firing
    candles out of this; verify_indicator_port.py replays the chart indicator's
    own state machine over it."""
    from backtest import resample

    idx = m5.index
    c = m5["close"].to_numpy(float)
    prev_c = np.concatenate([[np.nan], c[:-1]])

    def blend(prev_ema, alpha):          # the forming bar, exactly as the live code sees it
        return prev_ema + alpha * (c - prev_ema)

    # ---- 1 hour: EMAs, ATR(14), 20-period efficiency, all with the forming hour
    h1 = resample(m5, "1h")
    hc = h1["close"]
    prev_close = hc.shift(1)
    tr = pd.concat([h1["high"] - h1["low"], (h1["high"] - prev_close).abs(),
                    (h1["low"] - prev_close).abs()], axis=1).max(axis=1)
    H = pd.DataFrame({
        "e20": hc.ewm(span=20, adjust=False).mean().shift(1),
        "e50": hc.ewm(span=50, adjust=False).mean().shift(1),
        "prev_close": prev_close,
        "close20": hc.shift(20),
        "path19": hc.diff().abs().shift(1).rolling(19).sum(),
        "tr13": tr.shift(1).rolling(13).sum(),
    }).reindex(idx.floor("1h"))

    hour_key = pd.Series(idx.floor("1h"), index=idx)
    form_hi = m5["high"].groupby(hour_key).cummax().to_numpy(float)
    form_lo = m5["low"].groupby(hour_key).cummin().to_numpy(float)
    prev_hc = H["prev_close"].to_numpy(float)
    tr_now = np.maximum(form_hi - form_lo, np.maximum(np.abs(form_hi - prev_hc),
                                                      np.abs(form_lo - prev_hc)))
    atr1h = (H["tr13"].to_numpy(float) + tr_now) / 14.0
    e20h = blend(H["e20"].to_numpy(float), A20)
    e50h = blend(H["e50"].to_numpy(float), A50)
    path_h = H["path19"].to_numpy(float) + np.abs(c - prev_hc)
    with np.errstate(invalid="ignore", divide="ignore"):
        er_h1 = np.where(path_h > 0, np.abs(c - H["close20"].to_numpy(float)) / path_h, 0.0)

    # ---- 15 minutes: EMAs with the forming bar, range of the 15 COMPLETED bars
    m15 = resample(m5, "15min")
    qc = m15["close"]
    Q = pd.DataFrame({
        "e20": qc.ewm(span=20, adjust=False).mean().shift(1),
        "e50": qc.ewm(span=50, adjust=False).mean().shift(1),
        "hi15": m15["high"].shift(1).rolling(15).max(),
        "lo15": m15["low"].shift(1).rolling(15).min(),
    }).reindex(idx.floor("15min"))
    e20m = blend(Q["e20"].to_numpy(float), A20)
    e50m = blend(Q["e50"].to_numpy(float), A50)
    hi15 = Q["hi15"].to_numpy(float)
    lo15 = Q["lo15"].to_numpy(float)

    # ---- 1 day: EMAs with today's partial bar; the filter uses COMPLETED days only
    d1 = resample(m5, "1D")
    dc = d1["close"]
    D = pd.DataFrame({
        "e20": dc.ewm(span=20, adjust=False).mean().shift(1),
        "e50": dc.ewm(span=50, adjust=False).mean().shift(1),
        "close1": dc.shift(1),
        "close21": dc.shift(21),
        "path20": dc.diff().abs().shift(1).rolling(20).sum(),
    }).reindex(idx.floor("1D"))
    e20d = blend(D["e20"].to_numpy(float), A20)
    e50d = blend(D["e50"].to_numpy(float), A50)
    daily_bull = (c > e50d) & (e20d > e50d)
    daily_bear = (c < e50d) & (e20d < e50d)
    path_d = D["path20"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        er_d = np.where(path_d > 0,
                        np.abs(D["close1"].to_numpy(float) - D["close21"].to_numpy(float)) / path_d, 0.0)

    ready = ~(np.isnan(e50h) | np.isnan(e50m) | np.isnan(e50d) | np.isnan(atr1h)
              | np.isnan(hi15) | np.isnan(lo15) | np.isnan(prev_c))
    longs = ready & (e20h > e50h) & (e20m > e50m) & ~daily_bear & (c > hi15) & (prev_c <= hi15)
    shorts = ready & (e20h < e50h) & (e20m < e50m) & ~daily_bull & (c < lo15) & (prev_c >= lo15)

    return {"idx": idx, "close": c, "high": m5["high"].to_numpy(float),
            "low": m5["low"].to_numpy(float), "open": m5["open"].to_numpy(float),
            "atr1h": atr1h, "er_h1": er_h1, "er_d": er_d,
            "long": longs, "short": shorts}


def signals(m5: pd.DataFrame) -> pd.DataFrame:
    """The candles at which the Indie rules fire, with the values the gates need."""
    f = features(m5)
    idx, c = f["idx"], f["close"]
    pos = np.flatnonzero(f["long"] | f["short"])
    buy = f["long"][pos]
    return pd.DataFrame({
        "pos": pos, "time": idx[pos], "minute": idx[pos].hour * 60 + idx[pos].minute,
        "direction": np.where(buy, "buy", "sell"), "entry": c[pos],
        "risk": ATR_MULT * f["atr1h"][pos], "er_h1": f["er_h1"][pos], "er_d": f["er_d"][pos],
    })


# --------------------------------------------------------------- simulation
def simulate(m5, sig, cadence, er_min, use_filter, er_d_min, max_hold, start, end, costs=None):
    """`costs` defaults to the gold cost model, so every result produced before
    markets_study.py existed is unchanged; that study passes each market its own."""
    from backtest import simulate_trade
    from engine import TradePlan
    from trade_accounting import load_costs

    costs = load_costs(COSTS) if costs is None else costs
    n = len(m5)
    trades, in_trade_until = [], None
    for r in sig.itertuples(index=False):
        if r.time < start or r.time >= end:
            continue
        if r.minute % cadence:
            continue
        if r.er_h1 < er_min:
            continue
        if use_filter and r.er_d < er_d_min:
            continue
        if in_trade_until is not None and r.time < in_trade_until:
            continue
        if not r.risk > 0 or r.pos + 1 >= n:
            continue
        sign = 1 if r.direction == "buy" else -1
        plan = TradePlan(r.direction, r.entry - 0.1, r.entry + 0.1, r.entry - sign * r.risk,
                         r.entry + sign * r.risk, r.entry + sign * RR2 * r.risk, 1.0, RR2)
        res = simulate_trade(m5, r.pos + 1, plan, costs=costs, exit_rule="breakeven",
                             max_hold_hours=max_hold)
        res.update(time=r.time, direction=r.direction, risk=round(float(r.risk), 3),
                   er_h1=round(float(r.er_h1), 3), er_d=round(float(r.er_d), 3))
        trades.append(res)
        in_trade_until = res["exit_time"]
    return pd.DataFrame(trades)


def prepare():
    from backtest import load_real_data
    m5 = load_real_data(DATA)
    return m5, signals(m5)


def run_variant(name):
    t0 = time.time()
    m5, sig = prepare()
    cadence, er_min, use_filter, er_d_min, hold = VARIANTS[name]
    tr = simulate(m5, sig, cadence, er_min, use_filter, er_d_min, hold, START, END)
    if not tr.empty:
        tr["variant"] = name
    print(f"  {name:<34} {len(tr):>6} trades  ({time.time()-t0:.0f}s)", flush=True)
    return name, tr


# --------------------------------------------------------------- reporting
def trading_days(m5, a, b):
    days = pd.Series(m5.index.normalize().unique())
    return int(((days >= a) & (days < b) & (days.dt.dayofweek < 5)).sum())


def stats(df, ndays, years):
    if df is None or df.empty:
        return {"n": 0, "per_day": 0.0, "win": 0.0, "R": 0.0, "R_year": 0.0, "PF": 0.0,
                "maxDD": 0.0, "p_no_edge": 100.0, "per_trade": 0.0}
    x = df.sort_values("time")["r_net"].to_numpy(float)
    gl = -x[x <= 0].sum()
    eq = np.cumsum(x)
    rng = np.random.default_rng(7)
    boots = rng.choice(x, size=(10000, len(x)), replace=True).mean(1)
    return {"n": len(x), "per_day": round(len(x) / ndays, 2), "win": round(float((x > 0).mean()) * 100, 1),
            "R": round(float(x.sum()), 2), "R_year": round(float(x.sum()) / years, 2),
            "per_trade": round(float(x.mean()), 4),
            "PF": round(float(x[x > 0].sum() / gl), 3) if gl > 0 else float("inf"),
            "maxDD": round(float((eq - np.maximum.accumulate(eq)).min()), 2),
            "p_no_edge": round(float(np.mean(boots <= 0)) * 100, 1)}


def check():
    """Reproduce the known live-strategy result before trusting anything else."""
    t0 = time.time()
    m5, sig = prepare()
    print(f"{len(m5):,} M5 bars {m5.index[0].date()} -> {m5.index[-1].date()}; "
          f"{len(sig):,} raw Indie signals at 5-minute cadence ({time.time()-t0:.0f}s)")
    cadence, er_min, use_filter, er_d_min, hold = VARIANTS[BASE]
    tr = simulate(m5, sig, cadence, er_min, use_filter, er_d_min, hold, *REPRO)
    ref_path = os.path.join(HERE, "filter_study_trades.csv")
    ref = pd.read_csv(ref_path, parse_dates=["time"])
    ref = ref[(ref["variant"] == "F1 + daily-trend filter")
              & (ref["time"] >= REPRO[0]) & (ref["time"] < REPRO[1])]
    mine, theirs = set(tr["time"]), set(ref["time"])
    match = len(mine & theirs) / max(len(theirs), 1) * 100
    print(f"\nREPRODUCTION CHECK  {REPRO[0].date()} -> {REPRO[1].date()}")
    print(f"  reference (filter_study_trades.csv): {len(ref):>5} trades, {ref['r_net'].sum():+.2f}R")
    print(f"  this study's engine:                 {len(tr):>5} trades, {tr['r_net'].sum():+.2f}R")
    print(f"  entry timestamps matching the reference: {match:.1f}%")
    ok = (abs(len(tr) - len(ref)) <= 0.05 * len(ref) and abs(tr["r_net"].sum() - ref["r_net"].sum()) <= 3.0
          and match >= 90)
    print(f"  -> {'PASS, the study may proceed' if ok else 'FAIL, the study is void'}")
    return ok


def run():
    t0 = time.time()
    if not check():
        print("\nreproduction check failed — not running the candidates")
        return
    m5, _ = prepare()
    nd = {"TRAIN": trading_days(m5, *TRAIN), "TEST": trading_days(m5, *TEST)}
    yrs = {"TRAIN": (TRAIN[1] - TRAIN[0]).days / 365.25, "TEST": (TEST[1] - TEST[0]).days / 365.25}
    del m5

    print(f"\nrunning {len(VARIANTS)} variants", flush=True)
    with ProcessPoolExecutor(max_workers=min(8, len(VARIANTS))) as ex:
        results = dict(ex.map(run_variant, list(VARIANTS)))
    allt = pd.concat([t for t in results.values() if not t.empty], ignore_index=True)
    allt.to_csv(os.path.join(HERE, "frequency_trades.csv"), index=False)

    table = {}
    print("\n" + "=" * 124)
    print("ONE-A-DAY STUDY — Indie entries, same management, net of spread/swap/gap fills; 5-minute XAU/USD")
    print("=" * 124)
    print(f"{'variant':<34}{'split':<7}{'trades':>7}{'/day':>6}{'win%':>7}{'net R':>9}{'R/year':>8}"
          f"{'R/trade':>9}{'PF':>7}{'maxDD':>8}{'P(no edge)':>12}")
    for name in VARIANTS:
        tr = results[name]
        table[name] = {}
        for split, (a, b) in (("TRAIN", TRAIN), ("TEST", TEST)):
            part = tr[(tr["time"] >= a) & (tr["time"] < b)] if not tr.empty else tr
            s = stats(part, nd[split], yrs[split])
            table[name][split] = s
            print(f"{name:<34}{split:<7}{s['n']:>7}{s['per_day']:>6}{s['win']:>7}{s['R']:>9}"
                  f"{s['R_year']:>8}{s['per_trade']:>9}{s['PF']:>7}{s['maxDD']:>8}{s['p_no_edge']:>11}%")

    print("\nnet R by year:")
    print(allt.pivot_table(index=pd.to_datetime(allt["time"]).dt.year, columns="variant",
                           values="r_net", aggfunc="sum").round(1).to_string())

    print("\n" + "-" * 124)
    print("SELECTION RULE (fixed in advance, PREREGISTRATION_ONE_PER_DAY.md)")
    base = table[BASE]
    qualified = []
    for name in VARIANTS:
        if name == BASE:
            continue
        s = table[name]["TRAIN"]
        checks = {">=0.8 trades/day": s["per_day"] >= 0.8, "net R > 0": s["R"] > 0, "PF >= 1.05": s["PF"] >= 1.05}
        ok = all(checks.values())
        print(f"  TRAIN {name:<34} {'qualifies' if ok else 'fails    '}  " +
              "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()))
        if ok:
            qualified.append(name)

    verdict = {"candidate": None, "confirmed": False}
    if qualified:
        cand = max(qualified, key=lambda n: table[n]["TRAIN"]["R"])
        s = table[cand]["TEST"]
        checks = {"net R > 0": s["R"] > 0, "PF >= 1.10": s["PF"] >= 1.10,
                  "P(no edge) <= 10%": s["p_no_edge"] <= 10, ">=0.8 trades/day": s["per_day"] >= 0.8,
                  "beats D0 R/year on TRAIN": table[cand]["TRAIN"]["R_year"] > base["TRAIN"]["R_year"],
                  "beats D0 R/year on TEST": s["R_year"] > base["TEST"]["R_year"]}
        verdict = {"candidate": cand, "confirmed": all(checks.values())}
        print(f"\n  best qualifying candidate: {cand}")
        for k, v in checks.items():
            print(f"    [{'x' if v else ' '}] {k}")
        print(f"  -> {'CONFIRMED' if verdict['confirmed'] else 'NOT CONFIRMED'}")
    else:
        print("\n  no candidate qualified on TRAIN")

    print("\nVERDICT:", f"adopt {verdict['candidate']}" if verdict["confirmed"]
          else "trading about once a day is NOT supported by this history; the live strategy stays as it is")
    with open(os.path.join(HERE, "frequency_decision.json"), "w") as f:
        json.dump({"verdict": verdict, "table": table}, f, indent=2, default=str)
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    {"check": check, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
