"""
Test harness for PREREGISTRATION_EDGES.md.

Every candidate is a builder returning deep_search_study.sig_frame rows (each
signal carries its own stop, targets, holding time and exit rule). Every market
is charged its own measured spread AND swap, both scaled with price (the lesson
of RESULTS_MASTER.md). Evaluation: the whole history after warm-up, five
chronological folds, bootstrap P(no edge), and the section's pass bar.

  python experiments/edge_study.py section1
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from frequency_study import features, stats
from best_per_market_study import daily_dir
from deep_search_study import BUILDERS as DEEP_BUILDERS, GOLD_BAR, simulate_family, WARMUP_DAYS, SPLIT
from markets_study import cost_model

# label -> (csv, broker cost key)
MARKETS = {
    "GOLD": ("xauusd_m5_dukascopy_23y.csv", "XAUUSD"),
    "BTC": ("btcusd_m5_dukascopy_23y.csv", "BTCUSD"),
    "NAS100": ("nas100_m5_dukascopy_23y.csv", "NAS100"),
    "ETH (Binance)": ("ethusd_m5_binance.csv", "ETHUSD"),
    "BTC (Binance)": ("btcusd_m5_binance.csv", "BTCUSD"),
    "US500": ("us500_m5_dukascopy_23y.csv", "US500"),
    "SILVER": ("xagusd_m5_dukascopy_23y.csv", "XAGUSD"),
    "GOLD (broker)": ("real_xauusd_5y.csv", "XAUUSD"),
}
RESULTS = os.path.join(HERE, "edge_decision.json")


def load_market(label):
    from backtest import load_real_data
    file, key = MARKETS[label]
    m5 = load_real_data(os.path.join(PROJ, file))
    costs = cost_model(key)
    price_now = float(m5["close"].tail(8640).median())
    m5 = m5.copy()
    if "spread" not in m5.columns:                  # the broker feed carries its real spread
        m5["spread"] = costs.spread_price * m5["close"] / price_now
    f = features(m5)
    f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
    return m5, f, costs, price_now


def evaluate(label, builder, window="all", folds=5):
    """window: 'all' (after warm-up), 'search' (the first 60% after warm-up) or
    'holdout' (the deep-search 40% window)."""
    m5, f, costs, price_now = load_market(label)
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=WARMUP_DAYS)
    end = last + pd.Timedelta(days=1)
    if window == "holdout":
        start = start + (last - start) * SPLIT
    elif window == "search":
        end = start + (last - start) * SPLIT
    tr = simulate_family(m5, builder(m5, f), costs, start, end, swap_scale_to=price_now)
    days = pd.Series(m5.index.normalize().unique())

    def span_stats(a, b, part):
        nd = int(((days >= a) & (days < b) & (days.dt.dayofweek < 5)).sum())
        return stats(part, max(nd, 1), max((b - a).days / 365.25, 0.1))

    overall = span_stats(start, end, tr)
    edges = [start + (end - start) * k / folds for k in range(folds + 1)]
    fold_stats = []
    for a, b in zip(edges[:-1], edges[1:]):
        part = tr[(tr["time"] >= a) & (tr["time"] < b)] if not tr.empty else tr
        s = span_stats(a, b, part)
        s["from"], s["to"] = str(a.date()), str(b.date())
        fold_stats.append(s)
    return {"market": label, "window": window, "from": str(start.date()), "to": str(last.date()),
            "overall": overall, "folds": fold_stats,
            "positive_folds": sum(1 for s in fold_stats if s["R"] > 0)}, tr


def gold_grade(s):
    checks = {"PF>=1.25": s["PF"] >= GOLD_BAR["pf"], "R/year>=4": s["R_year"] >= GOLD_BAR["r_year"],
              "R/trade>=0.08": s["per_trade"] >= GOLD_BAR["r_trade"],
              "P(no edge)<=5%": s["p_no_edge"] <= GOLD_BAR["p_no_edge"]}
    return all(checks.values()), checks


def show(res, name):
    o = res["overall"]
    print(f"\n{name} on {res['market']} ({res['window']}, {res['from']} -> {res['to']})")
    print(f"  {o['n']} trades, {o['per_day']}/day, win {o['win']}%, net {o['R']:+.2f}R, {o['R_year']:+.2f}R/yr, "
          f"{o['per_trade']:+.4f}R/trade, PF {o['PF']}, maxDD {o['maxDD']}, P(no edge) {o['p_no_edge']}%")
    print("  folds: " + " | ".join(f"{s['from'][:4]}-{s['to'][:4]} {s['R']:+.1f}R PF {s['PF']}" for s in res["folds"]))


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            return json.load(fh)
    return {}


def save_results(d):
    with open(RESULTS, "w") as fh:
        json.dump(d, fh, indent=2, default=str)


def cmd_section1():
    d = load_results()
    if "section1" in d:
        print("section 1 already run:")
        print(json.dumps(d["section1"], indent=2)[:3000])
        return
    sq3 = DEEP_BUILDERS["SQ3"]
    out = {}
    r, _ = evaluate("ETH (Binance)", sq3, "all")
    show(r, "1a SQ3")
    ok, checks = gold_grade(r["overall"])
    print("  " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()) +
          f"  => {'PASSED' if ok else 'FAILED'}")
    out["1a_eth"] = {**r, "passed": ok}
    for window in ("holdout", "all"):
        r, _ = evaluate("BTC (Binance)", sq3, window)
        show(r, f"1b SQ3 feed check")
        ok, checks = gold_grade(r["overall"])
        print("  " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()))
        out[f"1b_btc_binance_{window}"] = {**r, "gold_grade": ok}
    d["section1"] = out
    d["section1_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(d)


def primary_pass(r):
    o = r["overall"]
    checks = {"net R>0": o["R"] > 0, "PF>=1.10": o["PF"] >= 1.10, ">=4/5 folds": r["positive_folds"] >= 4,
              "P<=1%": o["p_no_edge"] <= 1.0, ">=100 trades": o["n"] >= 100}
    return all(checks.values()), checks


def run_job(job):
    from edge_candidates import builder_for
    cand, market = job
    t0 = time.time()
    r, tr = evaluate(market, builder_for(cand, market), "all")
    o = r["overall"]
    print(f"  {cand} {market:<14} {o['n']:>5} trades  net {o['R']:+8.2f}R  PF {o['PF']:<6} "
          f"folds+ {r['positive_folds']}/5  ({time.time()-t0:.0f}s)", flush=True)
    tr_out = tr[["time", "direction", "r_net", "outcome", "exit_time"]].copy() if not tr.empty else tr
    return cand, market, r, tr_out


def cmd_section3():
    from concurrent.futures import ProcessPoolExecutor
    from edge_candidates import CANDIDATES, CONFIRM
    d = load_results()
    if "section3" in d:
        print("section 3 already run")
        return
    jobs = [(cand, m) for cand, t in CANDIDATES.items() for m in t]
    print(f"SECTION 3 — {len(jobs)} primary tests\n", flush=True)
    with ProcessPoolExecutor(max_workers=5) as ex:
        out = list(ex.map(run_job, jobs))
    res, trades = {}, []
    for cand, market, r, tr in out:
        ok, checks = primary_pass(r)
        show(r, cand)
        print("  " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()) +
              f"  => {'PASS' if ok else 'fail'}")
        res[f"{cand}|{market}"] = {**r, "primary_pass": ok}
        if len(tr):
            trades.append(tr.assign(candidate=cand, market=market))
    passers = [tuple(k.split("|")) for k, v in res.items() if v["primary_pass"]]
    conf_jobs = [(cand, cm) for cand, m in passers for cm in CONFIRM[m]]
    if conf_jobs:
        print(f"\nCONFIRMATION — {len(conf_jobs)} tests on data never touched\n", flush=True)
        with ProcessPoolExecutor(max_workers=4) as ex:
            cout = list(ex.map(run_job, conf_jobs))
        for cand, market, r, tr in cout:
            o = r["overall"]
            ok = o["R"] > 0 and o["PF"] >= 1.05
            show(r, cand + " (confirmation)")
            print(f"  => {'CONFIRMED' if ok else 'not confirmed'}")
            res[f"{cand}|{market}|confirm"] = {**r, "confirmed": ok}
            if len(tr):
                trades.append(tr.assign(candidate=cand, market=market + " (confirm)"))
    else:
        print("\nno candidate passed the primary bar — nothing to confirm")
    if trades:
        pd.concat(trades, ignore_index=True).to_csv(os.path.join(HERE, "edge_trades.csv"), index=False)
    d["section3"] = res
    d["section3_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(d)


if __name__ == "__main__":
    {"section1": cmd_section1, "section3": cmd_section3}[sys.argv[1]]()
