"""
Round 2 of the trap research (PREREGISTRATION_TRAPS.md, "ROUND 2 amendment"):
C05-ENGF, the failed reversal candle, tested ONCE as a single 4-market portfolio.

  python experiments/trap_round2.py search    # CW x exit variants, search window only
  python experiments/trap_round2.py holdout   # H-A (base exits) and H-B (V* exits), opened ONCE
  python experiments/trap_round2.py confirm   # confirmation markets, their holdout windows (reported)

Compute rule (the laptop shuts down under load): one market loaded at a time,
everything sequential in this one process. Results are written to
trap_round2.json after every market, so a crash loses at most one market.
"""
import gc
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import trap_candidates as tc
import trap_study as ts

OUT = os.path.join(HERE, "trap_round2.json")
TRADES = os.path.join(HERE, "trap_round2_trades.csv")
MARKETS4 = ["GOLD", "BTC", "NAS100", "US30"]
CONFIRM = {"GOLD": ["SILVER"], "NAS100": ["US500"], "US30": ["US500"],
           "BTC": ["BTC (Binance)", "ETH (Binance)"]}
LB = 12
CWS = (4, 8)
# name -> (tp1 in R, exit rule). tp2 = 2.5R and the 36h hold are shared.
EXITS = {"BASE": (1.0, "breakeven"), "V03": (0.3, "partial_tp1"),
         "V05": (0.5, "partial_tp1"), "V075": (0.75, "partial_tp1")}


def c05v(LB, CW, r1, rule, r2=2.5, hold=36.0):
    """tc.c05 with the exit made a parameter; c05v(LB, CW, 1.0, 'breakeven') == tc.c05(LB, CW)."""
    def build(m5, f):
        c, A = f["close"], f["atr1h"]
        h, parts = tc._c05_setups(f, LB, CW)
        hh, hl, p55 = h["high"].to_numpy(), h["low"].to_numpy(), h["pos55"].to_numpy(np.int64)
        P, BUY, STOP = [], [], []
        for E, X, short in parts:
            pos = p55[X]
            a, e = A[pos], c[pos]
            stop = np.maximum(hh[E] + 0.2 * a, e + 1.2 * a) if short else np.minimum(hl[E] - 0.2 * a, e - 1.2 * a)
            with np.errstate(invalid="ignore"):
                ok = np.abs(e - stop) <= 3.5 * a
            P.append(pos[ok]); BUY.append(np.full(ok.sum(), not short)); STOP.append(stop[ok])
        return tc._rr_emit(f, np.concatenate(P), np.concatenate(BUY), np.concatenate(STOP), r1, r2, hold, rule)
    return build


def builder(cw, ex):
    r1, rule = EXITS[ex]
    return c05v(LB, cw, r1, rule)


# ================================================================ bookkeeping
def load():
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save(d):
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, default=str)
    os.replace(tmp, OUT)


def add_trades(tr, **cols):
    if tr is None or not len(tr):
        return
    out = tr.reindex(columns=ts.TRADE_COLS).copy()
    for k, v in cols.items():
        out[k] = v
    out.to_csv(TRADES, mode="a", header=not os.path.exists(TRADES), index=False)


def trades(phase):
    if not os.path.exists(TRADES):
        return pd.DataFrame()
    t = pd.read_csv(TRADES, parse_dates=["time", "exit_time"])
    return t[t["phase"] == phase]


def drop_trades(phase, market):
    """A market that crashed half way is re-run whole: drop its partial trades."""
    if not os.path.exists(TRADES):
        return
    t = pd.read_csv(TRADES)
    bad = (t["phase"] == phase) & (t["market"] == market)
    if bad.any():
        t[~bad].to_csv(TRADES, index=False)


def pooled(x, rm=None):
    x = np.asarray(x, float)
    s = ts._core(x, None if rm is None else np.asarray(rm, float))
    s["t"] = round(float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))), 3) if len(x) > 2 and x.std() > 0 else 0.0
    return s


def stress(tr):
    """1.5x spread and swap. Spread enters once, swap is linear, so this is exact."""
    if not len(tr):
        return 0.0
    return round(float((tr["r_net"] - 0.5 * tr["spread_entry"] / tr["risk"] + 0.5 * tr["swap_r"]).sum()), 3)


# ================================================================ search
def cmd_search():
    d = load()
    d.setdefault("search", {})
    for m in MARKETS4:
        if all(f"{cw}|{ex}|{m}" in d["search"] for cw in CWS for ex in EXITS):
            print(f"[{m}] already done", flush=True)
            continue
        drop_trades("search", m)
        t0 = time.time()
        m5, f, costs, price_now = ts.load_market(m)
        w = ts.windows(m5)
        a, b = w["search"]
        days = pd.Series(m5.index.normalize().unique())
        print(f"[{m}] search {a.date()} -> {b.date()} ({time.time()-t0:.0f}s to load)", flush=True)
        for cw in CWS:
            for ex in EXITS:
                t1 = time.time()
                _, tr = ts.run_builder(m5, f, costs, price_now, builder(cw, ex), a, b, clip=True)
                tr = tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr
                s = ts.summarize(tr, a, b, days)
                key = f"{cw}|{ex}|{m}"
                d["search"][key] = {k: v for k, v in s.items() if k != "folds"}
                add_trades(tr, phase="search", key=key, cw=cw, exit=ex, market=m)
                ts.line("search", f"C05 LB12 CW{cw} {ex} {m}", s, time.time() - t1)
        # round-1 reproduction check: BASE must equal the registered C05 cells exactly
        r1 = ts.load_results().get("search", {})
        for cw in CWS:
            ref = r1.get(f"C05-ENGF[LB=12,CW={cw}]|{m}")
            mine = d["search"][f"{cw}|BASE|{m}"]
            if ref and (ref["n"] != mine["n"] or abs(ref["R"] - mine["R"]) > 1e-6):
                raise SystemExit(f"BASE CW{cw} {m} does not reproduce round 1: {mine['n']} / {mine['R']} "
                                 f"vs {ref['n']} / {ref['R']}")
        save(d)
        del m5, f
        gc.collect()
    select(d)


def select(d):
    t = trades("search")
    table = {}
    for cw in CWS:
        for ex in EXITS:
            g = t[(t["cw"] == cw) & (t["exit"] == ex)]
            s = pooled(g["r_net"], g["r_multiple"])
            s["markets_positive"] = int(sum(d["search"][f"{cw}|{ex}|{m}"]["R"] > 0 for m in MARKETS4))
            table[f"CW{cw} {ex}"] = s
    cw_star = max(CWS, key=lambda cw: table[f"CW{cw} BASE"]["t"])
    cand = [(table[f"CW{cw_star} {ex}"]["t"], ex) for ex in EXITS if ex != "BASE"
            and table[f"CW{cw_star} {ex}"]["R"] > 0]
    v_star = max(cand)[1] if cand else None
    d["pooled_search"] = table
    d["selection"] = {"CW_star": cw_star, "V_star": v_star,
                      "rule": "CW* = higher pooled search t (BASE); V* = highest pooled search t among "
                              "V03/V05/V075 at CW*, needs pooled R > 0"}
    save(d)
    print("\nPOOLED SEARCH (4 markets, net of costs)")
    for k, s in table.items():
        print(f"  {k:<12} n={s['n']:>5} win {s['win']:>5}% (p* {s.get('p_star')}%)  net {s['R']:>+8.2f}R  "
              f"{s['per_trade']:+.4f}R/tr  PF {s['PF']:<6} t {s['t']:>6}  markets+ {s['markets_positive']}/4")
    print(f"\nselected: CW* = {cw_star}, V* = {v_star}")


# ================================================================ holdout
def cmd_holdout():
    d = load()
    sel = d.get("selection")
    if not sel:
        raise SystemExit("run search first")
    if d.get("holdout_done"):
        print("the holdout was already opened; results:")
        report_holdout(d)
        return
    cw = sel["CW_star"]
    tests = {"H-A": "BASE"}
    if sel["V_star"]:
        tests["H-B"] = sel["V_star"]
    d.setdefault("holdout", {})
    for m in MARKETS4:
        if all(f"{h}|{m}" in d["holdout"] for h in tests):
            continue
        drop_trades("holdout", m)
        t0 = time.time()
        m5, f, costs, price_now = ts.load_market(m)
        w = ts.windows(m5)
        a, b = w["holdout"]
        days = pd.Series(m5.index.normalize().unique())
        print(f"[{m}] HOLDOUT {a.date()} -> {b.date()} ({time.time()-t0:.0f}s to load)", flush=True)
        gold_tr = {}
        for h, ex in tests.items():
            t1 = time.time()
            _, tr = ts.run_builder(m5, f, costs, price_now, builder(cw, ex), a, b)
            tr = tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr
            s = ts.summarize(tr, a, b, days, folds=4)
            s["stress_1p5_R"] = stress(tr)
            if tc.KIND.get(m) == "index":
                s["after_2022_R"] = round(float(tr.loc[tr["time"] >= ts.INDEX_REGIME, "r_net"].sum()), 3) if len(tr) else 0.0
            s["window"] = [str(a), str(b)]
            d["holdout"][f"{h}|{m}"] = s
            add_trades(tr, phase="holdout", key=f"{h}|{m}", cw=cw, exit=ex, market=m)
            ts.line("holdout", f"{h} C05 LB12 CW{cw} {ex} {m}", s, time.time() - t1)
            gold_tr[h] = tr
        save(d)
        del m5, f
        gc.collect()
        if m == "GOLD" and ts.have("GOLD (broker)"):
            m5, f, costs, price_now = ts.load_market("GOLD (broker)")
            ga, gb = ts.windows(m5, warmup_days=60)["all"]
            for h, ex in tests.items():
                _, tb = ts.run_builder(m5, f, costs, price_now, builder(cw, ex), ga, gb)
                dk = gold_tr[h]
                dk = dk[(dk["time"] >= ga) & (dk["time"] < gb)] if len(dk) else dk
                rb = float(tb["r_net"].sum()) if len(tb) else 0.0
                rd = float(dk["r_net"].sum()) if len(dk) else 0.0
                d["holdout"][f"{h}|GOLD"]["gold_broker_check"] = {
                    "from": str(ga.date()), "to": str(gb.date()), "broker_n": int(len(tb)),
                    "broker_R": round(rb, 3), "dukascopy_n": int(len(dk)), "dukascopy_R": round(rd, 3),
                    "sign_kept": (rb > 0) == (rd > 0)}
            save(d)
            del m5, f
            gc.collect()
    judge(d, tests)
    d["holdout_done"] = True
    d["holdout_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(d)
    report_holdout(d)


def judge(d, tests):
    t = trades("holdout")
    ps = d["pooled_search"]
    cw = d["selection"]["CW_star"]
    d["verdict"] = {}
    for h, ex in tests.items():
        g = t[t["key"].str.startswith(h + "|")]
        s = pooled(g["r_net"], g["r_multiple"])
        # P2: stationary bootstrap of the pooled daily R (one calendar over all 4 windows)
        cal = pd.date_range(g["time"].min().normalize(), g["time"].max().normalize(), freq="D")
        daily = g.assign(day=g["time"].dt.normalize()).groupby("day")["r_net"].sum().reindex(cal).fillna(0.0)
        p_adj, _, tstat = ts.spa_test(daily.to_frame(h), n_boot=2000, q=0.1)
        s["p_boot_daily"] = p_adj[h]
        s["p_boot_trades"] = ts.boot_p(g["r_net"].to_numpy(float))
        per_m = {m: d["holdout"][f"{h}|{m}"] for m in MARKETS4}
        quarters = [round(sum(per_m[m]["folds"][k]["R"] for m in MARKETS4 if len(per_m[m]["folds"]) > k), 3)
                    for k in range(4)]
        s["quarters_R"] = quarters
        s["stress_1p5_R"] = round(sum(per_m[m]["stress_1p5_R"] for m in MARKETS4), 3)
        srch = ps[f"CW{cw} {ex}"]["per_trade"]
        ck = {"P1 net R>0 & PF>=1.15": s["R"] > 0 and s["PF"] >= 1.15,
              "P2 bootstrap p<=0.025": s["p_boot_daily"] <= 0.025,
              "P3 >=3/4 markets R>0": sum(per_m[m]["R"] > 0 for m in MARKETS4) >= 3,
              "P4 1.5x costs R>0": s["stress_1p5_R"] > 0,
              "P5 >=3/4 quarters R>=0": sum(q >= 0 for q in quarters) >= 3,
              "P6 R/tr >= 50% of search": s["per_trade"] >= 0.5 * srch}
        s["search_per_trade"] = srch
        s["checks"] = ck
        s["PASS"] = all(ck.values())
        s["exit"] = ex
        d["verdict"][h] = s


def report_holdout(d):
    cw = d["selection"]["CW_star"]
    for h, s in d.get("verdict", {}).items():
        print("\n" + "=" * 100)
        print(f"{h}: C05 LB12 CW{cw} exits {s['exit']}  —  POOLED HOLDOUT (net of costs)")
        print(f"  n={s['n']}  win {s['win']}% (geometry p* {s.get('p_star')}%)  net {s['R']:+.2f}R  "
              f"{s['per_trade']:+.4f}R/tr (search {s['search_per_trade']:+.4f})  PF {s['PF']}  t {s['t']}")
        print(f"  bootstrap p daily {s['p_boot_daily']}  per-trade P(no edge) {s['p_boot_trades']}%  "
              f"quarters {s['quarters_R']}  1.5x costs {s['stress_1p5_R']:+.2f}R")
        for m in MARKETS4:
            x = d["holdout"][f"{h}|{m}"]
            extra = ""
            if "after_2022_R" in x:
                extra += f"  after-2022 {x['after_2022_R']:+.1f}R"
            if "gold_broker_check" in x:
                gb = x["gold_broker_check"]
                extra += f"  broker {gb['broker_R']:+.1f}R vs duka {gb['dukascopy_R']:+.1f}R"
            print(f"    {m:<7} {x['from']}->{x['to']}  n={x['n']:>4}  win {x['win']:>5}%  net {x['R']:>+7.2f}R  "
                  f"PF {x['PF']:<6} maxDD {x['maxDD']:+.1f}R  quarters {[q['R'] for q in x['folds']]}{extra}")
        print("  " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in s["checks"].items()))
        print(f"  ==> {'PASS' if s['PASS'] else 'FAIL'}")


# ================================================================ confirmation (reported)
def cmd_confirm():
    d = load()
    if not d.get("holdout_done"):
        raise SystemExit("run holdout first")
    cw = d["selection"]["CW_star"]
    tests = {h: v["exit"] for h, v in d["verdict"].items()}
    d.setdefault("confirm", {})
    for cm in sorted({c for v in CONFIRM.values() for c in v}):
        if all(f"{h}|{cm}" in d["confirm"] for h in tests):
            continue
        if not ts.have(cm):
            print(f"[{cm}] data missing")
            continue
        m5, f, costs, price_now = ts.load_market(cm)
        a, b = ts.windows(m5)["holdout"]
        days = pd.Series(m5.index.normalize().unique())
        for h, ex in tests.items():
            t1 = time.time()
            _, tr = ts.run_builder(m5, f, costs, price_now, builder(cw, ex), a, b)
            tr = tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr
            s = ts.summarize(tr, a, b, days, folds=4)
            s["stress_1p5_R"] = stress(tr)
            d["confirm"][f"{h}|{cm}"] = s
            add_trades(tr, phase="confirm", key=f"{h}|{cm}", cw=cw, exit=ex, market=cm)
            ts.line("confirm", f"{h} {ex} {cm}", s, time.time() - t1)
        save(d)
        del m5, f
        gc.collect()
    for h in tests:
        rs = {cm: d["confirm"][f"{h}|{cm}"]["R"] for cm in sorted({c for v in CONFIRM.values() for c in v})
              if f"{h}|{cm}" in d["confirm"]}
        print(f"{h}: confirmation markets net R {rs}")


if __name__ == "__main__":
    {"search": cmd_search, "holdout": cmd_holdout, "confirm": cmd_confirm,
     "select": lambda: select(load())}[sys.argv[1] if len(sys.argv) > 1 else "search"]()
