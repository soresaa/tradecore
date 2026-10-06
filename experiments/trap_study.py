"""
Runs the pre-registered trap candidates (trap_candidates.py, TRAP_RESEARCH_BRIEF.md).

  python experiments/trap_study.py smoke     # signal counts on the last 60 days, no backtest
  python experiments/trap_study.py search    # every cell x market on the first 60% after warm-up
  python experiments/trap_study.py holdout   # search survivors only, last 40%, opened ONCE
  python experiments/trap_study.py confirm   # holdout survivors on their confirmation markets

Compute rule (the user's laptop crashes under load): ONE market loaded at a
time, every test run sequentially in this single process, no process pool.
Every finished test is written to trap_decision.json (and its trades appended
to trap_trades.csv) immediately, so a crash loses at most one test and a
re-run resumes where it stopped.
"""
import dataclasses
import gc
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

from frequency_study import features
from best_per_market_study import daily_dir
from deep_search_study import simulate_family, WARMUP_DAYS, SPLIT
from markets_study import cost_model
import edge_study
import trap_candidates as tc

# label -> (csv, broker cost key). US30 costs were measured from the broker on
# 2026-09-26 (US30m: spread 1.90 pts, swap -10.62/night long), so it has its own key.
MARKETS = dict(edge_study.MARKETS)
MARKETS["US30"] = ("us30_m5_dukascopy_23y.csv", "US30")

RESULTS = os.path.join(HERE, "trap_decision.json")
TRADES = os.path.join(HERE, "trap_trades.csv")
C02_FROM = pd.Timestamp("2019-09-01")          # Binance perpetuals era
GOLD_SPLIT = pd.Timestamp("2015-04-01")        # S11: both sides of 2015-03
INDEX_REGIME = pd.Timestamp("2022-01-01")      # H7: daily 0DTE
BTC_REGIME = pd.Timestamp("2026-05-25")        # H7: CME 24/7 (reported only)
MAX_PROMOTED = 5
TRADE_COLS = ["time", "exit_time", "direction", "outcome", "r_multiple", "r_net", "tp1_hit", "risk",
              "spread_entry", "swap_r", "tag", "baseline"]


# ================================================================ data
def have(label):
    return os.path.exists(os.path.join(PROJ, MARKETS[label][0]))


def tail_close_median(path, n=8640):
    """Median close of the last n rows, read from the end of the file only."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 140 * (n + 20)))
        lines = fh.read().decode("utf-8", "ignore").splitlines()[1:]
    closes = []
    for ln in lines[-n:]:
        parts = ln.split(",")
        try:
            closes.append(float(parts[5]))
        except (IndexError, ValueError):
            pass
    return float(np.median(closes))


def market_costs(label, price_now):
    return cost_model(MARKETS[label][1])


def load_market(label, spread_mult=1.0, swap_mult=1.0):
    from backtest import load_real_data
    m5 = load_real_data(os.path.join(PROJ, MARKETS[label][0]))
    price_now = float(m5["close"].tail(8640).median())
    costs = market_costs(label, price_now)
    if swap_mult != 1.0:
        costs = dataclasses.replace(costs, swap_long_per_night=costs.swap_long_per_night * swap_mult,
                                    swap_short_per_night=costs.swap_short_per_night * swap_mult)
    m5 = m5.copy()
    if "spread" not in m5.columns:               # the broker feed carries its real spread
        m5["spread"] = costs.spread_price * m5["close"] / price_now
    m5["spread"] = m5["spread"] * spread_mult
    f = features(m5)
    f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
    return m5, f, costs, price_now


def windows(m5, warmup_days=WARMUP_DAYS):
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=warmup_days)
    cut = start + (last - start) * SPLIT
    end = last + pd.Timedelta(days=1)
    return {"search": (start, cut), "holdout": (cut, end), "all": (start, end)}


def cand_window(cand, w):
    a, b = w
    if cand == "C02-FUNDFADE-BTC":
        a = max(a, C02_FROM)
    return a, b


def check_stamps(f, label):
    """G1: bars must be stamped at their OPEN. For an index CFD the cash-open
    range jump must sit on the bar whose NY open is 09:30 (13:30 UTC in EDT,
    14:30 UTC in EST); with close stamps it would sit on the 09:35 bar."""
    mins, _, dow = tc.ny(f)
    rng = f["high"] - f["low"]
    wk = dow < 5

    def mean_range(m):
        sel = wk & (mins == m)
        return float(np.nanmean(rng[sel])) if sel.any() else float("nan")
    r25, r30, r35 = mean_range(565), mean_range(570), mean_range(575)
    ok = np.isfinite(r30) and r30 >= r35 and (not np.isfinite(r25) or r30 >= 1.3 * r25)
    return {"market": label, "range_0925": round(r25, 3), "range_0930": round(r30, 3),
            "range_0935": round(r35, 3), "open_stamped": bool(ok)}


# ================================================================ statistics
def boot_p(x, n_boot=5000, seed=7):
    """Bootstrap P(mean <= 0) in %, chunked so memory stays small."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return 100.0
    rng = np.random.default_rng(seed)
    chunk = max(1, 2_000_000 // len(x))
    le = 0
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        le += int((rng.choice(x, size=(k, len(x)), replace=True).mean(1) <= 0).sum())
        done += k
    return round(100.0 * le / n_boot, 2)


def _core(x, rmult=None):
    n = len(x)
    if n == 0:
        return {"n": 0, "R": 0.0, "per_trade": 0.0, "win": 0.0, "PF": 0.0}
    wins = x > 0
    gl = -x[~wins].sum()
    out = {"n": int(n), "R": round(float(x.sum()), 3), "per_trade": round(float(x.mean()), 4),
           "win": round(100.0 * float(wins.mean()), 2),
           "PF": round(float(x[wins].sum() / gl), 3) if gl > 0 else float("inf")}
    if rmult is not None:
        aw = x[wins].mean() if wins.any() else 0.0
        al = abs(x[~wins].mean()) if (~wins).any() else float("nan")
        B = aw / al if al and np.isfinite(al) and al > 0 else float("inf")
        mc = float(np.mean(rmult - x))
        out["B"] = round(float(B), 3)
        out["p_star"] = round(100.0 * (1 + mc) / (1 + B), 2) if np.isfinite(B) else 0.0
        out["margin"] = round(out["win"] - out["p_star"], 2)
    return out


def summarize(tr, a, b, m5_days, folds=5, boot=True):
    """Everything the pass bars and the win-rate report need for one window."""
    if tr is not None and len(tr):
        tr = tr[(tr["time"] >= a) & (tr["time"] < b)].sort_values("time")
    nd = int(((m5_days >= a) & (m5_days < b) & (m5_days.dt.dayofweek < 5)).sum())
    yrs = max((b - a).days / 365.25, 0.1)
    if tr is None or not len(tr):
        return {"n": 0, "R": 0.0, "per_trade": 0.0, "win": 0.0, "PF": 0.0, "R_year": 0.0,
                "per_day": 0.0, "maxDD": 0.0, "R_over_DD": 0.0, "p_no_edge": 100.0,
                "positive_folds": 0, "worst_fold_R": 0.0, "last_third_R": 0.0, "folds": [],
                "from": str(a.date()), "to": str(b.date())}
    x = tr["r_net"].to_numpy(float)
    rm = tr["r_multiple"].to_numpy(float)
    s = _core(x, rm)
    cost = rm - x
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    dd = float((eq - peak).min())
    wins = x > 0
    streak = best = 0
    for w in wins:
        streak = 0 if w else streak + 1
        best = max(best, streak)
    k5 = max(1, int(math.ceil(0.05 * len(x))))
    s.update({
        "R_year": round(s["R"] / yrs, 3), "per_day": round(len(x) / max(nd, 1), 3),
        "maxDD": round(dd, 3), "R_over_DD": round(s["R"] / abs(dd), 3) if dd < 0 else float("inf"),
        "p_no_edge": boot_p(x) if boot else None,
        "mean_cost": round(float(cost.mean()), 4), "median_cost": round(float(np.median(cost)), 4),
        "worst": round(float(x.min()), 3), "worst5_mean": round(float(np.sort(x)[:k5].mean()), 3),
        "median_win": round(float(np.median(x[wins])), 3) if wins.any() else 0.0,
        "skew": round(float(pd.Series(x).skew()), 3) if len(x) > 2 else 0.0,
        "longest_losing_streak": int(best),
        "target_hit": round(100.0 * float((tr["outcome"] == "TP2").mean()), 2),
        "tp1_hit": round(100.0 * float(tr["tp1_hit"].astype(bool).mean()), 2),
        "from": str(a.date()), "to": str(b.date()),
    })
    if "baseline" in tr.columns and tr["baseline"].notna().any():
        s["baseline_hit"] = round(100.0 * float(pd.to_numeric(tr["baseline"]).mean()), 2)
        s["hit_minus_baseline"] = round(s["target_hit"] - s["baseline_hit"], 2)
    edges = [a + (b - a) * k / folds for k in range(folds + 1)]
    fl = []
    for fa, fb in zip(edges[:-1], edges[1:]):
        part = tr[(tr["time"] >= fa) & (tr["time"] < fb)]
        fs = _core(part["r_net"].to_numpy(float), part["r_multiple"].to_numpy(float) if len(part) else None)
        fs.update({"from": str(fa.date()), "to": str(fb.date())})
        fl.append(fs)
    s["folds"] = fl
    s["positive_folds"] = sum(1 for q in fl if q["R"] > 0)
    s["worst_fold_R"] = min(q["R"] for q in fl)
    t3 = a + (b - a) * 2 / 3
    s["last_third_R"] = round(float(tr.loc[tr["time"] >= t3, "r_net"].sum()), 3)
    if "tag" in tr.columns and tr["tag"].notna().any():
        s["by_tag"] = {str(k): _core(g["r_net"].to_numpy(float), g["r_multiple"].to_numpy(float))
                       for k, g in tr.groupby("tag")}
    return s


def split_R(tr, a, cut, b):
    if tr is None or not len(tr):
        return 0.0, 0.0
    x1 = tr[(tr["time"] >= a) & (tr["time"] < cut)]["r_net"].sum()
    x2 = tr[(tr["time"] >= cut) & (tr["time"] < b)]["r_net"].sum()
    return round(float(x1), 3), round(float(x2), 3)


# ================================================================ bookkeeping
def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_results(d):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, default=str)
    os.replace(tmp, RESULTS)


def append_trades(tr, **cols):
    if tr is None or not len(tr):
        return
    out = tr.reindex(columns=TRADE_COLS).copy()
    for k, v in cols.items():
        out[k] = v
    out.to_csv(TRADES, mode="a", header=not os.path.exists(TRADES), index=False)


def read_trades(phase=None):
    if not os.path.exists(TRADES):
        return pd.DataFrame()
    t = pd.read_csv(TRADES, parse_dates=["time", "exit_time"])
    return t if phase is None else t[t["phase"] == phase]


def purge_trades(keys, phase):
    """Drop trades of tests that were written but not recorded (a crash between
    the two writes), so a re-run cannot double them."""
    if not keys or not os.path.exists(TRADES):
        return
    t = pd.read_csv(TRADES)
    bad = (t["phase"] == phase) & t["key"].isin(keys)
    if bad.any():
        t[~bad].to_csv(TRADES, index=False)


def run_builder(m5, f, costs, price_now, builder, a, b, clip=False):
    """clip=True: trades are simulated only on bars before `b`, so a search trade
    that is still open at the cut is closed there instead of using holdout bars."""
    sig = builder(m5, f)
    sim = m5.iloc[:int(m5.index.searchsorted(b))] if clip else m5
    tr = simulate_family(sim, sig, costs, a, b, swap_scale_to=price_now)
    if len(tr):
        extra = [c for c in ("tag", "baseline") if c in sig.columns]
        if extra:
            tr = tr.merge(sig[["time"] + extra], on="time", how="left")
    return sig, tr


def line(phase, key, s, secs):
    print(f"  {phase:<8}{key:<72} n={s['n']:>6}  win {s['win']:>5}%  net {s['R']:>+9.2f}R  "
          f"{s['per_trade']:>+.3f}R/tr  PF {s['PF']:<6} folds+ {s.get('positive_folds', 0)}/5  ({secs:.0f}s)",
          flush=True)


# ================================================================ evaluate (edge_study style)
def evaluate(label, builder, window="search", folds=5):
    """edge_study.evaluate with the 'search' window: [start, start + (last-start)*SPLIT)."""
    m5, f, costs, price_now = load_market(label)
    a, b = windows(m5)[window]
    _, tr = run_builder(m5, f, costs, price_now, builder, a, b)
    days = pd.Series(m5.index.normalize().unique())
    return summarize(tr, a, b, days, folds), tr


# ================================================================ search
def search_jobs():
    """market -> ordered list of (key, kind, id). kind: cell | control | g9."""
    jobs = {}
    for cid, meta in tc.CELLS.items():
        for m in meta["markets"]:
            jobs.setdefault(m, []).append((f"{cid}|{m}", "cell", cid))
            for ctl in meta["controls"]:
                k = f"{ctl}|{m}"
                if all(k != j[0] for j in jobs[m]):
                    jobs[m].append((k, "control", ctl))
    for cand, fac in tc.LEVELS.items():
        for m in tc.PRIMARY[cand]:
            jobs.setdefault(m, []).append((f"G9:{cand}|{m}", "g9", cand))
    return jobs


def cmd_search():
    d = load_results()
    d.setdefault("search", {})
    d.setdefault("g9", {})
    d.setdefault("stamps", {})
    d.setdefault("pre_mechanism", {})
    jobs = search_jobs()
    total = sum(len(v) for v in jobs.values())
    print(f"SEARCH: {total} tests ({len(tc.CELLS)} cells + controls + G9) on the first "
          f"{int(SPLIT * 100)}% after a {WARMUP_DAYS}-day warm-up; sequential, one market at a time\n", flush=True)
    for m, lst in jobs.items():
        if not have(m):
            print(f"[{m}] data file {MARKETS[m][0]} not found yet - skipped", flush=True)
            continue
        todo = [j for j in lst if (j[0] not in d["g9"] if j[1] == "g9" else j[0] not in d["search"])]
        if not todo:
            print(f"[{m}] all {len(lst)} tests already recorded", flush=True)
            continue
        purge_trades([j[0] for j in todo if j[1] != "g9"], "search")
        t0 = time.time()
        m5, f, costs, price_now = load_market(m)
        w = windows(m5)
        days = pd.Series(m5.index.normalize().unique())
        print(f"[{m}] {len(m5):,} bars {m5.index[0].date()} -> {m5.index[-1].date()}, search "
              f"{w['search'][0].date()} -> {w['search'][1].date()}, costs {costs.source} ({time.time()-t0:.0f}s)",
              flush=True)
        if tc.KIND.get(m) == "index" and m not in d["stamps"]:
            st = check_stamps(f, m)
            d["stamps"][m] = st
            save_results(d)
            print(f"  G1 stamp check: {st}", flush=True)
            if not st["open_stamped"]:
                raise SystemExit(f"G1 FAILED on {m}: the cash-open jump is not on the 09:30 NY bar. "
                                 "Bars may be close-stamped: shift every clock rule by -5 min before running.")
        for key, kind, ident in todo:
            t1 = time.time()
            if kind == "g9":
                up, dn = tc.LEVELS[ident](m)(m5, f)
                mask = np.asarray((m5.index >= w["search"][0]) & (m5.index < w["search"][1]))
                g = tc.level_placebo(f, up, dn, mask=mask)
                d["g9"][key] = g
                save_results(d)
                print(f"  g9      {key:<72} events {g['n']:>6}  back-through {g['event_pct']}% vs "
                      f"placebo {g['placebo_pct']}%  diff {g['diff_pts']:+} pts  ({time.time()-t1:.0f}s)",
                      flush=True)
                continue
            meta = tc.CELLS.get(ident) or tc.CONTROLS[ident]
            cand = meta["candidate"]
            a, b = cand_window(cand, w["search"])
            builder = tc.builder_for(ident, m)
            sig, tr = run_builder(m5, f, costs, price_now, builder, w["search"][0], b, clip=True)
            s = summarize(tr, a, b, days)
            s.update({"cell": ident, "candidate": cand, "market": m, "kind": kind,
                      "control_kind": meta.get("kind"), "signals_total": int(len(sig))})
            if m == "GOLD":
                s["gold_split_R"] = split_R(tr, a, GOLD_SPLIT, b)
            if cand == "C02-FUNDFADE-BTC" and w["search"][0] < C02_FROM:
                pre = summarize(tr, w["search"][0], C02_FROM, days, boot=False)
                d["pre_mechanism"][key] = {k: pre[k] for k in ("n", "R", "per_trade", "win", "PF")}
            trs = tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr
            append_trades(trs, phase="search", key=key, cell=ident, candidate=cand, kind=kind, market=m)
            d["search"][key] = s
            save_results(d)
            line(kind, key, s, time.time() - t1)
        del m5, f
        gc.collect()
    d = judge_search(d)
    d = confirm_search(d)
    save_results(d)
    report_search(d)


# ---------------------------------------------------------------- judging
def _S(d, key):
    return d["search"].get(key)


def cell_checks(cid, m, s):
    cand = tc.CELLS[cid]["candidate"]
    worst = s.get("worst", 0.0)
    ck = {
        "S1 netR>0 & PF>=1.15": s["R"] > 0 and s["PF"] >= 1.15,
        "S2 >=4/5 folds": s["positive_folds"] >= 4,
        "S3 trades": s["n"] >= 150 or (cand in tc.LOW_FREQ and s["n"] >= 100
                                       and (s.get("p_no_edge") or 100) <= 2.0),
        "S4 R/tr>=0.05 & cost<=0.10": s["per_trade"] >= 0.05 and s.get("median_cost", 9) <= 0.10,
        "S6 last third>0": s["last_third_R"] > 0,
        "S7 R/DD>=1.5": s["R_over_DD"] >= 1.5,
        "S8 tail": not (s["win"] > 75 and worst < -10 * s.get("median_win", 0)
                        and s["R"] < 3 * abs(worst)),
    }
    if m == "GOLD":
        r1, r2 = s.get("gold_split_R", (0, 0))
        ck["S11 gold split"] = r1 > 0 and r2 > 0
    return ck


def s9_check(d, cid, m, s, trades):
    """The candidate beats its declared control / placebo (search window)."""
    meta = tc.CELLS[cid]
    cand = meta["candidate"]
    out = {}
    for ctl in meta["controls"]:
        cm = tc.CONTROLS[ctl]
        if cm["kind"] in ("benchmark", "diagnostic"):
            continue
        cs = _S(d, f"{ctl}|{m}")
        if cs is None:                              # mandatory control missing -> cannot pass
            out[f"S9 vs {ctl} (missing)"] = False
            continue
        gap = s["per_trade"] - cs["per_trade"]
        out[f"S9 vs {ctl}"] = gap >= 0.05
        if cm["kind"] == "placebo_c06":
            out[f"S9 tp1-hit vs {ctl}"] = s.get("tp1_hit", 0) - cs.get("tp1_hit", 0) >= 5.0
    if cand == "C03-GAPHALF":
        out["S9 hit - baseline >= 3pts"] = s.get("hit_minus_baseline", -99) >= 3.0
    if cand == "C04-GAPFILL" and len(trades):
        t4 = trades[trades["key"] == f"{cid}|{m}"]
        t3 = trades[(trades["candidate"] == "C03-GAPHALF") & (trades["market"] == m)
                    & (trades["kind"] == "cell")]
        days4 = set(t4["time"].dt.normalize())
        same = t3[t3["time"].dt.normalize().isin(days4)]
        base = same if len(same) >= 20 else t3
        ref = float(base["r_net"].mean()) if len(base) else 0.0
        out["S9 vs C03 same days"] = s["per_trade"] - ref >= 0.05
    if cand == "C13-EXH3":
        ok = True
        for R in (1.0, 2.0):
            n3 = _S(d, f"C13-EXH3[N=3,R={R}]|{m}")
            n2 = _S(d, f"C13-EXH3[N=2,R={R}]|{m}")
            n1 = _S(d, f"C13-EXH3:CONTROL-N1[R={R}]|{m}")
            if not (n3 and n2 and n1 and n3["per_trade"] > n1["per_trade"] and n3["per_trade"] > n2["per_trade"]):
                ok = False
        out["S9 C13 dose-response (N=3 > N=2, N=1)"] = ok
    if cand in tc.LEVELS:
        g = d["g9"].get(f"G9:{cand}|{m}")
        out["G9 level-placebo >= +5pts"] = bool(g and np.isfinite(g.get("diff_pts", np.nan))
                                               and g["diff_pts"] >= 5.0)
    return out


def spa_test(series, n_boot=1000, q=0.1, seed=11):
    """Hansen SPA / White RC over all cells: stationary bootstrap (mean block
    1/q days) of the daily net-R series on one calendar. Returns per-cell
    single-step adjusted p-values and the SPA p-value of the best cell."""
    keys = list(series.columns)
    X = series.to_numpy(float)
    T, K = X.shape
    mu = X.mean(0)
    rng = np.random.default_rng(seed)
    cs = np.vstack([np.zeros((1, K)), np.cumsum(np.vstack([X, X]), 0)])
    bm = np.empty((n_boot, K))
    nblk = int(T * q * 2) + 50
    for b in range(n_boot):
        L = rng.geometric(q, size=nblk)
        while L.sum() < T:
            L = np.concatenate([L, rng.geometric(q, size=nblk)])
        cum = np.cumsum(L)
        m = int(np.searchsorted(cum, T)) + 1
        L = L[:m].copy()
        L[-1] -= cum[m - 1] - T
        st = rng.integers(0, T, size=m)
        bm[b] = (cs[st + L] - cs[st]).sum(0) / T
    omega = np.sqrt(T) * bm.std(0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.where(omega > 0, np.sqrt(T) * mu / omega, -np.inf)
        thr = -np.sqrt(2 * np.log(np.log(T)))
        g = np.where(stat >= thr, mu, 0.0)
        Z = np.where(omega > 0, np.sqrt(T) * (bm - g) / omega, -np.inf)
    mx = Z.max(1)
    p = {k: round(float((mx >= s).mean()), 4) for k, s in zip(keys, stat)}
    p_spa = float((np.maximum(mx, 0) >= max(float(np.max(stat)), 0.0)).mean())
    return p, p_spa, {k: round(float(s), 3) if np.isfinite(s) else None for k, s in zip(keys, stat)}


def judge_search(d):
    trades = read_trades("search")
    cells_done = {k: v for k, v in d["search"].items() if v.get("kind") == "cell"}
    # ---- S12 over ALL cells tried (C08 legs and C03/C04 included, one family)
    spa_p, p_best = {}, None
    if cells_done and len(trades):
        spans = [(pd.Timestamp(v["from"]), pd.Timestamp(v["to"])) for v in cells_done.values()]
        cal = pd.date_range(min(a for a, _ in spans), max(b for _, b in spans), freq="D")
        tt = trades[trades["kind"] == "cell"]
        daily = (tt.assign(day=tt["time"].dt.normalize()).groupby(["key", "day"])["r_net"].sum()
                 .unstack("key").reindex(index=cal, columns=list(cells_done)).fillna(0.0))
        print(f"\nS12 reality check: {daily.shape[1]} cells x {daily.shape[0]} days, 1000 stationary-bootstrap "
              "resamples ...", flush=True)
        spa_p, p_best, tstat = spa_test(daily)
        d["spa"] = {"p_adjusted": spa_p, "p_spa_best": p_best, "t_stat": tstat, "cells": daily.shape[1]}
    verdict = {}
    by_cm = {}
    for key, s in cells_done.items():
        cid, m = key.split("|")
        by_cm.setdefault((tc.CELLS[cid]["candidate"], m), []).append((cid, s))
    for (cand, m), lst in by_cm.items():
        checks = {}
        for cid, s in lst:
            ck = cell_checks(cid, m, s)
            ck.update(s9_check(d, cid, m, s, trades))
            ck["S12 SPA p<0.05"] = spa_p.get(f"{cid}|{m}", 1.0) < 0.05
            checks[cid] = ck
        grid_pos = sum(1 for _, s in lst if s["R"] > 0)
        groups = [lst]
        if cand == "C08-SMT1":                       # two hypotheses, one carried cell per leg
            groups = [[x for x in lst if tc.CELLS[x[0]]["params"]["leg"] == leg] for leg in ("A", "B")]
        for grp in groups:
            s14 = [(cid, s) for cid, s in grp
                   if all(v for k, v in checks[cid].items() if k.split()[0] in ("S1", "S2", "S3", "S4"))]
            carried = max(s14, key=lambda t: t[1]["worst_fold_R"])[0] if s14 else None
            vk = f"{cand}|{m}" + (f"|leg{tc.CELLS[grp[0][0]]['params']['leg']}" if cand == "C08-SMT1" else "")
            passed = bool(carried) and grid_pos >= 3 and all(checks[carried].values())
            verdict[vk] = {"candidate": cand, "market": m, "carried_cell": carried,
                           "S5 grid >=3/4 cells R>0": grid_pos >= 3, "cell_checks": checks,
                           "spa_p": spa_p.get(f"{carried}|{m}") if carried else None,
                           "pass_S1_S9_S11_S12": passed}
    d["verdict_search"] = verdict
    return d


def confirm_search(d):
    """S10: the frozen carried cell on its confirmation market(s), search window."""
    d.setdefault("confirm_search", {})
    todo = [(v["carried_cell"], v["market"]) for v in d.get("verdict_search", {}).values()
            if v["pass_S1_S9_S11_S12"]]
    need = {}
    for cid, m in todo:
        for cm in tc.CONFIRM[tc.CELLS[cid]["candidate"]].get(m, []):
            need.setdefault(cm, []).append(cid)
    for cm, cids in need.items():
        if not have(cm):
            print(f"[{cm}] confirmation file missing - S10 cannot pass for {cids}")
            continue
        left = [c for c in cids if f"{c}|{cm}" not in d["confirm_search"]]
        if not left:
            continue
        purge_trades([f"{c}|{cm}" for c in left], "confirm_search")
        m5, f, costs, price_now = load_market(cm)
        w = windows(m5)
        days = pd.Series(m5.index.normalize().unique())
        for cid in left:
            t1 = time.time()
            cand = tc.CELLS[cid]["candidate"]
            a, b = cand_window(cand, w["search"])
            _, tr = run_builder(m5, f, costs, price_now, tc.builder_for(cid, cm), w["search"][0], b, clip=True)
            s = summarize(tr, a, b, days)
            key = f"{cid}|{cm}"
            append_trades(tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr, phase="confirm_search",
                          key=key, cell=cid, candidate=cand, kind="confirm", market=cm)
            d["confirm_search"][key] = s
            save_results(d)
            line("S10", key, s, time.time() - t1)
        del m5, f
        gc.collect()
    promoted = []
    for vk, v in d.get("verdict_search", {}).items():
        if not v["pass_S1_S9_S11_S12"]:
            v["S10 confirmation"] = False
            continue
        cid, m = v["carried_cell"], v["market"]
        ok = True
        conf = tc.CONFIRM[v["candidate"]].get(m, [])
        for cm in conf:
            s = d["confirm_search"].get(f"{cid}|{cm}")
            if s is None:
                ok = False
            elif cm == "BTC (Binance)" and v["candidate"] != "C08-SMT1":
                ok &= s["R"] > 0 and s["PF"] >= 1.10
            elif cm == "ETH (Binance)" and m == "BTC":
                ok &= s["R"] >= 0
            else:
                ok &= s["R"] > 0 and s["PF"] >= 1.05
        v["S10 confirmation"] = bool(ok and conf)
        if v["S10 confirmation"]:
            promoted.append((v["spa_p"] if v["spa_p"] is not None else 1.0, cid, m))
    promoted.sort()
    d["promoted"] = [{"cell": cid, "market": m, "spa_p": p} for p, cid, m in promoted[:MAX_PROMOTED]]
    d["search_judged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return d


def report_search(d):
    print("\n" + "=" * 110)
    print("SEARCH VERDICT (carried cell per candidate x market; every number net of costs)")
    print("=" * 110)
    for vk, v in sorted(d.get("verdict_search", {}).items()):
        cid = v["carried_cell"]
        s = d["search"].get(f"{cid}|{v['market']}") if cid else None
        head = f"{vk:<42} "
        if not s:
            print(head + "no cell passed S1-S4")
            continue
        fails = [k for k, ok in v["cell_checks"][cid].items() if not ok]
        if not v["S5 grid >=3/4 cells R>0"]:
            fails.append("S5")
        if v["pass_S1_S9_S11_S12"] and not v.get("S10 confirmation"):
            fails.append("S10")
        print(head + f"{cid.split('[')[1][:-1]:<22} n={s['n']:>5} win {s['win']}% (p* {s.get('p_star')}%, "
              f"B {s.get('B')}) net {s['R']:+.1f}R PF {s['PF']} SPA p {v['spa_p']}  "
              + ("PASS" if not fails else "fail: " + ", ".join(fails)))
    print(f"\npromoted to the holdout (max {MAX_PROMOTED}): {d.get('promoted') or 'none'}")


# ================================================================ holdout
def cmd_holdout():
    d = load_results()
    if not d.get("promoted"):
        print("nothing was promoted by the search - the holdout stays sealed")
        return
    if d.get("holdout_done"):
        print("the holdout has already been opened for these pairs:")
        print(json.dumps(d.get("holdout"), indent=1, default=str)[:4000])
        return
    d.setdefault("holdout", {})
    purge_trades([f"{p['cell']}|{p['market']}" for p in d["promoted"]
                  if f"{p['cell']}|{p['market']}" not in d["holdout"]], "holdout")
    for p in d["promoted"]:
        cid, m = p["cell"], p["market"]
        key = f"{cid}|{m}"
        if key in d["holdout"]:
            continue
        t1 = time.time()
        cand = tc.CELLS[cid]["candidate"]
        m5, f, costs, price_now = load_market(m)
        w = windows(m5)
        days = pd.Series(m5.index.normalize().unique())
        a, b = w["holdout"]
        _, tr = run_builder(m5, f, costs, price_now, tc.builder_for(cid, m), a, b)
        s = summarize(tr, a, b, days, folds=4)
        ss = d["search"][key]
        # H6 cost stress: spread only enters at entry and swap is linear, so 1.5x is exact
        if len(tr):
            stress = tr["r_net"] - 0.5 * tr["spread_entry"] / tr["risk"] + 0.5 * tr["swap_r"]
            s["stress_1p5_R"] = round(float(stress.sum()), 3)
        else:
            s["stress_1p5_R"] = 0.0
        if tc.KIND.get(m) == "index":
            s["after_2022_R"] = round(float(tr.loc[tr["time"] >= INDEX_REGIME, "r_net"].sum()), 3) if len(tr) else 0.0
        if m in ("BTC", "BTC (Binance)"):
            s["btc_regime_R"] = split_R(tr, a, BTC_REGIME, b)
        low = cand in tc.LOW_FREQ and ss["n"] < 150
        ck = {"H1 netR>0 & PF>=1.10": s["R"] > 0 and s["PF"] >= 1.10,
              "H2 R/tr >= 50% of search": s["per_trade"] >= 0.5 * ss["per_trade"],
              "H3 >=3/4 quarters R>=0": sum(1 for q in s["folds"] if q["R"] >= 0) >= 3,
              "H4 trades": (s["n"] >= 40 and (s.get("p_no_edge") or 100) <= 10) if low else s["n"] >= 60,
              "H6 1.5x costs R>0": s["stress_1p5_R"] > 0,
              "H8 win within 10pts": abs(s["win"] - ss["win"]) <= 10}
        if tc.KIND.get(m) == "index":
            ck["H7 after 2022-01 R>0"] = s["after_2022_R"] > 0
        append_trades(tr[(tr["time"] >= a) & (tr["time"] < b)] if len(tr) else tr, phase="holdout", key=key,
                      cell=cid, candidate=cand, kind="cell", market=m)
        del m5, f
        gc.collect()
        if m == "GOLD":
            if have("GOLD (broker)"):
                s["gold_broker_check"] = gold_broker_check(cid, tr)
                ck["H6 gold broker sign kept"] = bool(s["gold_broker_check"]["sign_kept"])
            else:
                ck["H6 gold broker sign kept"] = False
        s["checks"] = ck
        s["pass_H1_H4_H6_H8"] = all(ck.values())
        d["holdout"][key] = s
        save_results(d)
        line("holdout", key, s, time.time() - t1)
        print("           " + "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in ck.items()), flush=True)
    d["holdout_done"] = True
    d["holdout_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(d)


def gold_broker_check(cid, duka_tr):
    """H6 (gold): the frozen cell on the broker feed with its real spread column,
    over its 2025-04..2026-09 span after a 60-day feature warm-up, next to the
    Dukascopy trades over the same dates. The sign must not flip."""
    m5, f, costs, price_now = load_market("GOLD (broker)")
    a, b = windows(m5, warmup_days=60)["all"]
    _, tr = run_builder(m5, f, costs, price_now, tc.builder_for(cid, "GOLD (broker)"), a, b)
    rb = float(tr["r_net"].sum()) if len(tr) else 0.0
    dk = duka_tr[(duka_tr["time"] >= a) & (duka_tr["time"] < b)] if len(duka_tr) else duka_tr
    rd = float(dk["r_net"].sum()) if len(dk) else 0.0
    del m5, f
    gc.collect()
    return {"from": str(a.date()), "to": str(b.date()), "broker_n": int(len(tr)), "broker_R": round(rb, 3),
            "dukascopy_n": int(len(dk)), "dukascopy_R": round(rd, 3),
            "sign_kept": (rb > 0) == (rd > 0)}


# ================================================================ confirm (H5)
def cmd_confirm():
    d = load_results()
    ho = d.get("holdout", {})
    surv = [k for k, v in ho.items() if v.get("pass_H1_H4_H6_H8")]
    if not surv:
        print("no holdout survivor - nothing to confirm")
        return
    d.setdefault("confirm", {})
    for key in surv:
        cid, m = key.split("|")
        cand = tc.CELLS[cid]["candidate"]
        res = {}
        for cm in tc.CONFIRM[cand].get(m, []):
            ck = f"{cid}|{cm}"
            if ck in d["confirm"]:
                res[cm] = d["confirm"][ck]
                continue
            if not have(cm):
                print(f"[{cm}] missing - H5 cannot pass for {key}")
                continue
            t1 = time.time()
            m5, f, costs, price_now = load_market(cm)
            w = windows(m5)
            days = pd.Series(m5.index.normalize().unique())
            a, b = w["holdout"]
            _, tr = run_builder(m5, f, costs, price_now, tc.builder_for(cid, cm), a, b)
            s = summarize(tr, a, b, days, folds=4)
            append_trades(tr, phase="confirm", key=ck, cell=cid, candidate=cand, kind="confirm", market=cm)
            d["confirm"][ck] = s
            res[cm] = s
            save_results(d)
            line("confirm", ck, s, time.time() - t1)
            del m5, f
            gc.collect()
        conf = tc.CONFIRM[cand].get(m, [])
        ok = bool(conf) and all(cm in res for cm in conf) and all(
            (res[cm]["R"] >= 0 if (cm == "ETH (Binance)" and m == "BTC") else res[cm]["R"] > 0) for cm in conf)
        ho[key]["H5 confirmation"] = ok
        ho[key]["FINAL_PASS"] = ok and ho[key]["pass_H1_H4_H6_H8"]
        print(f"  {key}: H5 {'pass' if ok else 'fail'} -> {'FINAL PASS' if ho[key]['FINAL_PASS'] else 'not passed'}")
    d["holdout"] = ho
    d["confirm_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_results(d)


# ================================================================ smoke test
def cmd_smoke(days_back=60, markets=("GOLD", "NAS100")):
    """Build every cell/control/G9 level on the LAST `days_back` days only and
    print signal counts. No simulation, no backtest."""
    from backtest import load_real_data
    for m in markets:
        t0 = time.time()
        full = load_real_data(os.path.join(PROJ, MARKETS[m][0]))
        m5 = full[full.index >= full.index[-1] - pd.Timedelta(days=days_back)].copy()
        del full
        f = features(m5)
        f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
        print(f"\n[{m}] {len(m5):,} bars {m5.index[0]} -> {m5.index[-1]} ({time.time()-t0:.0f}s)")
        if tc.KIND.get(m) == "index":
            print(f"  G1 stamp check: {check_stamps(f, m)}")
        ids = [c for c, meta in tc.CELLS.items() if m in meta["markets"]]
        ids += [c for c, meta in tc.CONTROLS.items() if m in meta["markets"]]
        for cid in ids:
            t1 = time.time()
            sig = tc.builder_for(cid, m)(m5, f)
            nb = int((sig["direction"] == "buy").sum()) if len(sig) else 0
            print(f"  {cid:<64} {len(sig):>5} signals ({nb} buy / {len(sig) - nb} sell)  "
                  f"hold {sig['hold'].median() if len(sig) else float('nan'):.2f}h  ({time.time()-t1:.1f}s)",
                  flush=True)
        for cand, fac in tc.LEVELS.items():
            if m in tc.PRIMARY[cand]:
                up, dn = fac(m)(m5, f)
                g = tc.level_placebo(f, up, dn)
                print(f"  G9:{cand:<60} {g}")
        del m5, f
        gc.collect()


if __name__ == "__main__":
    cmds = {"smoke": cmd_smoke, "search": cmd_search, "holdout": cmd_holdout, "confirm": cmd_confirm}
    cmds[sys.argv[1] if len(sys.argv) > 1 else "smoke"]()
