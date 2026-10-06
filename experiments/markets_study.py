"""
Runs PREREGISTRATION_MARKETS.md: the unchanged live strategy on other markets.

  python experiments/markets_study.py

Same entry rules, same management, same simulator as the gold results. Each
market is charged ITS OWN measured spread and swap from broker_costs_multi.json.
Nothing is tuned per market: a market either works with the gold settings or it
is out.
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

from frequency_study import signals, simulate, stats     # the engine checked against 712 trades

COSTS_JSON = os.path.join(HERE, "broker_costs_multi.json")   # includes NAS100 since 2026-09-21
WARMUP_DAYS = 300          # daily EMA50 + the filter's 20 completed days need a year-ish
MIN_YEARS = 5.0
MIN_TRADES = 30
D0 = (60, 0.25, True, 0.20, 24.0)    # the live settings: hourly, ER .25, filter on, 24h hold

# label -> (csv file, key in broker_costs_multi.json)
MARKETS = {
    "XAU/USD gold": ("xauusd_m5_dukascopy_23y.csv", "XAUUSD"),
    "EUR/USD": ("eurusd_m5_dukascopy_23y.csv", "EURUSD"),
    "GBP/USD": ("gbpusd_m5_dukascopy_23y.csv", "GBPUSD"),
    "USD/JPY": ("usdjpy_m5_dukascopy_23y.csv", "USDJPY"),
    "XAG/USD silver": ("xagusd_m5_dukascopy_23y.csv", "XAGUSD"),
    "US oil (WTI)": ("usoil_m5_dukascopy_23y.csv", "USOIL"),
    "BTC/USD": ("btcusd_m5_dukascopy_23y.csv", "BTCUSD"),
    "NAS100": ("nas100_m5_dukascopy_23y.csv", "NAS100"),
}


def cost_model(key: str):
    from trade_accounting import CostModel
    with open(COSTS_JSON, encoding="utf-8") as f:
        snap = json.load(f)
    s = snap["symbols"].get(key)
    if not s or not s.get("available"):
        raise SystemExit(f"{key}: not in {COSTS_JSON} — run fetch_broker_costs_multi.py")
    m = s.get("swap_model", {})
    per_unit = m.get("kind") == "price_per_unit"
    hist = s.get("spread_history", {})
    return CostModel(
        spread_price=float(hist.get("median_price", 0.0)),
        swap_long_per_night=float(m.get("long", 0.0)) if per_unit else 0.0,
        swap_short_per_night=float(m.get("short", 0.0)) if per_unit else 0.0,
        triple_swap_weekday=int(s.get("triple_swap_weekday_pandas", 2)),
        point=float(s.get("point", 0.001)),
        source=f"{s['broker_symbol']} @ {snap['fetched_at']}"
               + ("" if per_unit else f" (swap mode {m.get('kind')!r} not convertible, swap = 0)"),
    )


def run_market(args):
    """`scaled`: charge the measured spread as a SHARE of price, so a 2004 trade in
    $400 gold is not charged today's $0.24 (the price is 10x higher now and the
    stop with it). `flat` charges today's spread at every date, as the earlier
    gold studies did, and is reported as a sensitivity check."""
    label, scaled = args
    from backtest import load_real_data
    file, key = MARKETS[label]
    path = os.path.join(PROJ, file)
    if not os.path.exists(path):
        return label, scaled, None, {"error": f"missing {file}"}
    t0 = time.time()
    m5 = load_real_data(path)
    costs = cost_model(key)
    price_now = float(m5["close"].tail(8640).median())      # ~30 days of 5-minute bars
    if scaled:
        m5 = m5.copy()
        m5["spread"] = costs.spread_price * m5["close"] / price_now
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=WARMUP_DAYS)
    years = (last - start).days / 365.25
    mid = start + (last - start) / 2
    sig = signals(m5)
    cadence, er_min, use_filter, er_d, hold = D0
    tr = simulate(m5, sig, cadence, er_min, use_filter, er_d, hold, start, last, costs=costs)
    if not tr.empty:
        tr["market"] = label
    days = pd.Series(m5.index.normalize().unique())
    meta = {
        "scaled": scaled, "price_now": price_now,
        "first": str(first), "last": str(last), "start": str(start), "mid": str(mid),
        "years": round(years, 1), "bars": int(len(m5)), "signals": int(len(sig)),
        "spread": costs.spread_price, "swap_long": costs.swap_long_per_night,
        "swap_short": costs.swap_short_per_night, "costs": costs.source,
        "median_stop": float(tr["risk"].median()) if not tr.empty else float("nan"),
        "train_days": int(((days >= start) & (days < mid) & (days.dt.dayofweek < 5)).sum()),
        "test_days": int(((days >= mid) & (days <= last) & (days.dt.dayofweek < 5)).sum()),
        "train_years": (mid - start).days / 365.25, "test_years": (last - mid).days / 365.25,
    }
    if not tr.empty:
        meta["median_spread_charged"] = float(
            (tr["spread_entry"]).median()) if "spread_entry" in tr else costs.spread_price
    print(f"  {label:<16} {'scaled' if scaled else 'flat  '} spread: {len(tr):>5} trades over "
          f"{meta['years']:>4.1f} years ({time.time()-t0:.0f}s)", flush=True)
    return label, scaled, tr, meta


def max_concurrent(tr: pd.DataFrame) -> int:
    """How many positions are open at once across markets — each one risks 1R."""
    if tr.empty:
        return 0
    ev = [(pd.Timestamp(t), 1) for t in tr["time"]] + [(pd.Timestamp(t), -1) for t in tr["exit_time"]]
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def main():
    t0 = time.time()
    labels = [k for k in MARKETS if os.path.exists(os.path.join(PROJ, MARKETS[k][0]))]
    missing = [k for k in MARKETS if k not in labels]
    if missing:
        print(f"no data file yet for: {', '.join(missing)}")
    with ProcessPoolExecutor(max_workers=min(8, 2 * len(labels))) as ex:
        out = list(ex.map(run_market, [(l, True) for l in labels] + [(l, False) for l in labels]))
    results = [(l, tr, m) for l, sc, tr, m in out if sc]
    flat = {l: (tr, m) for l, sc, tr, m in out if not sc}

    rows, table, keep = [], {}, []
    print("\n" + "=" * 132)
    print("THE SAME STRATEGY ON OTHER MARKETS — unchanged rules, each market charged its own spread and swap")
    print("=" * 132)
    print(f"{'market':<16}{'split':<7}{'from':<12}{'to':<12}{'trades':>7}{'/day':>6}{'win%':>7}"
          f"{'net R':>9}{'R/year':>8}{'R/trade':>9}{'PF':>7}{'maxDD':>8}{'P(no edge)':>12}")
    for label, tr, meta in results:
        if tr is None:
            print(f"{label:<16} {meta['error']}")
            continue
        table[label] = {"meta": meta}
        mid, start, last = pd.Timestamp(meta["mid"]), pd.Timestamp(meta["start"]), pd.Timestamp(meta["last"])
        for split, (a, b, nd, yrs) in (("TRAIN", (start, mid, meta["train_days"], meta["train_years"])),
                                       ("TEST", (mid, last + pd.Timedelta(days=1), meta["test_days"], meta["test_years"]))):
            part = tr[(tr["time"] >= a) & (tr["time"] < b)] if not tr.empty else tr
            s = stats(part, max(nd, 1), max(yrs, 0.1))
            table[label][split] = s
            print(f"{label:<16}{split:<7}{str(a.date()):<12}{str(b.date()):<12}{s['n']:>7}{s['per_day']:>6}"
                  f"{s['win']:>7}{s['R']:>9}{s['R_year']:>8}{s['per_trade']:>9}{s['PF']:>7}"
                  f"{s['maxDD']:>8}{s['p_no_edge']:>11}%")
        if not tr.empty:
            rows.append(tr)

    allt = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not allt.empty:
        allt.to_csv(os.path.join(HERE, "markets_trades.csv"), index=False)

    print("\ncost and stop size per market (the spread as a share of 1R is what decides a market):")
    for label in table:
        m = table[label]["meta"]
        charged = m.get("median_spread_charged", m["spread"])
        share = charged / m["median_stop"] * 100 if m["median_stop"] == m["median_stop"] else float("nan")
        print(f"  {label:<16} median stop {m['median_stop']:>10.4f}  spread charged {charged:>9.5f} "
              f"= {share:>5.2f}% of 1R   (today {m['spread']:.5f} at price {m['price_now']:.2f})   "
              f"swap/night {m['swap_long']:+.5f}/{m['swap_short']:+.5f}")

    print("\nsensitivity — the same markets charged TODAY'S spread at every date instead:")
    for label in table:
        tr, _ = flat.get(label, (None, None))
        if tr is None or tr.empty:
            continue
        meta = table[label]["meta"]
        mid = pd.Timestamp(meta["mid"])
        a = stats(tr[tr["time"] < mid], max(meta["train_days"], 1), max(meta["train_years"], 0.1))
        b = stats(tr[tr["time"] >= mid], max(meta["test_days"], 1), max(meta["test_years"], 0.1))
        print(f"  {label:<16} TRAIN {a['R']:>8.2f}R PF {a['PF']:<7} | TEST {b['R']:>8.2f}R PF {b['PF']}")

    print("\n" + "-" * 132)
    print("INCLUSION RULE (fixed in advance, PREREGISTRATION_MARKETS.md)")
    for label in table:
        tr_s, te_s, m = table[label]["TRAIN"], table[label]["TEST"], table[label]["meta"]
        enough = m["years"] >= MIN_YEARS and tr_s["n"] >= MIN_TRADES and te_s["n"] >= MIN_TRADES
        checks = {
            "TRAIN R>0": tr_s["R"] > 0, "TRAIN PF>=1.05": tr_s["PF"] >= 1.05,
            "TEST R>0": te_s["R"] > 0, "TEST PF>=1.05": te_s["PF"] >= 1.05,
            "TEST P(no edge)<=10%": te_s["p_no_edge"] <= 10, "enough data": enough,
        }
        ok = all(checks.values())
        print(f"  {label:<16} {'INCLUDE' if ok else 'exclude'}  " +
              "  ".join(f"[{'x' if v else ' '}] {k}" for k, v in checks.items()))
        if ok:
            keep.append(label)

    print(f"\nmarkets that pass: {', '.join(keep) if keep else 'none'}")
    verdict = {"included": keep, "portfolio": None}
    if keep and not allt.empty:
        port = allt[allt["market"].isin(keep)].sort_values("time").reset_index(drop=True)
        first = pd.Timestamp(min(table[l]["meta"]["start"] for l in keep))
        last = pd.Timestamp(max(table[l]["meta"]["last"] for l in keep))
        cal = pd.date_range(first, last, freq="B")
        yrs = (last - first).days / 365.25
        s = stats(port, len(cal), yrs)
        conc = max_concurrent(port)
        by_year = port.groupby(pd.to_datetime(port["time"]).dt.year)["r_net"].sum()
        print("\n" + "=" * 132)
        print(f"PORTFOLIO of the markets that passed — {first.date()} -> {last.date()}")
        print("=" * 132)
        print(f"  {s['n']} trades, {s['per_day']} per trading day, win {s['win']}%, "
              f"net {s['R']:+.2f}R ({s['R_year']:+.2f}R per year), PF {s['PF']}, "
              f"combined max drawdown {s['maxDD']}R, P(no edge) {s['p_no_edge']}%")
        print(f"  worst year {by_year.min():+.2f}R, losing years {int((by_year < 0).sum())} of {len(by_year)}")
        print(f"  MOST POSITIONS OPEN AT ONCE: {conc} — that many R at risk simultaneously")
        print("\n  net R by year and market:")
        print(port.pivot_table(index=pd.to_datetime(port["time"]).dt.year, columns="market",
                               values="r_net", aggfunc="sum").round(1).to_string())
        verdict["portfolio"] = {**s, "max_concurrent": conc, "worst_year": round(float(by_year.min()), 2)}

    with open(os.path.join(HERE, "markets_decision.json"), "w") as f:
        json.dump({"verdict": verdict, "table": table}, f, indent=2, default=str)
    print(f"\n({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
