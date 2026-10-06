"""
Round 4 lab: DIAGNOSE -> FIX -> RE-TEST, one market at a time, with a locked final test.

Windows per market, after the 300-day warm-up (L = last bar - start):
  dev   first 60%          fix and re-test here as often as needed
  val   60% -> 80%         every fix must also hold here (it is logged: looks are counted)
  test  80% -> end         LOCKED: opened once per market by the lead, at the very end

Usage (one python process at a time — the laptop shuts down under load):

    from improve_lab import Lab
    lab = Lab("GOLD")                               # loads the market once (10-30 s)
    t = lab.trades("T24", "dev")                    # a base strategy, dev window
    lab.report(t, "T24 base")                       # one stats line
    lab.diagnose(t)                                 # where does it lose? (bucket tables)
    spec = {"base": "T24",
            "filters": ["hour_utc not in (20, 21)", "er_d >= 0.25"],     # pandas query strings
            "exits": {"r1": 1.0, "r2": 2.35, "rule": "breakeven", "hold": 24, "stop_mult": 1.0}}
    lab.report(lab.trades(spec, "dev"), "fix 1 dev")
    lab.report(lab.trades(spec, "val"), "fix 1 val")

A spec is plain JSON: base (a name in BASES, or "tc:<round-1 cell id>"), optional
filters (pandas query strings over the ENTRY FEATURES below, all known at the
signal bar's close), optional exits (recomputed from the base stop distance x
stop_mult; rule in breakeven / no_breakeven / trail_1r / partial_tp1; hold in hours).
Every run is appended to improve_log.jsonl.

ENTRY FEATURES (per signal, at the signal bar's close; direction-aware where noted):
  buy            True = long
  hour_utc, weekday (0=Mon), session ('asia' <07 UTC, 'london' 07-12, 'ny' 12-21, 'late' 21+)
  ny_hour        New York clock hour (DST-aware)
  er_h1, er_d    1-hour / daily efficiency ratio (trendiness, 0..1)
  aligned        +1 trade WITH the daily trend, -1 AGAINST it, 0 no daily trend
  dist50         (close - daily EMA50) / atr1h, signed WITH the trade (+ = trade side above/below as a trend trade)
  mom24          24-hour move / atr1h, signed WITH the trade (+ = price already moved our way)
  rpos24         where the close sits in the prior 24 hours' range, 0 = worst end for the trade, 1 = best end
  atr_ratio      atr1h / median atr1h of the prior 20 days (volatility regime)
  day_used       today's (UTC) range so far / daily ATR(14)
  risk_atr       stop distance / atr1h
"""
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

import trap_study as ts
import trap_candidates as tc
from deep_search_study import BUILDERS as DEEP, simulate_family, sig_frame, daily_atr, WARMUP_DAYS
from best_per_market_study import map_to_5m
from trap_round2 import c05v
import mtf_strategies as mtf               # DON / DONX / TRAP / SQZ / PB / MAX on 15min, 1h, 4h, 1D

LOG = os.path.join(HERE, "improve_log.jsonl")
UNLOCK = "YES-FINAL"

BASES = dict(DEEP)                                  # T24 TSW DDN BBM BB1 LON OVN SQ3 RS2
BASES["C05"] = c05v(12, 4, 1.0, "breakeven")        # the TRAP reversal (round 2)
BASES["C05_CW8"] = c05v(12, 8, 1.0, "breakeven")
BASE_NOTES = {
    "T24": "gold live rules: 1h+15m EMA trend, 15m range break at HH:00, er_h1>=.25, er_d>=.20; SL 1.5 atr1h, 1R->BE, 2.35R, 24h",
    "TSW": "1h 20-bar Donchian break with the daily trend at HH:00; SL 2 atr1h, 2R->BE, 4R, 120h",
    "DDN": "20-day Donchian break; SL 2 daily ATR, trail after 2R, 4R, 240h",
    "BBM": "15m Bollinger fade in choppy days, target the middle band; SL 1 atr1h, 12h",
    "BB1": "15m Bollinger fade in choppy days; SL 1 atr1h, TP 1R, 12h",
    "LON": "London-open momentum of the last 4h; SL 1 atr1h, hold to 16:00 London",
    "OVN": "21:00 UTC with the daily trend, 10h; SL 1.5 atr1h",
    "SQ3": "BTC squeeze: quietest fifth of 20 days, 12h range break at HH:00; SL 1.5 atr1h, trail after 2R, 4R, 72h",
    "RS2": "1h RSI(2) extreme with the daily trend; SL 1.5 atr1h, TP 1.5R, 48h",
    "C05": "TRAP reversal: hammer/star at a fresh 12h extreme against the daily trend, broken within 4h; SL beyond the candle, 1R->BE, 2.5R, 36h",
    "C05_CW8": "C05 with an 8-hour confirmation window",
}
PRIMARY = ["GOLD", "BTC", "NAS100", "US30"]
ALSO = ["SILVER", "US500", "BTC (Binance)", "ETH (Binance)"]      # robustness checks (dev/val only)


def _window_bounds(m5):
    first, last = m5.index[0], m5.index[-1]
    start = first + pd.Timedelta(days=WARMUP_DAYS)
    L = last - start
    return {"dev": (start, start + L * 0.6), "val": (start + L * 0.6, start + L * 0.8),
            "test": (start + L * 0.8, last + pd.Timedelta(days=1))}


class Lab:
    def __init__(self, market):
        if market not in ts.MARKETS:
            raise ValueError(f"unknown market {market!r}; choose from {sorted(ts.MARKETS)}")
        t0 = time.time()
        self.market = market
        self.m5, self.f, self.costs, self.price_now = ts.load_market(market)
        self.w = _window_bounds(self.m5)
        self.days = pd.Series(self.m5.index.normalize().unique())
        self._feat = None
        self._built = {}
        print(f"[{market}] {len(self.m5):,} bars {self.m5.index[0].date()} -> {self.m5.index[-1].date()} | "
              + " | ".join(f"{k} {a.date()}->{b.date()}" for k, (a, b) in self.w.items()) + f" ({time.time()-t0:.0f}s)")

    # ------------------------------------------------------------ features
    def features(self):
        """Per-5m-bar feature arrays (computed once)."""
        if self._feat is not None:
            return self._feat
        from backtest import resample
        f, m5 = self.f, self.m5
        idx, c, A = f["idx"], f["close"], f["atr1h"]
        n = len(c)
        mins, _, dow_ny = tc.ny(f)
        d1 = resample(m5, "1D")
        e50 = map_to_5m(d1["close"].ewm(span=50, adjust=False).mean(), idx, "1D")
        e50 = e50 + (2.0 / 51.0) * (c - e50)
        h = tc.hourly(f)
        med = pd.Series(h["A"].to_numpy(float)).shift(1).rolling(480, min_periods=240).median().to_numpy()
        atr_med = tc.prior_hour(h.index, med, idx)
        ns = idx.as_unit("ns").asi8
        j = np.searchsorted(ns, ns - 24 * 3600 * 10**9, side="left")
        c24 = np.where(j < np.arange(n), c[np.minimum(j, n - 1)], np.nan)
        H24, L24 = tc.h20(f, 24)
        key = np.asarray(idx.floor("1D"))
        dhi = pd.Series(f["high"]).groupby(key).cummax().to_numpy(float)
        dlo = pd.Series(f["low"]).groupby(key).cummin().to_numpy(float)
        atr_d = daily_atr(m5, idx)
        trend = np.where(f["daily_bull"], 1, np.where(f["daily_bear"], -1, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            self._feat = {
                "hour_utc": np.asarray(idx.hour), "weekday": np.asarray(idx.dayofweek),
                "ny_hour": mins // 60, "er_h1": f["er_h1"], "er_d": f["er_d"], "trend": trend,
                "dist50_raw": (c - e50) / A, "mom24_raw": (c - c24) / A,
                "rpos24_raw": (c - L24) / (H24 - L24), "atr_ratio": A / atr_med,
                "day_used": (dhi - dlo) / atr_d,
            }
        return self._feat

    def _entry_features(self, sig):
        F = self.features()
        p = sig["pos"].to_numpy(np.int64)
        buy = (sig["direction"] == "buy").to_numpy()
        s = np.where(buy, 1.0, -1.0)
        hr = F["hour_utc"][p]
        out = pd.DataFrame({
            "buy": buy, "hour_utc": hr, "weekday": F["weekday"][p], "ny_hour": F["ny_hour"][p],
            "session": np.where(hr < 7, "asia", np.where(hr < 12, "london", np.where(hr < 21, "ny", "late"))),
            "er_h1": F["er_h1"][p], "er_d": F["er_d"][p],
            "aligned": (F["trend"][p] * s).astype(int),
            "dist50": F["dist50_raw"][p] * s, "mom24": F["mom24_raw"][p] * s,
            "rpos24": np.where(buy, F["rpos24_raw"][p], 1.0 - F["rpos24_raw"][p]),
            "atr_ratio": F["atr_ratio"][p], "day_used": F["day_used"][p],
            "risk_atr": sig["risk"].to_numpy() / self.f["atr1h"][p],
        }, index=sig.index)
        return out

    # ------------------------------------------------------------ building
    def _base_sig(self, base):
        if base not in self._built:
            if base.startswith("tc:"):
                builder = tc.builder_for(base[3:], self.market)
            elif mtf.from_name(base) is not None:
                builder = mtf.from_name(base)
            else:
                builder = BASES[base]
            self._built[base] = builder(self.m5, self.f)
        return self._built[base]

    def signals(self, spec):
        """spec (str base name or dict) -> (sig_frame with feature columns)."""
        if isinstance(spec, str):
            spec = {"base": spec}
        sig = self._base_sig(spec["base"]).copy()
        sig = pd.concat([sig, self._entry_features(sig)], axis=1)
        for q in spec.get("filters", []) or []:
            sig = sig.query(q, engine="python")
        ex = spec.get("exits")
        if ex:
            k = float(ex.get("stop_mult", 1.0))
            sgn = np.where(sig["direction"] == "buy", 1.0, -1.0)
            risk = sig["risk"].to_numpy() * k
            e = sig["entry"].to_numpy()
            sig = sig.assign(stop=e - sgn * risk, risk=risk,
                             tp1=e + sgn * float(ex["r1"]) * risk, tp2=e + sgn * float(ex["r2"]) * risk,
                             rule=ex.get("rule", "breakeven"))
            if ex.get("hold") is not None:
                sig = sig.assign(hold=float(ex["hold"]))
        return sig.sort_values("pos").reset_index(drop=True)

    def trades(self, spec, window="dev"):
        if window == "test" and os.environ.get("TRADECORE_UNLOCK_TEST") != UNLOCK:
            raise PermissionError("the TEST window is locked; only the lead opens it, once, at the end")
        if window not in self.w:
            raise ValueError("window must be dev, val or test")
        a, b = self.w[window]
        sig = self.signals(spec)
        sim = self.m5.iloc[:int(self.m5.index.searchsorted(b))] if window != "test" else self.m5
        tr = simulate_family(sim, sig, self.costs, a, b, swap_scale_to=self.price_now)
        if len(tr):
            feats = [c for c in sig.columns if c not in ("time", "pos", "direction", "entry", "stop", "risk", "tp1", "tp2", "hold", "rule")
                     and c not in tr.columns]
            tr = tr.merge(sig[["time"] + feats], on="time", how="left")
            tr = tr[(tr["time"] >= a) & (tr["time"] < b)].reset_index(drop=True)
        tr.attrs.update(market=self.market, window=window, spec=spec if isinstance(spec, dict) else {"base": spec},
                        a=str(a), b=str(b))
        return tr

    # ------------------------------------------------------------ reporting
    def stats(self, tr):
        a, b = pd.Timestamp(tr.attrs["a"]), pd.Timestamp(tr.attrs["b"])
        s = ts.summarize(tr if len(tr) else None, a, b, self.days, folds=5 if tr.attrs["window"] == "dev" else 4)
        yrs = max((b - a).days / 365.25, 0.1)
        s["trades_per_year"] = round(s["n"] / yrs, 1)
        if len(tr):
            s["stress_1p5_R"] = round(float((tr["r_net"] - 0.5 * tr["spread_entry"] / tr["risk"] + 0.5 * tr["swap_r"]).sum()), 2)
        return s

    def report(self, tr, label=""):
        s = self.stats(tr)
        line = (f"{self.market:<14}{tr.attrs['window']:<5}{label:<34} n={s['n']:>5} ({s.get('trades_per_year', 0):>5}/yr) "
                f"win {s['win']:>5}% (p* {s.get('p_star', '-')})  net {s['R']:>+8.2f}R  {s['per_trade']:>+.3f}R/tr  "
                f"PF {s['PF']:<6} maxDD {s.get('maxDD', 0):>+6.1f}  folds+ {s.get('positive_folds', 0)}/{5 if tr.attrs['window']=='dev' else 4}  "
                f"P(no edge) {s.get('p_no_edge')}%  1.5x {s.get('stress_1p5_R', 0):+.1f}")
        print(line, flush=True)
        rec = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "market": self.market, "window": tr.attrs["window"],
               "label": label, "spec": tr.attrs["spec"],
               **{k: s.get(k) for k in ("n", "trades_per_year", "win", "p_star", "R", "per_trade", "PF", "maxDD",
                                        "positive_folds", "p_no_edge", "stress_1p5_R", "last_third_R")}}
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        return s

    def diagnose(self, tr, min_n=25, show=True):
        """Where does it win and lose? Bucket tables per entry feature (win %, R/trade, n, total R)."""
        if not len(tr):
            print("no trades")
            return {}
        out = {}
        cat = ["buy", "session", "weekday", "aligned", "hour_utc"]
        num = ["er_h1", "er_d", "dist50", "mom24", "rpos24", "atr_ratio", "day_used", "risk_atr"]
        for col in cat + num:
            if col not in tr.columns:
                continue
            x = tr[col]
            if col in num:
                try:
                    g = pd.qcut(x, 4, duplicates="drop")
                except ValueError:
                    continue
            else:
                g = x
            t = tr.groupby(g, observed=True)["r_net"].agg(n="size", win=lambda r: round(100 * (r > 0).mean(), 1),
                                                         R_tr=lambda r: round(r.mean(), 3), R="sum")
            t["R"] = t["R"].round(2)
            t = t[t["n"] >= min_n] if col == "hour_utc" else t
            out[col] = t
            if show:
                print(f"\n-- {col}")
                print(t.to_string())
        ex = tr["outcome"].value_counts().to_dict()
        if show:
            print(f"\n-- exits: {ex}")
        out["exits"] = ex
        return out


def log_summary(market=None):
    """How many dev/val looks were taken (the multiple-testing count)."""
    if not os.path.exists(LOG):
        return {}
    rows = [json.loads(l) for l in open(LOG, encoding="utf-8")]
    if market:
        rows = [r for r in rows if r["market"] == market]
    return pd.DataFrame(rows).groupby(["market", "window"]).size().to_dict() if rows else {}


if __name__ == "__main__":
    mk = sys.argv[1] if len(sys.argv) > 1 else "GOLD"
    base = sys.argv[2] if len(sys.argv) > 2 else "T24"
    lab = Lab(mk)
    t = lab.trades(base, "dev")
    lab.report(t, f"{base} base")
    lab.diagnose(t)
