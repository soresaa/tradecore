"""
Live signals for the round-4 / round-5 strategies, computed by the study's OWN code.

The app hands over about 250 days of broker 5-minute candles. This module runs the exact
functions the study ran on 23 years of history (experiments/frequency_study.features,
best_per_market_study.daily_dir, improve_lab.Lab.signals with the strategy's spec) on that window
and reads whether the newest closed candle is a signal. There is no second copy of the rules that
could drift from the backtest. experiments/verify_lab_live.py checks that the shorter live window
gives the same signals as the full history (0 missed, 0 different, 0 invented).

A spec is the same JSON the study used, e.g.
    {"base": "DON:4h:20", "filters": ["session != 'late'", "er_d < 0.4", "rpos24 >= 0.85"]}
Signals are decided on the LAST 5-minute candle of each completed timeframe bar (03:55 for the
00:00-04:00 bar), so the app decides with decision_minutes = the bar length and
decision_offset_minutes = 5. Broker server time is UTC (Exness), the same clock as the study.
Places no orders.
"""
from __future__ import annotations

import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

from engine import Signal, TradePlan

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
if EXP not in sys.path:
    sys.path.insert(0, EXP)

TF_MINUTES = {"5min": 5, "15min": 15, "1h": 60, "4h": 240, "1D": 1440}
# Bases that are not timeframe names, and the candle they decide on.
SPECIAL_BASES = {"RC": "5min",           # gold round-number cascade (edge_candidates.c4_round_cascade), every candle
                 "BBH": "5min"}          # EUR/USD 1h band fade (experiments/fx_improve.fam_BBH): signals on the HH:00 candle
LOOKBACK_BARS = 70000          # ~250 days: daily EMA50, 20-day medians and the 4h channel all settle
MIN_BARS = 30000


def tf_of(spec: dict) -> str:
    if spec["base"] in SPECIAL_BASES:
        return SPECIAL_BASES[spec["base"]]
    parts = spec["base"].split(":")
    return parts[1] if len(parts) > 1 and parts[1] in TF_MINUTES else "1h"


def closes_bar(t: pd.Timestamp, tf: str) -> bool:
    """Is the 5-minute candle opened at t the last candle of its tf bar?"""
    end = t + pd.Timedelta(minutes=5)
    return (end.hour * 60 + end.minute) % TF_MINUTES[tf] == 0


_LABS: "dict" = {}                  # candles -> Lab, shared by every strategy on the same symbol and candle
_LABS_LOCK = __import__("threading").Lock()


def _candles_key(m5: pd.DataFrame):
    c = m5["close"]
    return (len(m5), m5.index[0], m5.index[-1], float(c.iloc[-1]), float(m5["high"].iloc[-1]),
            float(m5["low"].iloc[-1]), round(float(c.iloc[-300:].sum()), 6))


def study_lab(m5: pd.DataFrame):
    """The Lab for these candles, built once per symbol and candle: the gold 4h breakout, big target, 3R and
    round-number cards all read the same XAUUSDm candles, so the features are computed once, not four times.
    A Lab only adds caches as it is used (_feat, _built per base), so sharing it changes no result."""
    key = _candles_key(m5)
    with _LABS_LOCK:
        lab = _LABS.get(key)
        if lab is None:
            lab = _build_lab(m5)
            _LABS[key] = lab
            while len(_LABS) > 4:                    # a few symbols at most; drop the oldest
                _LABS.pop(next(iter(_LABS)))
        return lab


def _build_lab(m5: pd.DataFrame):
    """An improve_lab.Lab over these candles (no file, no costs): the study's own feature code."""
    from improve_lab import Lab
    from frequency_study import features
    from best_per_market_study import daily_dir
    m5 = m5[["open", "high", "low", "close"]].astype(float).assign(
        volume=m5["volume"].to_numpy(float) if "volume" in m5.columns else 0.0)
    f = features(m5)
    f["daily_bull"], f["daily_bear"] = daily_dir(m5, f)
    lab = Lab.__new__(Lab)
    lab.market, lab.m5, lab.f, lab._feat, lab._built = "LIVE", m5, f, None, {}
    return lab


def _register(lab, base: str):
    """Study builders that improve_lab does not know by name."""
    if base == "RC" and "RC" not in lab._built:
        import edge_candidates as ec
        lab._built["RC"] = ec.c4_round_cascade("GOLD")(lab.m5, lab.f)
    if base == "BBH" and "BBH" not in lab._built:
        import fx_improve
        lab._built["BBH"] = fx_improve.fam_BBH(lab.m5, lab.f)


def study_signals(m5: pd.DataFrame, spec: dict):
    """(lab, all signals of the spec in this window, the base signals before filters)."""
    lab = study_lab(m5)
    with _LABS_LOCK:                                 # one strategy at a time per shared Lab
        _register(lab, spec["base"])
        base = lab.signals({"base": spec["base"]})
        sig = lab.signals(spec)
    return lab, sig, base


def _describe(lab, spec: dict, tf: str) -> Dict[str, str]:
    """What the screen shows: the timeframe bar, its channel / trend state, the daily trend."""
    import mtf_strategies as mtf
    out = {}
    if spec["base"] == "RC":                       # round numbers: the levels and the new-24-hour-extreme lines
        try:
            import math
            import trap_candidates as tc
            c = float(lab.f["close"][-1])
            step = 50.0 if c >= 1000 else 10.0
            lo = math.floor(c / step) * step
            out["round levels"] = f"{lo:.0f} / {lo + step:.0f}"
            H24, L24 = tc.h20(lab.f, 24)
            h, l = float(H24[-1]), float(L24[-1])
            if math.isfinite(h) and math.isfinite(l) and h > l:
                out["buy needs above"] = f"{l + 0.97 * (h - l):.2f}"
                out["sell needs below"] = f"{l + 0.03 * (h - l):.2f}"
            out["1h trend strength"] = f"{float(lab.f['er_h1'][-1]):.2f} (needs 0.12+)"
        except Exception as e:
            out["info"] = f"n/a ({e})"
        return out
    if spec["base"] == "BBH":                      # the last completed hour's band and the daily chop
        try:
            from backtest import resample
            h1 = resample(lab.m5, "1h")["close"].iloc[:-1]          # completed hours only
            mid, sd = h1.rolling(20).mean().iloc[-1], h1.rolling(20).std(ddof=0).iloc[-1]
            out["1h band"] = f"{mid - 2 * sd:.5f} - {mid + 2 * sd:.5f} (middle {mid:.5f})"
            er = float(lab.f["er_d"][-1])
            out["daily chop"] = f"{er:.2f} ({'choppy - can trade' if er < 0.20 else 'trending - no trade'})"
        except Exception as e:
            out["info"] = f"n/a ({e})"
        return out
    try:
        b = mtf.tf_bars(lab.f, tf)
        kind = spec["base"].split(":")[0]
        last = b.iloc[-1]
        out[f"last {tf} close"] = f"{last['close']:.2f}"
        if kind in ("DON", "DONX"):
            n = int(spec["base"].split(":")[2]) if len(spec["base"].split(":")) > 2 else 20
            hi = b["high"].shift(1).rolling(n).max().iloc[-1]
            lo = b["low"].shift(1).rolling(n).min().iloc[-1]
            out[f"{tf} channel ({n} bars)"] = f"{lo:.2f} - {hi:.2f}"
        else:
            import trend_bases as tb
            k = kind[:-1] if kind.endswith("X") and kind[:-1] in tb.KINDS else kind
            st = tb.current_state(b, k)
            out[f"{k} trend ({tf})"] = {1: "UP", -1: "DOWN"}.get(st, "none")
        i = len(lab.m5) - 1
        out["daily trend"] = ("UP" if lab.f["daily_bull"][i] else "DOWN" if lab.f["daily_bear"][i] else "none")
    except Exception as e:                         # the display must never block a decision
        out["info"] = f"n/a ({e})"
    return out


def _why_filtered(lab, spec: dict, base_row) -> str:
    feats = lab._entry_features(pd.DataFrame([base_row]))
    failed = []
    for q in spec.get("filters", []) or []:
        try:
            if feats.query(q, engine="python").empty:
                failed.append(q)
        except Exception:
            failed.append(q)
    return ", ".join(failed) if failed else "a filter"


def make_signal_fn(spec: dict, label: str):
    """-> fn(data, symbol) for PaperTrader: BUY / SELL with the study's stop and targets, or NO TRADE + why."""
    tf = tf_of(spec)

    def fn(data, symbol: str = "") -> Signal:
        m5 = data.get("5M")
        if m5 is None or len(m5) < MIN_BARS:
            have = 0 if m5 is None else len(m5)
            return Signal("NO TRADE", 0, {}, f"not enough history yet ({have:,} of {MIN_BARS:,} "
                                              f"5-minute candles)", None, {})
        t = m5.index[-1]
        lab, sig, base = study_signals(m5, spec)
        info = _describe(lab, spec, tf)
        if not closes_bar(t, tf):
            nxt = (t + pd.Timedelta(minutes=5)).ceil(f"{TF_MINUTES[tf]}min")
            return Signal("NO TRADE", 0, {}, f"{label}: waits for the {tf} candle to close "
                                              f"(next decision {nxt:%H:%M} UTC)", None, info)
        last = len(m5) - 1
        hit = sig[sig["pos"] == last]
        if hit.empty:
            b = base[base["pos"] == last]
            if b.empty:
                return Signal("NO TRADE", 0, {}, f"{label}: no signal on this {tf} candle", None, info)
            return Signal("NO TRADE", 0, {}, f"{label}: {b.iloc[0]['direction'].upper()} setup skipped "
                                              f"by the tested filter ({_why_filtered(lab, spec, b.iloc[0])})",
                          None, info)
        r = hit.iloc[0]
        e, stop, tp1, tp2 = float(r["entry"]), float(r["stop"]), float(r["tp1"]), float(r["tp2"])
        risk = abs(e - stop)
        plan = TradePlan(r["direction"], e, e, stop, tp1, tp2, abs(tp1 - e) / risk, abs(tp2 - e) / risk)
        return Signal("BUY" if r["direction"] == "buy" else "SELL", 60, {spec["base"]: 1}, None, plan, info)

    fn.spec = spec
    fn.tf = tf
    return fn
