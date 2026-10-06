"""
Per-market strategies for the four markets beside gold.

One of these has real evidence; three do not. All four run as a forward test.

    market    strategy                  evidence
    BTC/USD   volatility squeeze SQ3    PASSED A SEALED HOLDOUT: 177 trades the search
                                        never saw, +60.3R, PF 1.58, +0.34R/trade,
                                        9 positive years of 9 (RESULTS_DEEP.md)
    EUR/USD   Bollinger fade            best of its market but below every bar
    GBP/USD   Bollinger fade            same, and it moves with EUR/USD
    USD/JPY   the gold breakout         LOST in the older half of its history

Gold keeps `live_strategy.generate_signal_indie_trend` and is not touched here:
its swing variant was tested on the same holdout and came out worse
(PF 1.12 vs 1.25, drawdown -35R vs -23R).

A second gold slot, XAUUSD_RC, forward-tests the round-number stop cascade
from the research round (experiments/RESULTS_EDGES.md, candidate C4): positive
over 2004-2026 but below the pre-registered bar, so it is paper-only and never
replaces the live gold strategy.

Management is per market, not one size for all. Gold and the three weak markets
use the gold rules (stop 1.5 x 1-hour ATR, TP1 = 1R to breakeven, TP2 = 2.35R,
24-hour limit). Bitcoin uses what its study measured: the same stop, first
target 2R which arms a stop trailing 1R behind the best price, second target
4R, 72-hour limit. One position at a time, decisions on the closed HH:00
candle, no orders placed.

The entry maths in this file is the SAME code path the study used
(`experiments/best_per_market_study.py` imports these helpers), so a live
signal and a backtested signal cannot drift apart.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from engine import Signal, TradePlan, atr
from live_strategy import ER_MIN, daily_efficiency_ratio

ATR_MULT = 1.5          # stop distance, as everywhere else in this project
RR2 = 2.35
# The Bitcoin squeeze strategy that passed the sealed holdout keeps its own
# targets and holding time: first target 2R, second 4R, stop trails 1R behind
# once 2R is reached, out after 72 hours (experiments/RESULTS_DEEP.md).
SQUEEZE_RR = (2.0, 4.0)
SQUEEZE_HOLD_HOURS = 72.0
BB_LEN, BB_SIGMA = 20, 2.0
SQUEEZE_ATR_LEN, SQUEEZE_WINDOW_H, SQUEEZE_PCT = 14, 480, 0.20   # 480 hours = 20 days
SQUEEZE_RANGE_H = 12

# How much 5-minute history each strategy needs behind the decision candle.
# The squeeze reads a 20-day window of hourly bars, so it needs far more than
# the 2,000 bars (7 days) the gold strategy runs on.
LOOKBACK_BARS = {"bb_fade": 2000, "squeeze": 8000, "indie_trend": 2000, "round_cascade": 2000}

# Gold round-number stop cascade (C4 in experiments/PREREGISTRATION_EDGES.md):
# no take-profit and no breakeven, the trade is closed 2 hours after entry.
# The study's simulator held it 115 minutes from the open of the candle after
# the signal, which is 2 hours from the moment the signal candle closed.
ROUND_HOLD_HOURS = 2.0
ROUND_NO_TARGET_R = 100.0    # a target that is never reached, exactly as tested


def _plan(direction: str, price: float, risk: float, rr=(1.0, RR2)) -> TradePlan:
    sign = 1.0 if direction == "buy" else -1.0
    r1, r2 = rr
    return TradePlan(direction, price, price, price - sign * risk,
                     price + sign * r1 * risk, price + sign * r2 * risk, r1, r2)


def _risk_from_atr1h(h1: pd.DataFrame) -> float:
    """1.5 x ATR(14) on the 1-hour frame, the forming hour included -- exactly
    what indie_signal_port does, so every strategy risks the same way."""
    return float(ATR_MULT * atr(h1).iloc[-1])


# --------------------------------------------------------------------- helpers
# These two functions are the single definition of each entry idea. The
# vectorized study calls them through experiments/best_per_market_study.py on
# whole arrays; the live runner calls them on the last candle.
def bb_fade_levels(m15_close: pd.Series) -> tuple:
    """Bollinger band of the COMPLETED 15-minute bars (never the forming one)."""
    mid = m15_close.rolling(BB_LEN).mean()
    sd = m15_close.rolling(BB_LEN).std(ddof=0)
    return mid + BB_SIGMA * sd, mid - BB_SIGMA * sd


def squeeze_state(h1: pd.DataFrame) -> tuple:
    """(is_quiet, 12-hour high, 12-hour low) from COMPLETED hourly bars."""
    prev_close = h1["close"].shift(1)
    tr = pd.concat([h1["high"] - h1["low"], (h1["high"] - prev_close).abs(),
                    (h1["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr_h = tr.rolling(SQUEEZE_ATR_LEN).mean()
    quiet = atr_h <= atr_h.rolling(SQUEEZE_WINDOW_H).quantile(SQUEEZE_PCT)
    return quiet, h1["high"].rolling(SQUEEZE_RANGE_H).max(), h1["low"].rolling(SQUEEZE_RANGE_H).min()


def _check(data: Dict[str, pd.DataFrame], need_15m: int, need_1h: int) -> Optional[Signal]:
    for tf, need in (("5M", 3), ("1H", need_1h), ("15M", need_15m), ("1D", 25)):
        if tf not in data or data[tf] is None or len(data[tf]) < need:
            return Signal("NO TRADE", 0, {}, f"not enough {tf} history yet", None, {})
    return None


# --------------------------------------------------------------------- strategies
def generate_signal_bb_fade(data: Dict[str, pd.DataFrame], symbol: str = "") -> Signal:
    """EUR/USD, GBP/USD candidate — mean reversion.

    When the daily market is CHOPPY (the opposite of the gold filter), a
    15-minute close that pushed outside the 2-sigma band and then closes back
    inside is faded toward the mean.
    """
    bad = _check(data, BB_LEN + 2, SQUEEZE_ATR_LEN + 2)
    if bad is not None:
        return bad
    m5, m15, h1, daily = data["5M"], data["15M"], data["1H"], data["1D"]

    upper, lower = bb_fade_levels(m15["close"])
    up_level, low_level = float(upper.iloc[-2]), float(lower.iloc[-2])   # completed bar
    close_now, close_prev = float(m5["close"].iloc[-1]), float(m5["close"].iloc[-2])
    er = daily_efficiency_ratio(daily)
    tf = {"1D efficiency": "n/a" if er is None else f"{er:.2f}",
          "band": f"{low_level:.5f} - {up_level:.5f}"}

    if er is None or not np.isfinite(up_level) or not np.isfinite(low_level):
        return Signal("NO TRADE", 0, {}, "bands or daily efficiency not ready", None, tf)
    if er >= ER_MIN:
        return Signal("NO TRADE", 0, {}, f"daily market is trending ({er:.2f}) — this strategy "
                                          f"only fades a choppy market", None, tf)
    risk = _risk_from_atr1h(h1)
    if not risk > 0:
        return Signal("NO TRADE", 0, {}, "ATR not ready", None, tf)

    if close_prev > up_level and close_now <= up_level:
        return Signal("SELL", 60, {"bb_fade": 1}, None, _plan("sell", close_now, risk), tf)
    if close_prev < low_level and close_now >= low_level:
        return Signal("BUY", 60, {"bb_fade": 1}, None, _plan("buy", close_now, risk), tf)
    return Signal("NO TRADE", 0, {}, "price has not come back inside the band", None, tf)


def generate_signal_squeeze(data: Dict[str, pd.DataFrame], symbol: str = "") -> Signal:
    """BTC/USD — volatility contraction breakout (SQ3).

    When the 1-hour ATR sits in the quietest fifth of the last 20 days, a break
    of the 12-hour range is taken in the direction of the break. Stop 1.5 x 1H
    ATR, first target 2R (which arms a stop that trails 1R behind the best
    price), second target 4R, out after 72 hours.

    This is the one strategy in the project besides gold that passed a sealed
    holdout: 177 trades on history the search never saw, +60.3R, PF 1.58,
    +0.34R per trade, nine positive years out of nine, shorts paying more than
    longs (experiments/RESULTS_DEEP.md). It is still a backtest.
    """
    bad = _check(data, BB_LEN + 2, SQUEEZE_WINDOW_H + SQUEEZE_ATR_LEN)
    if bad is not None:
        return bad
    m5, h1 = data["5M"], data["1H"]

    quiet, hi12, lo12 = squeeze_state(h1)
    is_quiet = bool(quiet.iloc[-2])                 # completed hour
    hi, lo = float(hi12.iloc[-2]), float(lo12.iloc[-2])
    close_now, close_prev = float(m5["close"].iloc[-1]), float(m5["close"].iloc[-2])
    tf = {"volatility": "quiet (squeeze)" if is_quiet else "normal",
          "12h range": f"{lo:.2f} - {hi:.2f}"}

    if not np.isfinite(hi) or not np.isfinite(lo):
        return Signal("NO TRADE", 0, {}, "12-hour range not ready", None, tf)
    if not is_quiet:
        return Signal("NO TRADE", 0, {}, "no volatility squeeze — this strategy only trades "
                                          "a break out of a quiet stretch", None, tf)
    risk = _risk_from_atr1h(h1)
    if not risk > 0:
        return Signal("NO TRADE", 0, {}, "ATR not ready", None, tf)

    if close_now > hi and close_prev <= hi:
        return Signal("BUY", 60, {"squeeze": 1}, None,
                      _plan("buy", close_now, risk, SQUEEZE_RR), tf)
    if close_now < lo and close_prev >= lo:
        return Signal("SELL", 60, {"squeeze": 1}, None,
                      _plan("sell", close_now, risk, SQUEEZE_RR), tf)
    return Signal("NO TRADE", 0, {}, "no break of the 12-hour range", None, tf)


def round_step_gold(price: float) -> float:
    """The '00/50' grid registered before the test: $50 above $1,000, $10 below."""
    return 50.0 if price >= 1000 else 10.0


def generate_signal_round_cascade(data: Dict[str, pd.DataFrame], symbol: str = "") -> Signal:
    """GOLD — round-number stop cascade (Osler 2003, 2005).

    Stop orders cluster just beyond round numbers, so a close through one
    tends to trigger more selling (or buying) in the same direction. A
    5-minute close beyond a $50 level that the previous close was on the other
    side of is traded in the direction of the cross. Stop 1.5 x 1H ATR, no
    target, no breakeven, closed 2 hours after entry. Decided on every closed
    5-minute candle.

    Research near-miss, not a proven strategy: 2004-2026 +128R over 8,253
    trades, PF 1.055, 4 of 5 periods positive, P(no edge) 4% — it failed the
    pre-registered bar of PF 1.10 and P 1% (experiments/RESULTS_EDGES.md).
    Same arithmetic as experiments/edge_candidates.c4_round_cascade.
    """
    m5, h1 = data.get("5M"), data.get("1H")
    if m5 is None or h1 is None or len(m5) < 2 or len(h1) < 15:
        return Signal("NO TRADE", 0, {}, "not enough history yet", None, {})
    close_now, close_prev = float(m5["close"].iloc[-1]), float(m5["close"].iloc[-2])
    step = round_step_gold(close_prev)
    level_now, level_prev = math.floor(close_now / step), math.floor(close_prev / step)
    tf = {"round levels": f"{level_now * step:.0f} / {(level_now + 1) * step:.0f}"}

    if level_now == level_prev:
        return Signal("NO TRADE", 0, {}, f"no ${step:.0f} round number crossed on this candle",
                      None, tf)
    risk = _risk_from_atr1h(h1)
    if not risk > 0:
        return Signal("NO TRADE", 0, {}, "ATR not ready", None, tf)
    no_target = (ROUND_NO_TARGET_R, ROUND_NO_TARGET_R)
    if level_now > level_prev:
        return Signal("BUY", 60, {"round_cascade": 1}, None,
                      _plan("buy", close_now, risk, no_target), tf)
    return Signal("SELL", 60, {"round_cascade": 1}, None,
                  _plan("sell", close_now, risk, no_target), tf)


# --------------------------------------------------------------------- playbook
def _indie_trend(data, symbol=""):
    from live_strategy import generate_signal_indie_trend
    return generate_signal_indie_trend(data, symbol=symbol)


# Round 4 winners (experiments/RESULTS_ROUND4.md). Their live signals come from lab_live, which runs
# the study's own code on ~250 days of broker candles, so the app cannot trade them differently from
# the backtest (checked by experiments/verify_lab_live.py). Decided on the LAST 5-minute candle of each
# 4h / 1h bar, as tested.
from lab_live import LOOKBACK_BARS as LAB_LOOKBACK, make_signal_fn

GOLD_BO4H_SPEC = {"base": "DON:4h:20", "filters": ["session != 'late'", "er_d < 0.4", "rpos24 >= 0.85"]}
BTC_BO1H_SPEC = {"base": "DON:1h:55"}
# Gold round numbers, improved and PASSED its final test (experiments/PREREGISTRATION_RC_IMPROVE.md, rc_improve.json):
# only crosses at a new 24-hour extreme, not in a choppy hour, not late at night; stop 1.5 x 1h ATR, ONE take profit
# at 0.5R, still closed after 2 hours.
# The gold 4h breakout with the two other exits the user chose to watch (2026-10-04). Same entries as GOLD_BO4H_SPEC.
GOLD_BO4H_BIG_SPEC = dict(GOLD_BO4H_SPEC, exits={"r1": 2.0, "r2": 6.0, "rule": "breakeven", "hold": 120.0, "stop_mult": 0.5})
GOLD_BO4H_3R_SPEC = dict(GOLD_BO4H_SPEC, exits={"r1": 0.75, "r2": 3.0, "rule": "partial_tp1", "hold": 120.0, "stop_mult": 1.0})
RC2_SPEC = {"base": "RC", "filters": ["rpos24 >= 0.97", "er_h1 >= 0.12", "session != 'late'"],
            "exits": {"r1": 0.5, "r2": 0.5, "rule": "no_breakeven", "stop_mult": 1.0}}
# filters chosen on the pooled EUR/USD + GBP/USD dev trades (experiments/fx_improve.json FILTERS.BBH)
EURUSD_BBH_SPEC = {"base": "BBH", "filters": ["day_used < 1.20198", "dist50 < 1.94916", "session != 'asia'"]}

# market key -> what the app runs, what it is worth, and how much history it needs
PLAYBOOK = {
    "XAUUSD": {"broker": "XAUUSDm", "name": "XAU/USD gold", "strategy": "indie_trend",
               "fn": _indie_trend, "status": "TESTED",
               "exit_rule": "breakeven", "max_hold": 24.0,
               "evidence": "2004-2026: +113R net, PF 1.25, 6 losing years of 23. "
                           "The only strategy here with evidence behind it.",
               "lookback": LOOKBACK_BARS["indie_trend"]},
    "BTCUSD": {"broker": "BTCUSDm", "name": "BTC/USD", "strategy": "squeeze 2R/4R trail",
               "fn": generate_signal_squeeze, "status": "FRAGILE",
               "exit_rule": "trail_1r", "max_hold": SQUEEZE_HOLD_HOURS,
               "evidence": "Passed a sealed test on Dukascopy data (PF 1.58) but on Binance data it is "
                           "flat over 2018-2026 (4 of 9 years positive) and it FAILS on Ethereum. Forward test only.",
               "lookback": LOOKBACK_BARS["squeeze"]},
    # USD/JPY: the gold 4h breakout + big target, copied UNCHANGED (experiments/usdjpy_transfer.py) — PASSED.
    "USDJPY_BO4H_BIG": {"broker": "USDJPYm", "cost_key": "USDJPY", "name": "USD/JPY 4h breakout - big target",
                        "strategy": "gold's 4h breakout rules unchanged: stop 1 x ATR, 2R -> stop to entry, TP 6R, 120h",
                        "fn": make_signal_fn(GOLD_BO4H_BIG_SPEC, "USD/JPY 4h big target"), "status": "TEST-PASSED",
                        "exit_rule": "breakeven", "max_hold": 120.0,
                        "decision_minutes": 240, "decision_offset": 5,
                        "evidence": "Gold's rules, not re-tuned. 2004-17 +40.4R PF 1.16; 2017-22 +53.5R PF 1.58; final "
                                    "test 2022-26: 136 trades, 23% won, +28.4R, PF 1.36, worst drop -25R. Paper forward test.",
                        "lookback": LAB_LOOKBACK},
    # The closest EUR/USD candidate (experiments/fx_improve.py): it did NOT pass (dev 2004-17 PF 1.142, bar 1.15) —
    # paper forward test only, shown under "Research only" in the app. Single target (tp1 = tp2 = 1h middle band).
    "EURUSD_BBH": {"broker": "EURUSDm", "cost_key": "EURUSD", "name": "EUR/USD 1h band fade (research)",
                   "strategy": "1h Bollinger fade on choppy days, target the middle band, stop 2 x 1h ATR, 24h",
                   "fn": make_signal_fn(EURUSD_BBH_SPEC, "EUR/USD 1h band fade"), "status": "RESEARCH",
                   "exit_rule": "breakeven", "max_hold": 24.0, "decision_minutes": 5,
                   "evidence": "Did NOT pass: 2004-17 988 trades, 62% won, +48.9R, PF 1.14 (needed 1.15). Paper "
                               "forward test only - do not trade it.",
                   "lookback": LAB_LOOKBACK},
    "EURUSD": {"exit_rule": "breakeven", "max_hold": 24.0, "broker": "EURUSDm", "name": "EUR/USD", "strategy": "bb_fade",
               "fn": generate_signal_bb_fade, "status": "UNPROVEN",
               "evidence": "2004-2026 both halves positive (+28.1R, +22.3R) but PF only ~1.05. "
                           "Forward test only.",
               "lookback": LOOKBACK_BARS["bb_fade"]},
    "GBPUSD": {"exit_rule": "breakeven", "max_hold": 24.0, "broker": "GBPUSDm", "name": "GBP/USD", "strategy": "bb_fade",
               "fn": generate_signal_bb_fade, "status": "UNPROVEN",
               "evidence": "2004-2026 both halves positive (+29.2R, +24.2R), PF ~1.05, and it "
                           "moves with EUR/USD so it is not a second confirmation.",
               "lookback": LOOKBACK_BARS["bb_fade"]},
    "USDJPY": {"exit_rule": "breakeven", "max_hold": 24.0, "broker": "USDJPYm", "name": "USD/JPY", "strategy": "indie_trend",
               "fn": _indie_trend, "status": "WEAK",
               "evidence": "The gold rules: LOST -9.8R in 2004-2015, made +90.8R in 2015-2026. "
                           "Half the history says no. Forward test only.",
               "lookback": LOOKBACK_BARS["indie_trend"]},
    # A second gold slot with its own journal (forward_XAUUSD_RC.csv), never the
    # live gold trader's. TP1 = TP2 = 100R is never reached, so "breakeven"
    # never moves the stop: the trade ends at the stop or after 2 hours.
    "XAUUSD_RC": {"broker": "XAUUSDm", "cost_key": "XAUUSD", "name": "XAU/USD round numbers",
                  "strategy": "round-number cascade $50, 2h exit",
                  "fn": generate_signal_round_cascade, "status": "UNPROVEN",
                  "exit_rule": "breakeven", "max_hold": ROUND_HOLD_HOURS,
                  "decision_minutes": 5, "no_target": True,
                  "evidence": "Bank stop-order cascades at $50 levels (Osler). 2004-2026: +128R over "
                              "8,253 trades but PF only 1.055, 4 of 5 periods positive — failed the "
                              "pre-registered bar. Forward test only.",
                  "lookback": LOOKBACK_BARS["round_cascade"]},
    "XAUUSD_BO4H_BIG": {"broker": "XAUUSDm", "cost_key": "XAUUSD", "name": "XAU/USD 4h breakout - big target",
                        "strategy": "same 4h breakouts, stop 1 x ATR, 2R -> stop to entry, TP 6R, out after 120h",
                        "fn": make_signal_fn(GOLD_BO4H_BIG_SPEC, "Gold 4h big target"), "status": "TEST-PASSED",
                        "exit_rule": "breakeven", "max_hold": 120.0, "decision_minutes": 240, "decision_offset": 5,
                        "evidence": "Final check 2022-04 to 2026-09: 139 trades, 24% won, +35.4R, PF 1.40, all 4 quarters "
                                    "positive, worst drop -10.8R, longest losing streak 13. Paper forward test.",
                        "lookback": LAB_LOOKBACK},
    "XAUUSD_BO4H_3R": {"broker": "XAUUSDm", "cost_key": "XAUUSD", "name": "XAU/USD 4h breakout - 3R",
                       "strategy": "same 4h breakouts, stop 2 x ATR, half at 0.75R + stop to entry, rest 3R, 120h",
                       "fn": make_signal_fn(GOLD_BO4H_3R_SPEC, "Gold 4h 3R"), "status": "TEST-PASSED",
                       "exit_rule": "partial_tp1", "max_hold": 120.0, "decision_minutes": 240, "decision_offset": 5,
                       "evidence": "2004-17 66% won PF 1.71; 2017-22 70% won PF 1.99; final 2022-26 64% won +19.5R PF 1.46, "
                                   "worst drop -6.0R (missed the win-rate-drop rule by 1 point). Paper forward test.",
                       "lookback": LAB_LOOKBACK},
    "XAUUSD_RC2": {"broker": "XAUUSDm", "cost_key": "XAUUSD", "name": "XAU/USD round numbers (improved)",
                   "strategy": "$50 cross at a new 24h extreme, TP 0.5R, stop 1.5 x 1h ATR, out after 2h",
                   "fn": make_signal_fn(RC2_SPEC, "Gold round numbers"), "status": "TEST-PASSED",
                   "exit_rule": "breakeven", "max_hold": ROUND_HOLD_HOURS, "decision_minutes": 5,
                   "evidence": "Improved and re-tested 2026-10-03: final test 2022-04 to 2026-09: 587 trades, 65% won, "
                               "+47.8R, PF 1.37, all 4 quarters positive (2017-22: 68% won, PF 1.56). Paper forward test.",
                   "lookback": LAB_LOOKBACK},
    "XAUUSD_BO4H": {"broker": "XAUUSDm", "cost_key": "XAUUSD", "name": "XAU/USD 4h breakout",
                    "strategy": "4h 20-bar breakout with the daily trend + 3 filters",
                    "fn": make_signal_fn(GOLD_BO4H_SPEC, "Gold 4h breakout"), "status": "TEST-PASSED",
                    "exit_rule": "breakeven", "max_hold": 120.0,
                    "decision_minutes": 240, "decision_offset": 5,
                    "evidence": "Round 4: passed its locked final test (2022-04 to 2026-09): 118 trades, "
                                "41.5% won, +17.7R net, PF 1.30. Stop 2 x 4h ATR, 2R -> stop to entry, "
                                "4R target, out after 120h. Paper forward test.",
                    "lookback": LAB_LOOKBACK},
    "BTCUSD_BO1H": {"broker": "BTCUSDm", "cost_key": "BTCUSD", "name": "BTC/USD 1h breakout",
                    "strategy": "1h 55-bar breakout with the daily trend",
                    "fn": make_signal_fn(BTC_BO1H_SPEC, "BTC 1h breakout"), "status": "TEST-PASSED",
                    "exit_rule": "breakeven", "max_hold": 30.0,
                    "decision_minutes": 60, "decision_offset": 5,
                    "evidence": "Round 4: passed its final test (2025-01 to 2026-09): 130 trades, 37.7% won, "
                                "+26.8R net, PF 1.39; on Binance data +7.8R. Stop 2 x 1h ATR, 2R -> stop to "
                                "entry, 4R target, out after 30h. Paper forward test.",
                    "lookback": LAB_LOOKBACK},
    # NAS100 day trade: its own engine (nas100_noise.py, run by forward_markets.NoiseMarket), not a PaperTrader —
    # it exits at half-hour checks against a band / VWAP, not at a fixed stop.
    "NAS100_NOISE": {"broker": "USTECm", "cost_key": "NAS100", "name": "NAS100 day trade",
                     "strategy": "NOISE intraday momentum: half-hour checks 10:00-15:30 New York, out by 16:00",
                     "fn": None, "kind": "noise", "status": "TEST-PASSED",
                     "decision_minutes": 30, "max_hold": 6.5, "lookback": 9000, "forward_start": "2026-10-06",
                     "evidence": "Zarattini, Aziz & Barbon 2024 'Beat the Market'. Tested 2004-2026 (Dukascopy, net of "
                                 "the Exness spread): 2,182 days, 43% of days won, +107.7% of price, PF 1.26, both halves "
                                 "positive. App engine = backtest on 2,181/2,181 days. Paper forward test.",
                     },
}

# never the live gold slot; XAUUSD_RC is a separate paper journal
FORWARD_MARKETS = ("XAUUSD_BO4H", "XAUUSD_BO4H_BIG", "XAUUSD_BO4H_3R", "BTCUSD_BO1H", "XAUUSD_RC2", "NAS100_NOISE",
                   "USDJPY_BO4H_BIG", "EURUSD_BBH", "BTCUSD", "GBPUSD")


def resolve_market_strategy(key: str):
    entry = PLAYBOOK.get(key)
    return None if entry is None else entry["fn"]
