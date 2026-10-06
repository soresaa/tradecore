"""
TRADECORE AI — Signal & Scoring Engine
========================================
Deterministic, rule-based multi-timeframe trading signal engine.

Design principles (per spec):
  - No signal is "manufactured" just because one was requested.
  - A wick alone never triggers a signal: break -> candle CLOSE -> retest -> confirmation candle.
  - Multi-timeframe conflict -> NO TRADE.
  - Confidence = strength of the detected setup, NOT a win-probability claim.
  - Everything here operates on pandas DataFrames with columns:
        ['open', 'high', 'low', 'close', 'volume']  (volume optional / may be NaN)
    indexed by a DatetimeIndex, one row per candle, ascending by time.

This module is intentionally dependency-light (pandas + numpy only) so it can
run standalone against any historical CSV before any broker/live-data
integration exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd


# =====================================================================
# 1. CANDLE FEATURES & PATTERN RECOGNITION
# =====================================================================

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds body/wick/range/direction columns used by every downstream engine."""
    out = df.copy()
    out["body"] = (out["close"] - out["open"]).abs()
    out["range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]
    out["direction"] = np.sign(out["close"] - out["open"])  # 1 bull, -1 bear, 0 doji
    out["body_pct"] = (out["body"] / out["range"]).fillna(0.0)
    return out


CANDLE_PATTERNS = [
    "bullish_engulfing", "bearish_engulfing",
    "hammer", "shooting_star",
    "morning_star", "evening_star",
    "bullish_pin_bar", "bearish_pin_bar",
    "doji", None,
]


def detect_candle_pattern(df: pd.DataFrame, i: int) -> Optional[str]:
    """Detects the pattern completed AT index i (needs features from add_candle_features)."""
    if i < 0 or i >= len(df):
        return None
    c0 = df.iloc[i]

    if c0["body_pct"] < 0.1:
        return "doji"

    # --- Engulfing (needs previous candle) ---
    if i >= 1:
        c1 = df.iloc[i - 1]
        bull_engulf = (
            c1["direction"] < 0 and c0["direction"] > 0
            and c0["close"] >= c1["open"] and c0["open"] <= c1["close"]
        )
        bear_engulf = (
            c1["direction"] > 0 and c0["direction"] < 0
            and c0["close"] <= c1["open"] and c0["open"] >= c1["close"]
        )
        if bull_engulf:
            return "bullish_engulfing"
        if bear_engulf:
            return "bearish_engulfing"

    # --- Hammer / Shooting star (single candle, small body, long opposite wick) ---
    if c0["range"] and c0["body_pct"] < 0.35:
        if c0["lower_wick"] >= 2 * c0["body"] and c0["upper_wick"] <= 0.3 * c0["body"] + 1e-9:
            return "hammer" if c0["direction"] >= 0 else "bullish_pin_bar"
        if c0["upper_wick"] >= 2 * c0["body"] and c0["lower_wick"] <= 0.3 * c0["body"] + 1e-9:
            return "shooting_star" if c0["direction"] <= 0 else "bearish_pin_bar"

    # --- Morning / Evening star (3-candle) ---
    if i >= 2:
        c2, c1 = df.iloc[i - 2], df.iloc[i - 1]
        small_middle = c1["body_pct"] < 0.3
        if (c2["direction"] < 0 and small_middle and c0["direction"] > 0
                and c0["close"] > (c2["open"] + c2["close"]) / 2):
            return "morning_star"
        if (c2["direction"] > 0 and small_middle and c0["direction"] < 0
                and c0["close"] < (c2["open"] + c2["close"]) / 2):
            return "evening_star"

    return None


BULLISH_PATTERNS = {"bullish_engulfing", "hammer", "morning_star", "bullish_pin_bar"}
BEARISH_PATTERNS = {"bearish_engulfing", "shooting_star", "evening_star", "bearish_pin_bar"}


# =====================================================================
# 2. MARKET STRUCTURE ENGINE (swings, HH/HL/LH/LL, BOS/CHoCH)
# =====================================================================

@dataclass
class Swing:
    index: int
    price: float
    kind: str          # 'high' or 'low'
    label: Optional[str] = None   # 'HH','HL','LH','LL'


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> List[Swing]:
    """Fractal swing detection: a bar is a swing high/low if it's the extreme
    within [i-left, i+right]. Requires `right` bars of confirmation (lag by design —
    a swing isn't real until it's actually held)."""
    swings: List[Swing] = []
    highs, lows = df["high"].values, df["low"].values
    n = len(df)
    for i in range(left, n - right):
        window_h = highs[i - left: i + right + 1]
        window_l = lows[i - left: i + right + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == left:
            swings.append(Swing(i, highs[i], "high"))
        if lows[i] == window_l.min() and np.argmin(window_l) == left:
            swings.append(Swing(i, lows[i], "low"))
    swings.sort(key=lambda s: s.index)
    return swings


def label_structure(swings: List[Swing]) -> List[Swing]:
    """Labels each swing HH/HL/LH/LL relative to the previous swing of the same kind."""
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None
    for s in swings:
        if s.kind == "high":
            s.label = "HH" if (last_high and s.price > last_high.price) else \
                      ("LH" if last_high else None)
            last_high = s
        else:
            s.label = "HL" if (last_low and s.price > last_low.price) else \
                      ("LL" if last_low else None)
            last_low = s
    return swings


@dataclass
class StructureState:
    bias: str                 # 'bullish' | 'bearish' | 'neutral'
    event: Optional[str]      # 'BOS' | 'CHoCH' | None
    last_swing_high: Optional[float]
    last_swing_low: Optional[float]


def structure_state(df: pd.DataFrame, swings: List[Swing]) -> StructureState:
    """Determines current trend bias and whether the most recent break was a
    BOS (continuation) or CHoCH (reversal), by comparing latest close to the
    most recent confirmed swing high/low."""
    if not swings:
        return StructureState("neutral", None, None, None)

    last_close = df["close"].iloc[-1]
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    last_high = highs[-1].price if highs else None
    last_low = lows[-1].price if lows else None

    # infer prevailing bias from the last two labeled swings
    labeled = [s for s in swings if s.label]
    prior_bias = "neutral"
    if labeled:
        prior_bias = "bullish" if labeled[-1].label in ("HH", "HL") else "bearish"

    event = None
    bias = prior_bias
    if last_high is not None and last_close > last_high:
        bias = "bullish"
        event = "BOS" if prior_bias == "bullish" else "CHoCH"
    elif last_low is not None and last_close < last_low:
        bias = "bearish"
        event = "BOS" if prior_bias == "bearish" else "CHoCH"

    return StructureState(bias, event, last_high, last_low)


# =====================================================================
# 3. LIQUIDITY ENGINE
# =====================================================================

@dataclass
class LiquidityLevel:
    price: float
    kind: str          # 'equal_highs' | 'equal_lows' | 'prev_day_high' | 'prev_day_low'


def equal_levels(swings: List[Swing], tolerance_pct: float = 0.0006) -> List[LiquidityLevel]:
    """Clusters nearby swing highs/lows into equal-high / equal-low liquidity pools."""
    levels: List[LiquidityLevel] = []
    for kind, key in (("equal_highs", "high"), ("equal_lows", "low")):
        pts = sorted([s.price for s in swings if s.kind == key])
        cluster: List[float] = []
        for p in pts:
            if cluster and abs(p - cluster[-1]) / cluster[-1] > tolerance_pct:
                if len(cluster) >= 2:
                    levels.append(LiquidityLevel(float(np.mean(cluster)), kind))
                cluster = [p]
            else:
                cluster.append(p)
        if len(cluster) >= 2:
            levels.append(LiquidityLevel(float(np.mean(cluster)), kind))
    return levels


def prev_session_levels(df: pd.DataFrame) -> List[LiquidityLevel]:
    """Previous calendar day's high/low, a classic liquidity target."""
    daily = df.groupby(df.index.date).agg(high=("high", "max"), low=("low", "min"))
    if len(daily) < 2:
        return []
    prev = daily.iloc[-2]
    return [
        LiquidityLevel(float(prev["high"]), "prev_day_high"),
        LiquidityLevel(float(prev["low"]), "prev_day_low"),
    ]


def detect_sweep(df: pd.DataFrame, level: float, direction: str, lookback: int = 20) -> bool:
    """A 'sweep' = price wicks through the level then closes back on the origin
    side within `lookback` bars — classic stop-hunt / liquidity grab."""
    window = df.iloc[-lookback:]
    if direction == "above":     # sweeping buy-side liquidity (highs)
        wicked = (window["high"] > level).any()
        closed_back = window["close"].iloc[-1] < level
        return bool(wicked and closed_back)
    else:                        # sweeping sell-side liquidity (lows)
        wicked = (window["low"] < level).any()
        closed_back = window["close"].iloc[-1] > level
        return bool(wicked and closed_back)


# =====================================================================
# 4. CONFIRMATION STATE MACHINE (break -> close -> retest -> confirm)
# =====================================================================

class ConfState(Enum):
    WAITING = "WAITING"
    BREAK_DETECTED = "BREAK_DETECTED"
    CANDLE_CLOSED = "CLOSED — analyzing"
    RETEST = "RETEST"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED — NO TRADE"


@dataclass
class ConfirmationResult:
    state: ConfState
    direction: Optional[str]     # 'buy' | 'sell'
    level: float
    confirmation_candle_index: Optional[int] = None


def run_confirmation(df: pd.DataFrame, level: float, direction: str,
                      retest_tolerance_pct: float = 0.0008,
                      max_bars: int = 30) -> ConfirmationResult:
    """
    Walks the tail of `df` looking for: close beyond level -> price returns to
    retest the level -> a rejection candle in the trade direction confirms it.
    `direction`: 'buy' (breaking above resistance) or 'sell' (breaking below support).
    A raw wick beyond the level, with no close beyond it, is NEVER enough.
    """
    tail = df.iloc[-max_bars:].reset_index(drop=True)
    state = ConfState.WAITING
    broke_at = None

    for i in range(len(tail)):
        c = tail.iloc[i]
        if state == ConfState.WAITING:
            if direction == "buy" and c["close"] > level:
                state, broke_at = ConfState.CANDLE_CLOSED, i
            elif direction == "sell" and c["close"] < level:
                state, broke_at = ConfState.CANDLE_CLOSED, i

        elif state == ConfState.CANDLE_CLOSED:
            near_level = abs(c["low" if direction == "buy" else "high"] - level) / level <= retest_tolerance_pct \
                if direction == "buy" else abs(c["high"] - level) / level <= retest_tolerance_pct
            touched = (c["low"] <= level * (1 + retest_tolerance_pct)) if direction == "buy" \
                else (c["high"] >= level * (1 - retest_tolerance_pct))
            if touched:
                state = ConfState.RETEST

        elif state == ConfState.RETEST:
            pattern = detect_candle_pattern(add_candle_features(tail), i)
            held = c["close"] > level if direction == "buy" else c["close"] < level
            rejection = pattern in (BULLISH_PATTERNS if direction == "buy" else BEARISH_PATTERNS)
            if held and (rejection or c["direction"] == (1 if direction == "buy" else -1)):
                return ConfirmationResult(ConfState.CONFIRMED, direction, level, i)
            if not held:
                return ConfirmationResult(ConfState.REJECTED, direction, level)

    return ConfirmationResult(state, direction, level,
                               broke_at if state != ConfState.WAITING else None)


# =====================================================================
# 5. INDICATORS (EMA, RSI, MACD, ATR)
# =====================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def efficiency_ratio(close: pd.Series, period: int = 20) -> float:
    """Kaufman's Efficiency Ratio: net displacement / total path length over
    the window. ~1.0 = clean trend, ~0 = pure chop. This is what was missing:
    structure and macro both answer 'which direction', never 'is there enough
    conviction here to trade at all' -- a market chopping sideways with a
    slight upward lean still passes every directional gate."""
    if len(close) < period + 1:
        return 0.0
    window = close.iloc[-period - 1:]
    net = abs(window.iloc[-1] - window.iloc[0])
    path = window.diff().abs().sum()
    return float(net / path) if path > 0 else 0.0


# =====================================================================
# 6. WEIGHTED SCORING ENGINE
# =====================================================================

WEIGHTS = {
    "structure": 18, "liquidity": 15, "breakout_retest": 16, "price_action": 12,
    "momentum": 10, "volume": 5, "htf_alignment": 14, "macro": 4,
}
MAX_SCORE = sum(WEIGHTS.values())  # 94


@dataclass
class TimeframeAnalysis:
    timeframe: str
    bias: str                       # 'bullish' | 'bearish' | 'neutral'
    structure: StructureState
    liquidity: List[LiquidityLevel]
    momentum_score: float           # -1..1
    last_pattern: Optional[str]
    df: pd.DataFrame


def analyze_timeframe(name: str, df: pd.DataFrame) -> TimeframeAnalysis:
    feat = add_candle_features(df)
    swings = label_structure(find_swings(feat))
    struct = structure_state(feat, swings)
    liq = equal_levels(swings) + prev_session_levels(feat)

    r = rsi(feat["close"]).iloc[-1]
    m_line, m_sig, m_hist = macd(feat["close"])
    macd_bias = 1 if m_hist.iloc[-1] > 0 else -1
    rsi_bias = 1 if r > 55 else (-1 if r < 45 else 0)
    ema20, ema50 = ema(feat["close"], 20).iloc[-1], ema(feat["close"], 50).iloc[-1]
    ema_bias = 1 if ema20 > ema50 else -1
    momentum_score = np.clip((macd_bias + rsi_bias + ema_bias) / 3, -1, 1)

    pattern = detect_candle_pattern(feat, len(feat) - 1)

    return TimeframeAnalysis(name, struct.bias, struct, liq, float(momentum_score), pattern, feat)


def macro_bias(daily_df: pd.DataFrame) -> str:
    """Genuine higher-timeframe filter (daily EMA20/50), independent of the 1H
    bias that already defines trade direction -- 1H checking against itself is
    tautological. Requires the daily close AND EMA20 to both sit on the same
    side of EMA50; an ambiguous daily trend returns 'neutral' rather than
    forcing a side."""
    if len(daily_df) < 55:
        return "neutral"
    e50 = ema(daily_df["close"], 50).iloc[-1]
    e20 = ema(daily_df["close"], 20).iloc[-1]
    last_close = daily_df["close"].iloc[-1]
    if last_close > e50 and e20 > e50:
        return "bullish"
    if last_close < e50 and e20 < e50:
        return "bearish"
    return "neutral"


def score_direction(direction: str, htf: TimeframeAnalysis, ltf: TimeframeAnalysis,
                     confirmation: ConfirmationResult, macro_high_impact: bool = False,
                     has_volume: bool = False, macro: Optional[str] = None) -> Tuple[int, Dict[str, int]]:
    """Scores a candidate BUY/SELL against the weighted rubric. Returns (score, breakdown)."""
    want_bias = "bullish" if direction == "buy" else "bearish"
    breakdown: Dict[str, int] = {}

    breakdown["structure"] = WEIGHTS["structure"] if ltf.bias == want_bias else \
        (WEIGHTS["structure"] // 2 if ltf.structure.event == "BOS" else 0)

    breakdown["liquidity"] = WEIGHTS["liquidity"] if ltf.liquidity else WEIGHTS["liquidity"] // 3

    breakdown["breakout_retest"] = WEIGHTS["breakout_retest"] if confirmation.state == ConfState.CONFIRMED else 0

    breakdown["price_action"] = WEIGHTS["price_action"] if ltf.last_pattern in \
        (BULLISH_PATTERNS if direction == "buy" else BEARISH_PATTERNS) else 0

    mom = ltf.momentum_score if direction == "buy" else -ltf.momentum_score
    breakdown["momentum"] = int(round(WEIGHTS["momentum"] * max(mom, 0)))

    breakdown["volume"] = WEIGHTS["volume"] if has_volume else 0

    # FIX: was `htf.bias == want_bias` where htf==1H and want_bias is DERIVED
    # from 1H's own bias -- always true by construction, never filtered anything.
    # Now scored against the genuine daily macro trend instead.
    if macro is None:
        breakdown["htf_alignment"] = WEIGHTS["htf_alignment"] if htf.bias == want_bias else 0
    else:
        breakdown["htf_alignment"] = WEIGHTS["htf_alignment"] if macro == want_bias else \
            (WEIGHTS["htf_alignment"] // 2 if macro == "neutral" else 0)

    breakdown["macro"] = 0 if macro_high_impact else WEIGHTS["macro"]

    return sum(breakdown.values()), breakdown


# =====================================================================
# 7. RISK MANAGEMENT (SL/TP)
# =====================================================================

@dataclass
class TradePlan:
    direction: str
    entry_low: float
    entry_high: float
    sl: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float


def build_trade_plan(direction: str, price: float, atr_value: float,
                      structure_level: float) -> TradePlan:
    """SL placed beyond the invalidating structure level with an ATR buffer;
    TP1 = 1R, TP2 = ~2.35R — never an arbitrary fixed point count."""
    buffer = 0.5 * atr_value
    if direction == "buy":
        sl = min(structure_level, price - atr_value) - buffer
        risk = price - sl
        tp1, tp2 = price + risk, price + risk * 2.35
    else:
        sl = max(structure_level, price + atr_value) + buffer
        risk = sl - price
        tp1, tp2 = price - risk, price - risk * 2.35

    entry_low, entry_high = min(price, price - 0.1 * atr_value), max(price, price + 0.1 * atr_value)
    return TradePlan(direction, entry_low, entry_high, sl, tp1, tp2, 1.0, round(2.35, 2))


# =====================================================================
# 8. MULTI-TIMEFRAME SIGNAL COMBINER (the entry point)
# =====================================================================

@dataclass
class Signal:
    verdict: str                    # 'BUY' | 'SELL' | 'NO TRADE'
    confidence: int                 # 0-100
    breakdown: Dict[str, int]
    reason: Optional[str]
    plan: Optional[TradePlan]
    timeframes: Dict[str, str]      # tf -> bias, for display


def generate_signal(data: Dict[str, pd.DataFrame], symbol: str = "XAU/USD",
                     macro_high_impact: bool = False) -> Signal:
    """
    data: dict of timeframe -> OHLCV DataFrame, expected keys among
          '1H' (direction), '15M' (setup), '5M' (confirmation/entry), '1M' (optional precision),
          '1D' (optional but strongly recommended -- genuine macro trend filter,
          needs 55+ daily bars of history, independent of the 1H/15M/5M lookback window)
    """
    required = ["1H", "15M", "5M"]
    missing = [tf for tf in required if tf not in data or len(data[tf]) < 60]
    if missing:
        return Signal("NO TRADE", 0, {}, f"Insufficient data for {missing}", None,
                       {tf: "?" for tf in required})

    h1 = analyze_timeframe("1H", data["1H"])
    m15 = analyze_timeframe("15M", data["15M"])
    m5 = analyze_timeframe("5M", data["5M"])
    tf_bias = {"1H": h1.bias, "15M": m15.bias, "5M": m5.bias}

    if "1M" in data and len(data["1M"]) >= 60:
        m1 = analyze_timeframe("1M", data["1M"])
        tf_bias["1M"] = m1.bias
    else:
        m1 = None

    # --- Directional agreement across 1H + 15M is required before we even look for entries ---
    if h1.bias == "neutral" or m15.bias == "neutral" or h1.bias != m15.bias:
        return Signal("NO TRADE", 0, {}, "Multi-timeframe conflict (1H/15M not aligned)",
                       None, tf_bias)

    direction = "buy" if h1.bias == "bullish" else "sell"

    # --- Trend-conviction filter: reject setups where 1H is just chopping
    # sideways with a slight lean, not genuinely trending. This is what let
    # the system keep buying a post-rally consolidation range in the
    # 2025-04..07 failure case (macro filter lagged the actual trend having
    # already ended). ---
    er = efficiency_ratio(h1.df["close"], period=20)
    tf_bias["1H trend conviction"] = f"{er:.2f}"
    if er < 0.25:
        return Signal("NO TRADE", 0, {},
                       f"1H not trending with conviction (efficiency ratio {er:.2f} < 0.25) -- likely chop/range",
                       None, tf_bias)

    # --- Genuine higher-timeframe (daily) filter: reject counter-trend entries.
    # This directly targets the diagnosed failure mode: 1H/15M flip "bullish"
    # on corrective bounces inside a larger downtrend, and vice versa. ---
    macro = None
    if "1D" in data:
        macro = macro_bias(data["1D"])
        tf_bias["1D (macro)"] = macro
        if (direction == "buy" and macro == "bearish") or (direction == "sell" and macro == "bullish"):
            return Signal("NO TRADE", 0, {},
                           f"Counter-trend: 1H/15M {direction.upper()} against daily {macro} trend",
                           None, tf_bias)

    level = m15.structure.last_swing_low if direction == "buy" else m15.structure.last_swing_high
    if level is None:
        return Signal("NO TRADE", 0, {}, "No clear structure level on 15M to confirm against",
                       None, tf_bias)

    entry_tf = m1 if m1 is not None else m5
    confirmation = run_confirmation(entry_tf.df, level, direction)

    has_volume = "volume" in entry_tf.df.columns and entry_tf.df["volume"].notna().any()
    score, breakdown = score_direction(direction, h1, entry_tf, confirmation,
                                        macro_high_impact, has_volume, macro)
    confidence = int(round(100 * score / MAX_SCORE))

    if confirmation.state != ConfState.CONFIRMED:
        return Signal("NO TRADE", confidence, breakdown,
                       f"Breakout not yet confirmed (state: {confirmation.state.value})",
                       None, tf_bias)

    if confidence < 60:
        return Signal("NO TRADE", confidence, breakdown,
                       "Setup confirmed but score below minimum threshold (60)", None, tf_bias)

    price = float(entry_tf.df["close"].iloc[-1])
    # BUG FIX: risk/SL sizing must reflect the intended holding period, not the
    # entry timeframe's own noise. Pricing stops off a 5-min ATR put them well
    # inside normal 5-min noise (~$1 on gold) -> stopped out almost immediately
    # regardless of whether the trade thesis was right. Use the 1H ATR instead,
    # since that's the timeframe that actually defines "direction" in this system.
    atr_val = float(atr(h1.df).iloc[-1])
    plan = build_trade_plan(direction, price, atr_val, level)

    return Signal("BUY" if direction == "buy" else "SELL", confidence, breakdown,
                   None, plan, tf_bias)


# =====================================================================
# 9. DISPLAY
# =====================================================================

def format_panel(symbol: str, tf_for_entry: str, signal: Signal) -> str:
    icon = {"BUY": "\U0001F7E2 BUY", "SELL": "\U0001F534 SELL", "NO TRADE": "\u26AA NO TRADE"}[signal.verdict]
    lines = [
        f"XAU/USD — {tf_for_entry}".replace("XAU/USD", symbol),
        icon,
        "",
    ]
    for tf, bias in signal.timeframes.items():
        lines.append(f"{tf:<5} {bias.upper()}")
    lines.append("")
    if signal.verdict == "NO TRADE":
        lines.append(f"Setup strength: {signal.confidence}%")
        lines.append(f"Reason: {signal.reason}")
    else:
        for k, v in signal.breakdown.items():
            lines.append(f"{k:<16} +{v}")
        lines.append(f"{'TOTAL':<16} {sum(signal.breakdown.values())}/{MAX_SCORE}  ({signal.confidence}%)")
        p = signal.plan
        lines += [
            "",
            f"Entry   {p.entry_low:,.2f}–{p.entry_high:,.2f}",
            f"SL      {p.sl:,.2f}",
            f"TP1     {p.tp1:,.2f}  (R:R {p.rr1})",
            f"TP2     {p.tp2:,.2f}  (R:R {p.rr2})",
        ]
    return "\n".join(lines)
