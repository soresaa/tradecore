"""
Exact Python port of tradecore_indie.py's signal logic -- not an improved
version, the SAME simplified rules: EMA20/50 trend proxy instead of real
structure, single-close-beyond-range instead of break/retest/confirm, no
scoring, no candle patterns, no liquidity engine. This exists purely to get
an honest win-rate/profit-factor read on what the chart is actually drawing,
using the same trade-management simulation (simulate_trade) as the
validated engine so the comparison is apples to apples.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

from engine import ema, atr, efficiency_ratio, TradePlan, Signal


def generate_signal_indie(data: Dict[str, pd.DataFrame], symbol: str = "XAU/USD") -> Signal:
    required = ["1H", "15M", "5M", "1D"]
    missing = [tf for tf in required if tf not in data or len(data[tf]) < 55]
    if missing:
        return Signal("NO TRADE", 0, {}, f"Insufficient data for {missing}", None, {})

    daily, h1, m15, m5 = data["1D"], data["1H"], data["15M"], data["5M"]

    e20d, e50d = ema(daily["close"], 20).iloc[-1], ema(daily["close"], 50).iloc[-1]
    cd = daily["close"].iloc[-1]
    daily_bull = cd > e50d and e20d > e50d
    daily_bear = cd < e50d and e20d < e50d

    e20h, e50h = ema(h1["close"], 20).iloc[-1], ema(h1["close"], 50).iloc[-1]
    h1_bull, h1_bear = e20h > e50h, e20h < e50h
    h1_atr = float(atr(h1).iloc[-1])
    h1_eff = efficiency_ratio(h1["close"], period=20)

    tf_bias = {"1D": "bull" if daily_bull else ("bear" if daily_bear else "neutral"),
               "1H": "bull" if h1_bull else "bear", "1H_eff": f"{h1_eff:.2f}"}

    if h1_eff < 0.25:
        return Signal("NO TRADE", 0, {}, "1H efficiency ratio below 0.25 (chop)", None, tf_bias)

    e20_15, e50_15 = ema(m15["close"], 20).iloc[-1], ema(m15["close"], 50).iloc[-1]
    m15_bull, m15_bear = e20_15 > e50_15, e20_15 < e50_15
    m15_high = m15["high"].iloc[-16:-1].max()
    m15_low = m15["low"].iloc[-16:-1].min()

    close_now, close_prev = m5["close"].iloc[-1], m5["close"].iloc[-2]

    if h1_bull and m15_bull and not daily_bear:
        if close_now > m15_high and close_prev <= m15_high:
            sl = close_now - 1.5 * h1_atr
            risk = close_now - sl
            plan = TradePlan("buy", close_now - 0.1, close_now + 0.1, sl,
                              close_now + risk, close_now + risk * 2.35, 1.0, 2.35)
            return Signal("BUY", 70, {"indie_v1": 1}, None, plan, tf_bias)

    if h1_bear and m15_bear and not daily_bull:
        if close_now < m15_low and close_prev >= m15_low:
            sl = close_now + 1.5 * h1_atr
            risk = sl - close_now
            plan = TradePlan("sell", close_now - 0.1, close_now + 0.1, sl,
                              close_now - risk, close_now - risk * 2.35, 1.0, 2.35)
            return Signal("SELL", 70, {"indie_v1": 1}, None, plan, tf_bias)

    return Signal("NO TRADE", 0, {}, "No aligned breakout", None, tf_bias)
