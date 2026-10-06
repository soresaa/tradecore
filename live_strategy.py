"""
The live default strategy: Indie entries, skipped when the daily market is choppy.

Why: a diagnosis of four different strategies' trades (2021-2026) found that
all four earned less when the daily market was moving sideways — they are all
trend/breakout systems, and breakouts in a range reverse. The fix was written
down in advance (experiments/PREREGISTRATION_FILTERS.md) and then tested only
on 2004-2021, seventeen years the diagnosis never saw:

                               net R     PF    max DD   worst year
    Indie, unfiltered         +41.04   1.06   -49.81     -20.92
    Indie + this filter       +60.71   1.17   -23.97     -12.81

It passed every pre-registered check. Full detail: experiments/RESULTS_FILTERS.md.
It is still not a proven edge (P(no edge) about 4% over those years).

The thresholds below are exactly the ones that were tested. Do not tune them:
a tuned value would no longer be the thing that passed.
"""
from __future__ import annotations

from engine import Signal
from indie_signal_port import generate_signal_indie

ER_MIN = 0.20       # below this, the daily market counts as choppy
ER_DAYS = 20


def daily_efficiency_ratio(daily) -> float | None:
    """Kaufman efficiency ratio over the last ER_DAYS COMPLETED days: net move
    divided by the total distance travelled (1.0 = straight line, ~0 = chop).

    Both the backtest and the live MT5 feed hand over daily bars ending with
    today's still-forming bar, so that bar is dropped — the value must not
    change during the day."""
    if daily is None or len(daily) < ER_DAYS + 2:
        return None
    close = daily["close"].iloc[:-1].iloc[-(ER_DAYS + 1):]
    path = close.diff().abs().sum()
    return float(abs(close.iloc[-1] - close.iloc[0]) / path) if path > 0 else 0.0


def generate_signal_indie_trend(data, symbol: str = "XAU/USD") -> Signal:
    sig = generate_signal_indie(data, symbol=symbol)
    if sig.verdict not in ("BUY", "SELL"):
        return sig
    er = daily_efficiency_ratio(data.get("1D"))
    if sig.timeframes is not None and er is not None:
        sig.timeframes["1D trend efficiency"] = f"{er:.2f}"
    if er is None or er < ER_MIN:
        shown = "n/a" if er is None else f"{er:.2f}"
        return Signal("NO TRADE", sig.confidence, sig.breakdown,
                      f"Indie {sig.verdict} skipped: daily market choppy "
                      f"(20-day efficiency {shown} < {ER_MIN})",
                      None, sig.timeframes)
    return sig
