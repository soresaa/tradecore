"""
Shared trade-accounting rules for TRADECORE AI.

backtest.py and paper_trader.py both have to answer the same question --
"this position is closing at a known price, what R did it make?" -- and they
answered it differently. The backtest credited a flat +1.0R to any trade that
reached TP1 and then timed out (and labelled it "BE", a breakeven outcome
paying a full winning R), while the paper trader marked to the real price.
Same event, two numbers: forward results could never be honestly compared to
backtest results, and the gap would have looked like "live underperforms
backtest" when it was really just two different formulas.

This module is the single definition of those rules. Both runners import from
here so they cannot drift apart again.

engine.py is frozen and deliberately untouched -- this is trade accounting,
not signal logic, and none of it changes what the engine decides to trade.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

# Max holding period per trade, in wall-clock hours. Both runners measure it
# the same way: the live runner compares timestamps, and the backtest marks the
# trade at the first bar at or after entry + MAX_HOLD_HOURS. The backtest used
# to count MAX_HOLD_5M_BARS bars instead, which is only 24h on gap-free data --
# across a weekend 288 bars is ~3 days, a hold the live runner never allows.
MAX_HOLD_HOURS = 24
MAX_HOLD_5M_BARS = MAX_HOLD_HOURS * 12      # bar count on gap-free data only; not used for exits


def r_multiple(direction: str, entry: float, risk: float, exit_price: float) -> float:
    """Realized R for a position closed at `exit_price`. Positive = profit.

    `risk` is the absolute entry-to-stop distance, i.e. what 1R is worth on
    this specific trade -- R is always relative to the risk actually taken,
    never to a fixed point count.
    """
    if risk <= 0:
        raise ValueError(f"risk must be a positive entry-to-stop distance, got {risk!r}")
    gain = (exit_price - entry) if direction == "buy" else (entry - exit_price)
    return gain / risk


def timeout_exit(direction: str, entry: float, risk: float, exit_price: float,
                 tp1_hit: bool) -> tuple[str, float]:
    """Score a position force-closed at `exit_price` for exceeding the max
    holding period without reaching its stop or TP2.

    Marked to the actual price, never to a nominal target. A trade that banked
    TP1 and then drifted sideways for 24h is worth whatever it is worth at the
    close -- somewhere between breakeven and TP2 -- not a flat 1R.

    Once TP1 is hit the stop sits at breakeven, so the downside is floored at
    0R: any move back through entry would have closed the position at
    breakeven before the timeout could fire. In the backtest that floor is
    unreachable by construction (the bar loop returns on any touch of the
    stop); in live polling it is a real guard, since price can be observed
    below entry on a poll that also happens to be past the timeout.

    Returns (outcome_label, r_multiple rounded to 2dp).
    """
    r = r_multiple(direction, entry, risk, exit_price)
    if tp1_hit:
        r = max(r, 0.0)
    return ("TIMEOUT_AFTER_TP1" if tp1_hit else "TIMEOUT"), round(r, 2)


# =====================================================================
# WHAT A TRADE ACTUALLY COSTS
# =====================================================================
# Everything here is in PRICE units per 1 unit of volume (per ounce for gold),
# so a cost is directly comparable with the stop distance: cost / risk = cost
# in R. Values come from the broker via fetch_broker_costs.py, not from
# guesses, and both runners load the same file so they cannot disagree.

COSTS_FILE = "broker_costs.json"


@dataclass(frozen=True)
class CostModel:
    spread_price: float = 0.0           # fallback when a bar carries no spread
    swap_long_per_night: float = 0.0    # negative = you pay
    swap_short_per_night: float = 0.0
    triple_swap_weekday: int = 2        # pandas weekday (Monday=0); Wednesday by default
    stop_slippage_price: float = 0.0    # extra adverse price on stop exits
    point: float = 0.001
    source: str = "zero"

    @classmethod
    def zero(cls) -> "CostModel":
        return cls()

    @classmethod
    def from_json(cls, path: str = COSTS_FILE, stop_slippage_price: float = 0.0) -> "CostModel":
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        model = snap.get("swap_model", {})
        long_night = short_night = 0.0
        if model.get("kind") == "price_per_unit":
            long_night, short_night = float(model["long"]), float(model["short"])
        return cls(
            spread_price=float(snap.get("spread_history", {}).get("median_price", 0.0)),
            swap_long_per_night=long_night,
            swap_short_per_night=short_night,
            triple_swap_weekday=int(snap.get("triple_swap_weekday_pandas", 2)),
            stop_slippage_price=stop_slippage_price,
            point=float(snap.get("point", 0.001)),
            source=f"{path} ({snap.get('fetched_at', 'unknown date')})"
            + ("" if model.get("kind") == "price_per_unit"
               else f" -- swap mode {model.get('kind')!r} not convertible, swap treated as 0"),
        )

    def swap_per_night(self, direction: str) -> float:
        return self.swap_long_per_night if direction == "buy" else self.swap_short_per_night

    def describe(self) -> str:
        return (f"spread ${self.spread_price:.3f} (fallback), swap/night long "
                f"${self.swap_long_per_night:+.3f} short ${self.swap_short_per_night:+.3f}, "
                f"triple on weekday {self.triple_swap_weekday}, stop slippage "
                f"${self.stop_slippage_price:.3f} [{self.source}]")


def load_costs(path: str = COSTS_FILE, stop_slippage_price: float = 0.0) -> CostModel:
    """Broker costs if fetch_broker_costs.py has been run, otherwise zero costs
    (so nothing breaks, and a report that shows no costs says so explicitly)."""
    if os.path.exists(path):
        return CostModel.from_json(path, stop_slippage_price)
    return CostModel(stop_slippage_price=stop_slippage_price, source="no broker_costs.json -- costs treated as 0")


def rollover_nights(entry_time, exit_time, triple_swap_weekday: int = 2) -> float:
    """Swap-charging nights between two timestamps: one per server midnight
    crossed, none for a midnight that ends a Saturday or Sunday (the market is
    shut), and three on the broker's 3-day swap weekday, which is how brokers
    charge the weekend in advance."""
    t0, t1 = pd.Timestamp(entry_time), pd.Timestamp(exit_time)
    if t1 <= t0:
        return 0.0
    nights = 0.0
    midnight = t0.normalize() + pd.Timedelta(days=1)
    while midnight <= t1:
        ended = (midnight - pd.Timedelta(days=1)).weekday()   # the day that just finished
        if ended < 5:
            nights += 3.0 if ended == triple_swap_weekday else 1.0
        midnight += pd.Timedelta(days=1)
    return nights


def swap_cost_r(direction: str, risk: float, segments: Sequence[Tuple[float, object, object]],
                costs: CostModel) -> float:
    """Swap in R for a position held over `segments` -- each (fraction_of_position,
    start, end), so a partial close stops paying swap on the part it closed.
    Negative = a cost."""
    per_night = costs.swap_per_night(direction)
    if not per_night or risk <= 0:
        return 0.0
    total = 0.0
    for fraction, start, end in segments:
        if start is None or end is None:
            continue
        total += fraction * rollover_nights(start, end, costs.triple_swap_weekday) * per_night
    return total / risk
