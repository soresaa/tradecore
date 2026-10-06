"""
Real backtest — walk-forward, on real historical data.

At every 1H close, it calls the SAME generate_signal() used for live
trading, using only the trailing lookback window of real 5-minute candles
that would actually have been available at that moment (no lookahead).
If it fires a BUY/SELL, the resulting trade plan is then simulated forward
bar-by-bar on real 5-minute candles to see whether SL, TP1, or TP2 was hit
first — that's a real trade outcome, not an assumption.

Three measurement bugs fixed 2026-09-15 (engine.py untouched). Together they
had turned a single weekend into 49 copies of the same losing trade:
  - Decisions were made at hours with no market data at all (weekends, the
    daily break, holidays), re-reading Friday's candles and filling at
    Sunday's open. A decision hour is now evaluated only if the 5-minute bar
    opening at that exact hour exists -- the same rule the live runner uses.
  - The daily macro filter saw the WHOLE current day, including a close
    hours in the future. It now sees completed days plus today's bar so far.
  - A new trade could open while the previous one was still running, and
    the 24h timeout was a bar count that stretched to ~3 days over weekends.
    Both now use real timestamps.
"""
import sys
import time
import numpy as np
import pandas as pd

from engine import generate_signal, ConfState
from trade_accounting import (MAX_HOLD_HOURS, CostModel, load_costs, r_multiple,
                              swap_cost_r, timeout_exit)

LOOKBACK_5M_BARS = 2000        # ~7 days of real 5-min data feeding 1H/15M/5M
STEP_HOURS = 1                 # re-evaluate once per hour, like a live loop


def load_real_data(path: str, point: float = None) -> pd.DataFrame:
    """Real M5 candles. If the file carries the broker's `Spread` column (in
    points, written by fetch_broker_costs.py), it is converted to price units
    and kept as `spread`, so every trade can be charged the spread that
    actually applied at that moment instead of one assumed average."""
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"], format="%Y%m%d %H:%M:%S")
    df = df.set_index("dt").sort_index()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    cols = ["open", "high", "low", "close", "volume"]
    if "Spread" in df.columns:
        df["spread"] = df["Spread"] * (point if point is not None else load_costs().point)
        cols.append("spread")
    return df[cols]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()


def daily_as_of(daily_full: pd.DataFrame, m5_upto: pd.DataFrame, ts: pd.Timestamp,
                n_days: int = 250) -> pd.DataFrame:
    """Daily candles exactly as they looked at `ts`: completed days, plus
    today's bar built only from the 5-minute candles up to `ts`.

    Slicing a pre-resampled daily series with `index <= ts` is lookahead: the
    bar labelled today 00:00 already holds today's 23:55 close. A partial
    today bar is also what MT5's live D1 feed returns mid-day, so this matches
    the live runner as well as removing the leak."""
    day0 = ts.floor("D")
    done = daily_full[daily_full.index < day0].tail(n_days - 1)
    today = m5_upto[m5_upto.index >= day0]
    if today.empty:
        return done
    row = {"open": today["open"].iloc[0], "high": today["high"].max(),
           "low": today["low"].min(), "close": today["close"].iloc[-1]}
    if "volume" in today.columns:
        row["volume"] = today["volume"].sum()
    return pd.concat([done, pd.DataFrame([row], index=[day0])])


EXIT_RULES = ("breakeven", "partial_tp1", "trail_1r", "no_breakeven")


def simulate_trade(m5: pd.DataFrame, start_idx: int, plan, costs: CostModel = None,
                   exit_rule: str = "breakeven", max_hold_hours: float = MAX_HOLD_HOURS) -> dict:
    """Walks forward on real 5-min candles from the bar after signal generation
    until SL, TP1 (then managed per `exit_rule`), TP2, or timeout.

    Two scores come back:
      `r_multiple` -- gross: filled exactly at the stop/target level, no costs.
                      Comparable with every earlier number in this project.
      `r_net`      -- what the trade would really have paid: the spread that
                      applied on the entry bar, the broker's swap per night
                      held, optional stop slippage, and -- the one fill
                      assumption that is simply wrong when a market gaps -- a
                      stop that a bar opens beyond fills at that open, not at
                      the stop. Targets still fill at the level, which is
                      conservative (a gap through a limit fills better).

    The timeout is wall-clock, like the live runner: the trade is marked at the
    close of the first bar at or after entry + max_hold_hours. The result
    carries `exit_time`, the bar the trade closed on, so the caller cannot open
    the next trade before this one has actually finished.

    `exit_rule` is the pre-registered study switch (experiments/PREREGISTRATION.md);
    live trading uses "breakeven", which is what every result so far was measured on.
    """
    if exit_rule not in EXIT_RULES:
        raise ValueError(f"unknown exit_rule {exit_rule!r}, expected one of {EXIT_RULES}")
    costs = load_costs() if costs is None else costs

    entry = (plan.entry_low + plan.entry_high) / 2
    risk = abs(entry - plan.sl)
    direction = plan.direction
    buy = direction == "buy"
    stop = plan.sl
    hit_tp1 = False
    tp1_time = None
    best = entry
    entry_time = m5.index[start_idx]
    spread_entry = (float(m5["spread"].iloc[start_idx]) if "spread" in m5.columns
                    else costs.spread_price)

    # whole seconds: a fractional-hour hold (e.g. a session end) must not create a
    # sub-second timestamp, which pandas refuses to compare with a seconds index
    timeout_at = entry_time + pd.Timedelta(seconds=round(max_hold_hours * 3600))
    end_idx = min(int(m5.index.searchsorted(timeout_at, side="left")), len(m5) - 1)

    def close_at(outcome, level_price, fill_price, j, floor_at_zero=False):
        exit_level = r_multiple(direction, entry, risk, level_price)
        exit_fill = r_multiple(direction, entry, risk, fill_price)
        if floor_at_zero:
            exit_level, exit_fill = max(exit_level, 0.0), max(exit_fill, 0.0)
        if exit_rule == "partial_tp1" and hit_tp1:
            # half banked at TP1 (a limit fill at the level), half managed on
            exit_level = 0.5 * plan.rr1 + 0.5 * exit_level
            exit_fill = 0.5 * plan.rr1 + 0.5 * exit_fill
            segments = [(1.0, entry_time, tp1_time), (0.5, tp1_time, m5.index[j])]
        else:
            segments = [(1.0, entry_time, m5.index[j])]
        swap_r = swap_cost_r(direction, risk, segments, costs)
        net = exit_fill - (spread_entry / risk if risk > 0 else 0.0) + swap_r
        return {"outcome": outcome, "r_multiple": round(exit_level, 2), "r_net": round(net, 3),
                "tp1_hit": hit_tp1, "bars_held": j - start_idx, "exit_time": m5.index[j],
                "exit_price": round(float(fill_price), 3), "spread_entry": round(spread_entry, 4),
                "swap_r": round(swap_r, 4), "exit_rule": exit_rule}

    def stop_label():
        if exit_rule == "no_breakeven" or not hit_tp1:
            return "SL"
        if exit_rule == "trail_1r" and stop != entry:
            return "TRAIL"
        return "BE"

    for j in range(start_idx, end_idx):
        bar = m5.iloc[j]
        if buy:
            if bar["low"] <= stop:
                # a bar that OPENS below the stop never traded at the stop
                fill = min(stop, float(bar["open"])) - costs.stop_slippage_price
                return close_at(stop_label(), stop, fill, j)
            if not hit_tp1 and bar["high"] >= plan.tp1:
                hit_tp1 = True
                tp1_time = m5.index[j]
                if exit_rule != "no_breakeven":
                    stop = entry            # breakeven; trail_1r starts from here
            if bar["high"] >= plan.tp2:
                return close_at("TP2", plan.tp2, plan.tp2, j)
            if exit_rule == "trail_1r" and hit_tp1:
                best = max(best, float(bar["high"]))          # updated AFTER this bar's
                stop = max(stop, best - risk)                 # stop check, never before
        else:
            if bar["high"] >= stop:
                fill = max(stop, float(bar["open"])) + costs.stop_slippage_price
                return close_at(stop_label(), stop, fill, j)
            if not hit_tp1 and bar["low"] <= plan.tp1:
                hit_tp1 = True
                tp1_time = m5.index[j]
                if exit_rule != "no_breakeven":
                    stop = entry
            if bar["low"] <= plan.tp2:
                return close_at("TP2", plan.tp2, plan.tp2, j)
            if exit_rule == "trail_1r" and hit_tp1:
                best = min(best, float(bar["low"]))
                stop = min(stop, best + risk)

    # Timeout — force-closed at the close of the first bar past the max hold.
    # This used to credit a flat +1.0R to any trade that had reached TP1, and
    # label it "BE" — a number the trade never actually made, on a label that
    # claimed it made nothing. It also silently disagreed with paper_trader.py,
    # which always marked to price. Both now call the same function.
    last_close = float(m5.iloc[end_idx]["close"])
    banked_tp1 = hit_tp1 and exit_rule != "no_breakeven"
    outcome, _ = timeout_exit(direction, entry, risk, last_close, banked_tp1)
    return close_at(outcome, last_close, last_close, end_idx, floor_at_zero=banked_tp1)


def run_backtest(m5: pd.DataFrame, start: str, end: str, symbol="XAU/USD",
                 signal_fn=None, costs: CostModel = None, exit_rule: str = "breakeven",
                 max_hold_hours: float = MAX_HOLD_HOURS):
    """`signal_fn` defaults to the validated engine; backtest_indie_variant.py
    passes the Indie logic instead so both are measured by identical machinery."""
    signal_fn = generate_signal if signal_fn is None else signal_fn
    costs = load_costs() if costs is None else costs
    window = m5[(m5.index >= pd.Timestamp(start) - pd.Timedelta(days=8)) & (m5.index <= end)]
    trades = []
    in_trade_until = None

    # Daily series for the macro filter, resampled ONCE from full history (cheap),
    # independent of the 7-day 1H/15M/5M lookback window used for entries.
    # Never slice it by time directly -- daily_as_of() explains why.
    daily_full = resample(m5[m5.index <= end], "1D")

    hourly_marks = pd.date_range(start, end, freq=f"{STEP_HOURS}h")
    t0 = time.time()
    for n, ts in enumerate(hourly_marks):
        if in_trade_until is not None and ts < in_trade_until:
            continue

        # Market-closed guard: decide only if the 5-minute bar opening at this
        # exact hour exists. Weekends, the daily break and holidays have none;
        # deciding anyway re-reads the last session's candles and "enters" at
        # a price that no longer exists, then repeats it every hour.
        if ts not in window.index:
            continue

        bars_upto = window[window.index <= ts]
        if len(bars_upto) < LOOKBACK_5M_BARS // 2:
            continue
        m5_slice = bars_upto.iloc[-LOOKBACK_5M_BARS:]
        daily_upto = daily_as_of(daily_full, bars_upto, ts)

        data = {
            "1H": resample(m5_slice, "1h"),
            "15M": resample(m5_slice, "15min"),
            "5M": m5_slice,
            "1D": daily_upto,
        }
        sig = signal_fn(data, symbol=symbol)

        if sig.verdict in ("BUY", "SELL"):
            start_pos = m5.index.searchsorted(ts, side="right")
            if start_pos >= len(m5):
                continue
            result = simulate_trade(m5, start_pos, sig.plan, costs=costs,
                                    exit_rule=exit_rule, max_hold_hours=max_hold_hours)
            result.update({"time": ts, "verdict": sig.verdict, "confidence": sig.confidence})
            trades.append(result)
            in_trade_until = result["exit_time"]   # the bar it actually closed on

        if n % 500 == 0 and n > 0:
            print(f"  ...{n}/{len(hourly_marks)} hours scanned, "
                  f"{len(trades)} trades so far, {time.time()-t0:.0f}s elapsed", file=sys.stderr)

    return pd.DataFrame(trades)


def report(trades: pd.DataFrame):
    if trades.empty:
        print("No trades were triggered in this window (setup never confirmed with sufficient score).")
        return
    for col, label in (("r_multiple", "GROSS (level fills, no costs)"),
                       ("r_net", "NET   (spread + swap + gap fills)")):
        if col not in trades.columns:
            continue
        wins = trades[trades[col] > 0]
        losses = trades[trades[col] <= 0]
        gross_win = wins[col].sum()
        gross_loss = -losses[col].sum()
        pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
        equity = trades[col].cumsum()
        print(f"{label}: {len(trades)} trades, win rate {len(wins)/len(trades)*100:.1f}%, "
              f"total {trades[col].sum():+.2f}R, profit factor {pf:.2f}, "
              f"max drawdown {(equity - equity.cummax()).min():.2f}R")
    print(f"Avg confidence:  {trades['confidence'].mean():.0f}%")
    print(f"Outcome mix:     {trades['outcome'].value_counts().to_dict()}")


if __name__ == "__main__":
    m5 = load_real_data("real_xauusd_5y.csv")
    print(f"Loaded {len(m5):,} real 5-min XAU/USD candles: {m5.index[0]} -> {m5.index[-1]}\n")

    START, END = "2024-01-01", "2024-04-01"   # 3-month real window (kept short to run in this session)
    print(f"Walk-forward backtest: {START} -> {END} (hourly re-evaluation, no lookahead)\n")
    trades = run_backtest(m5, START, END)
    print()
    report(trades)
    trades.to_csv("backtest_trades.csv", index=False)
