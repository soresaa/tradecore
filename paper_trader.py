"""
Paper trading runner for TRADECORE AI.

This does NOT place real orders. It runs the frozen engine (engine.py, as
validated in the backtests) against a live or replayed data feed, tracks one
open paper position at a time exactly like the backtest did, and journals
every signal and outcome to a CSV. Point `report()` at that journal any time
to get a running scoreboard -- this is your live version of the "Signal
history" panel from the original spec.

THE ENGINE ITSELF IS FROZEN. Nothing in engine.py should change based on
paper-trading results until there's enough forward data to draw a real
conclusion -- same discipline as the backtest out-of-sample rule.

Two data feeds are provided:
  - ReplayDataFeed: walks forward through a historical CSV. Useful to prove
    the runner logic itself works, and doubles as the "Replay mode" feature
    from the original spec. NOT a validation run -- it's replaying data,
    not predicting anything new.
  - LiveDataFeed (template): a stub you fill in with your actual broker/data
    provider. This sandbox has no network access to any broker or market
    data API, so this class is documented but not runnable here -- it has
    to run on your machine, with your own account.

When decisions happen (fixed 2026-09-15): once per hour, on the 5-minute
candle that opens at HH:00, as soon as it has CLOSED -- exactly the data the
backtest decides on. Polling more often only manages an open position faster;
it no longer re-runs the engine on half-formed candles every few seconds,
which was a different system from the one that was validated. Nothing is
decided, and no open position is managed, while the market is closed.
"""
from __future__ import annotations

import csv
import json
import os
import time as time_module
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from engine import generate_signal, ConfState
from trade_accounting import (MAX_HOLD_HOURS, CostModel, load_costs, r_multiple,
                              swap_cost_r, timeout_exit)

LOOKBACK_5M_BARS = 2000

# Which entry logic the runner uses. All are measured by the same backtest, the
# same trade management and the same costs, so the choice only changes WHEN a
# trade is entered.
#   indie_trend -- Indie entries, skipped when the daily market is choppy
#                  (live_strategy.py). Chosen by a pre-registered test on
#                  2004-2021, years the filter idea never saw: +60.71R, PF 1.17,
#                  max drawdown -23.97R, against unfiltered Indie's +41.04R,
#                  PF 1.06, -49.81R. See experiments/RESULTS_FILTERS.md.
#   indie       -- the unfiltered Indie logic (2004-2026: +89.41R, PF 1.10).
#   engine      -- the original engine (2021-2026: +21.50R, PF 1.05).
# None of these is a proven edge.
STRATEGY_NAMES = ("indie_trend", "indie", "engine")
DEFAULT_STRATEGY = "indie_trend"


def resolve_strategy(name):
    """Name -> signal function. Unknown names fall back to the default rather
    than trading something the caller did not ask for."""
    if name == "engine":
        return generate_signal
    if name == "indie":
        from indie_signal_port import generate_signal_indie
        return generate_signal_indie
    if name != DEFAULT_STRATEGY:
        print(f"unknown strategy {name!r}, using {DEFAULT_STRATEGY!r}")
    from live_strategy import generate_signal_indie_trend
    return generate_signal_indie_trend
JOURNAL_COLUMNS = [
    "time_opened", "direction", "entry", "sl", "tp1", "tp2", "confidence",
    "breakdown", "status", "tp1_hit", "time_closed", "outcome", "r_multiple",
    # What the trade really paid, so the forward test MEASURES its costs
    # instead of assuming them: the side of the quote each fill actually took,
    # the swap for the nights it was held, and the resulting net R. Gross
    # `r_multiple` is unchanged, so it stays comparable with the backtest.
    "bar_opened", "fill_entry", "fill_exit", "swap_r", "r_net",
    # Where a trailing stop has ratcheted to, for strategies that use one
    # (BTC/USD). Blank for every strategy that does not -- gold never writes it.
    "trail_sl",
]


def _num(value, default: float) -> float:
    """Journal cells come back as strings, NaN or numbers depending on how the
    row was written; missing means 'not recorded', not zero."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f


# =====================================================================
# DATA FEED INTERFACE
# =====================================================================

class DataFeed(ABC):
    """Whatever the real data source is, it needs to answer these four
    questions. Implement this once for your broker/provider and everything
    else (signal generation, position management, journaling) just works."""

    @abstractmethod
    def now(self) -> pd.Timestamp: ...

    @abstractmethod
    def get_recent_5m(self, n_bars: int) -> pd.DataFrame:
        """Last n_bars of CLOSED 5-minute OHLCV candles, ending with the one
        latest_closed_bar_time() reports. Columns: open, high, low, close, volume."""
        ...

    @abstractmethod
    def latest_closed_bar_time(self) -> Optional[pd.Timestamp]:
        """Open time of the most recent CLOSED 5-minute candle. The runner
        decides once per hour, when this is the HH:00 candle."""
        ...

    @abstractmethod
    def get_daily(self, n_days: int) -> pd.DataFrame:
        """Last n_days of daily OHLCV candles up to `now()`, for the macro filter."""
        ...

    @abstractmethod
    def current_price(self) -> float: ...

    def current_quote(self) -> tuple:
        """(bid, ask). Triggers still use current_price(), the same side the
        backtest's candles are on, but a real fill pays the other side -- so
        this is what the journal records to measure the spread actually paid.
        A feed with no ask reports no spread."""
        price = self.current_price()
        return price, price

    def data_is_fresh(self) -> bool:
        """False while the market is closed (weekend, daily break, holiday) or
        the feed has stalled. Nothing is decided or managed on stale data."""
        return True


class ReplayDataFeed(DataFeed):
    """Walks forward through a historical CSV bar-by-bar, pretending each
    bar as it's reached is 'now'. Used to test the runner and as the
    Replay-mode feature -- not a substitute for a real forward test."""

    def __init__(self, csv_path: str, start: str, end: str):
        raw = pd.read_csv(csv_path)
        raw["dt"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"], format="%Y%m%d %H:%M:%S")
        raw = raw.set_index("dt").sort_index()
        raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                   "Close": "close", "Volume": "volume"})
        cols = ["open", "high", "low", "close", "volume"]
        if "Spread" in raw.columns:                  # real spread per bar, if fetched
            raw["spread"] = raw["Spread"] * load_costs().point
            cols.append("spread")
        self.full = raw[cols]
        self.daily_full = (self.full["close"].resample("1D").last().to_frame("close")
                            .join(self.full["open"].resample("1D").first().rename("open"))
                            .join(self.full["high"].resample("1D").max().rename("high"))
                            .join(self.full["low"].resample("1D").min().rename("low")).dropna())
        self.cursor_positions = self.full.index[
            (self.full.index >= start) & (self.full.index <= end)]
        self._i = 0

    def now(self) -> pd.Timestamp:
        return self.cursor_positions[self._i]

    def advance(self) -> bool:
        self._i += 1
        return self._i < len(self.cursor_positions)

    def get_recent_5m(self, n_bars: int) -> pd.DataFrame:
        upto = self.full[self.full.index <= self.now()]
        return upto.iloc[-n_bars:]

    def latest_closed_bar_time(self) -> pd.Timestamp:
        return self.now()     # replay treats the bar it has reached as closed

    def get_daily(self, n_days: int) -> pd.DataFrame:
        # Completed days + today's bar so far. Slicing daily_full by time would
        # hand the replay today's future close -- see backtest.daily_as_of.
        from backtest import daily_as_of
        return daily_as_of(self.daily_full, self.full[self.full.index <= self.now()],
                           self.now(), n_days)

    def current_price(self) -> float:
        return float(self.full.loc[self.now(), "close"])

    def current_quote(self) -> tuple:
        bid = self.current_price()
        if "spread" not in self.full.columns:
            return bid, bid
        return bid, bid + float(self.full.loc[self.now(), "spread"])


def find_mt5_terminal() -> Optional[str]:
    """Locate terminal64.exe so the app can start MT5 by itself instead of
    requiring you to open it first. Returns None if MT5 isn't installed."""
    import glob
    patterns = [
        r"C:\Program Files\MetaTrader 5*\terminal64.exe",
        r"C:\Program Files\*MetaTrader*\terminal64.exe",
        r"C:\Program Files\*MT5*\terminal64.exe",
        r"C:\Program Files (x86)\*MetaTrader*\terminal64.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\*MetaTrader*\terminal64.exe"),
        os.path.expandvars(r"%APPDATA%\*MetaTrader*\terminal64.exe"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


class MT5DataFeed(DataFeed):
    """
    Real MetaTrader5 implementation. Windows only, on the machine where the
    MT5 terminal is installed.

    You do NOT have to open MT5 first -- if it isn't running, this launches it
    and waits for it to log in. That only works when the launching process is
    itself unrestricted: a terminal started by a sandboxed/restricted parent
    fails to start its IPC dispatcher ("IPC dispatcher not started" in the
    terminal log) and the Python API can never attach, even though the terminal
    looks perfectly healthy and logged in on screen.

    symbol defaults to "XAUUSDm" -- check YOUR broker's Market Watch for the
    exact string. Many brokers suffix symbols (m, c, z, r...); Exness-MT5Trial9
    uses "m". A wrong suffix fails quietly: symbol_info_tick returns None
    rather than raising, which is why __init__ checks for it explicitly.
    """

    def __init__(self, symbol: str = "XAUUSDm", auto_launch: bool = True,
                 launch_timeout: int = 150, status=None):
        import MetaTrader5 as mt5
        self.mt5 = mt5
        self.symbol = symbol
        self._last_tick_msc = None
        self._last_tick_change = float("-inf")   # monotonic time a NEW tick was last seen
        self._stale_recheck_at = 0.0

        def say(msg):
            if status:
                try:
                    status(msg)
                except Exception:
                    pass
            print(msg, flush=True)

        if not mt5.initialize():
            if not auto_launch:
                raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}. "
                                    f"Is the MT5 terminal open and logged in?")

            path = find_mt5_terminal()
            if path is None:
                raise RuntimeError(
                    "MetaTrader 5 is not installed on this machine (no "
                    "terminal64.exe found). Install MT5 and log into your "
                    "account once, then this app can start it for you.")

            say(f"starting MetaTrader 5 ({os.path.basename(os.path.dirname(path))})...")
            deadline = time_module.time() + launch_timeout
            last_err = None
            attempt = 0
            while time_module.time() < deadline:
                attempt += 1
                if mt5.initialize(path=path):
                    break
                last_err = mt5.last_error()
                # The terminal needs time to start AND authorize with the
                # broker; initialize() fails until then. Keep retrying rather
                # than giving up on the first miss.
                say(f"waiting for MetaTrader 5 to be ready (attempt {attempt})...")
                time_module.sleep(5)
            else:
                raise RuntimeError(
                    f"MetaTrader 5 did not become ready within {launch_timeout}s "
                    f"(last error: {last_err}). If MT5 opened but asks you to log "
                    f"in, log in once manually -- it remembers after that.")

        acct = mt5.account_info()
        if acct is None:
            mt5.shutdown()
            raise RuntimeError(
                "MT5 is running but not logged into an account. Open it, log "
                "into your demo account once, and tick 'Save password' -- after "
                "that this app can start it unattended.")
        say(f"MT5 ready: account {acct.login} @ {acct.server}")

        if not mt5.symbol_select(symbol, True):
            mt5.shutdown()
            raise RuntimeError(
                f"Symbol '{symbol}' not found/selectable. Open Market Watch "
                f"(Ctrl+M) in MT5 and check the EXACT symbol name for gold on "
                f"your account -- it may have a different suffix than 'm'.")

        # After a cold start the first ticks can take a moment to arrive.
        tick = None
        for _ in range(10):
            tick = mt5.symbol_info_tick(symbol)
            self._observe_tick(tick)
            if tick is not None and tick.bid:
                break
            time_module.sleep(1)
        if tick is None:
            mt5.shutdown()
            raise RuntimeError(f"symbol_info_tick('{symbol}') returned None -- "
                                f"symbol selected but no live tick. Check the "
                                f"market is open (gold is closed at weekends).")
        say(f"MT5DataFeed connected: {symbol} @ {tick.bid}")

    FRESH_TICK_SECONDS = 120    # gold ticks many times a minute whenever it trades

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def _observe_tick(self, tick):
        if tick is None:
            return
        if self._last_tick_msc is None:           # first sighting proves nothing about liveness
            self._last_tick_msc = tick.time_msc
        elif tick.time_msc != self._last_tick_msc:
            self._last_tick_msc = tick.time_msc
            self._last_tick_change = time_module.monotonic()

    def data_is_fresh(self, sample_seconds: int = 10) -> bool:
        """True only if a NEW tick has arrived within FRESH_TICK_SECONDS.

        Deliberately timezone-free: MT5 stamps ticks in broker server time,
        which is not UTC on every broker, so comparing tick.time with the
        clock is unreliable. Watching whether ticks are still arriving isn't.
        With nothing seen recently it watches for up to `sample_seconds`; a
        closed verdict is then trusted for 60s so a weekend doesn't cost a
        10-second wait on every poll. current_price() feeds the same tracker,
        so a reopening market is noticed on the next price read."""
        now = time_module.monotonic()
        if now - self._last_tick_change <= self.FRESH_TICK_SECONDS:
            return True
        if now < self._stale_recheck_at:
            return False
        deadline = now + sample_seconds
        while time_module.monotonic() < deadline:
            self._observe_tick(self.mt5.symbol_info_tick(self.symbol))
            if time_module.monotonic() - self._last_tick_change <= self.FRESH_TICK_SECONDS:
                return True
            time_module.sleep(1)
        self._stale_recheck_at = time_module.monotonic() + 60
        return False

    def latest_closed_bar_time(self) -> Optional[pd.Timestamp]:
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M5, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return pd.Timestamp(int(rates[0]["time"]), unit="s")

    def get_recent_5m(self, n_bars: int) -> pd.DataFrame:
        # Start at position 1: position 0 is the candle still forming. The
        # backtest only ever decides on closed candles, so live must too.
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M5, 1, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned no M5 data for {self.symbol}: "
                                f"{self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.set_index("time").rename(
            columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]

    def get_daily(self, n_days: int) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_D1, 0, n_days)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned no D1 data for {self.symbol}: "
                                f"{self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df.set_index("time")[["open", "high", "low", "close"]]

    def current_price(self) -> float:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick('{self.symbol}') returned None mid-run "
                                f"-- connection to terminal may have dropped.")
        self._observe_tick(tick)
        return float(tick.bid)

    def current_quote(self) -> tuple:
        tick = self.mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick('{self.symbol}') returned None mid-run "
                                f"-- connection to terminal may have dropped.")
        self._observe_tick(tick)
        return float(tick.bid), float(tick.ask)


# =====================================================================
# JOURNAL
# =====================================================================

class Journal:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(JOURNAL_COLUMNS)
            return
        # A journal written before the cost columns existed is upgraded in
        # place: new columns are added empty and existing rows keep their data,
        # so a forward test already under way is never thrown away.
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in JOURNAL_COLUMNS if c not in header]
        if missing:
            df = pd.read_csv(path)
            for col in missing:
                df[col] = ""
            df[JOURNAL_COLUMNS].to_csv(path, index=False)

    def open_trade(self, ts, direction, plan, confidence, breakdown,
                   bar_opened=None, fill_entry=None):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                ts, direction, (plan.entry_low + plan.entry_high) / 2, plan.sl, plan.tp1, plan.tp2,
                confidence, json.dumps(breakdown), "open", 0, "", "", "",
                bar_opened if bar_opened is not None else "",
                fill_entry if fill_entry is not None else "", "", "", "",
            ])

    def _read(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def has_open_position(self) -> bool:
        df = self._read()
        return bool(len(df) and (df["status"] == "open").any())

    def get_open_position(self) -> Optional[dict]:
        df = self._read()
        open_rows = df[df["status"] == "open"]
        return None if open_rows.empty else open_rows.iloc[-1].to_dict()

    def update_trail(self, level: float):
        """Ratchets the open trade's trailing stop. Persisted, so a restart
        cannot silently hand the trade a looser stop than it had."""
        df = self._read()
        if df.empty:
            return
        idx = df.index[df["status"] == "open"]
        if len(idx) == 0:
            return
        df.loc[idx[-1], "trail_sl"] = round(float(level), 5)
        df.to_csv(self.path, index=False)

    def mark_tp1_hit(self):
        df = self._read().astype(object)
        idx = df[df["status"] == "open"].index[-1]
        df.loc[idx, "tp1_hit"] = 1
        df.to_csv(self.path, index=False)

    def close_last_open(self, ts, outcome, r_multiple, fill_exit=None, swap_r=None, r_net=None):
        df = self._read().astype(object)
        idx = df[df["status"] == "open"].index[-1]
        df.loc[idx, ["status", "time_closed", "outcome", "r_multiple"]] = \
            ["closed", str(ts), outcome, r_multiple]
        for col, value in (("fill_exit", fill_exit), ("swap_r", swap_r), ("r_net", r_net)):
            if value is not None:
                df.loc[idx, col] = value
        df.to_csv(self.path, index=False)

    def report(self):
        df = self._read()
        closed = df[df["status"] == "closed"]
        print(f"Journal: {self.path}")
        print(f"Total signals: {len(df)}  |  Open: {(df['status']=='open').sum()}  "
              f"|  Closed: {len(closed)}")
        if closed.empty:
            return
        for col, label in (("r_multiple", "GROSS"), ("r_net", "NET  ")):
            if col not in closed.columns:
                continue
            r = pd.to_numeric(closed[col], errors="coerce").dropna()
            if r.empty:
                continue
            wins, losses = r[r > 0], r[r <= 0]
            gl = -losses.sum()
            pf = (wins.sum() / gl) if gl > 0 else float("inf")
            print(f"{label}: win rate {len(wins)/len(r)*100:.1f}%  |  {r.sum():+.2f}R  "
                  f"|  profit factor {pf:.2f}  ({len(r)} trades)")


# =====================================================================
# PAPER TRADER
# =====================================================================

def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    agg = {"open": o, "high": h, "low": l, "close": c}
    if "volume" in df.columns:
        agg["volume"] = df["volume"].resample(rule).sum()
    return pd.DataFrame(agg).dropna()


class PaperTrader:
    def __init__(self, feed: DataFeed, journal_path: str, symbol: str = "XAU/USD",
                 on_signal=None, costs: CostModel = None, signal_fn=None):
        """`on_signal`, if given, is called with the full Signal object every
        time one is generated -- including NO TRADE, which carries the reason
        the engine held back. Purely an observation hook for UIs: it is called
        AFTER the trading decision is made, its return value is ignored, and
        any exception it raises is swallowed, so a broken display can never
        affect what gets traded or journaled."""
        self.feed = feed
        self.journal = Journal(journal_path)
        self.symbol = symbol
        self.on_signal = on_signal
        self._last_decision_bar = None
        # Same broker costs the backtest prices trades with, so a forward
        # result and a backtest result mean the same thing.
        self.costs = load_costs() if costs is None else costs
        # Entry logic. Defaults to the engine so existing callers are unchanged;
        # the app passes resolve_strategy(settings["strategy"]).
        self.signal_fn = generate_signal if signal_fn is None else signal_fn
        # How many 5-minute candles the strategy gets to look back on. The gold
        # strategy was measured on 2,000 (about 7 days) and keeps that number;
        # the volatility-squeeze strategy reads a 20-day window and is given
        # more by the caller. Changing the default would change what gold trades.
        self.lookback_5m = LOOKBACK_5M_BARS
        # How an open trade is managed. "breakeven" is what gold was measured
        # with and stays the default; "trail_1r" is the Bitcoin strategy that
        # passed the sealed holdout (experiments/RESULTS_DEEP.md), where the
        # stop follows 1R behind the best price once the first target is hit.
        self.exit_rule = "breakeven"
        self.max_hold_hours = MAX_HOLD_HOURS
        # How often a NEW entry may be decided, in minutes. Every tested strategy
        # decides on the closed HH:00 candle (60) and keeps that default; the gold
        # round-number cascade reacts to any closed 5-minute candle (5), because
        # that is how it was tested (experiments/RESULTS_EDGES.md, section 3).
        self.decision_minutes = 60
        # Minutes added to a candle's open time before that check. 0 = decide on the
        # FIRST candle of the period (HH:00, as every older strategy was tested). The
        # round-4/5 timeframe strategies (lab_live.py) decide on the LAST candle of
        # their 1h / 4h bar (HH:55, 03:55), so they use 5.
        self.decision_offset_minutes = 0

    def _close_position(self, pos, outcome, level_price, floor_at_zero=False):
        """Scores and journals a close exactly once, two ways: `r_multiple` as
        before (marked at the observed price, comparable with the backtest) and
        `r_net`, which uses the side of the quote a real fill would take plus
        the swap for the nights held."""
        direction = pos["direction"]
        entry, sl = float(pos["entry"]), float(pos["sl"])
        risk = abs(entry - sl)
        bid, ask = self.feed.current_quote()
        fill_exit = bid if direction == "buy" else ask
        fill_entry = _num(pos.get("fill_entry"), entry)

        gross = r_multiple(direction, entry, risk, level_price)
        net_exit = r_multiple(direction, fill_entry, risk, fill_exit)
        if floor_at_zero:            # the breakeven stop floors the downside at 0
            gross, net_exit = max(gross, 0.0), max(net_exit, 0.0)
        # partial_tp1 (the gold 4h 3R version): HALF was banked at TP1 (a limit fill at the level, as the
        # backtest scores it), the other half closes here -- the same 0.5 / 0.5 split simulate_trade uses
        if self.exit_rule == "partial_tp1" and bool(int(_num(pos.get("tp1_hit"), 0))):
            tp1 = float(pos["tp1"])
            gross = 0.5 * r_multiple(direction, entry, risk, tp1) + 0.5 * gross
            net_exit = 0.5 * r_multiple(direction, fill_entry, risk, tp1) + 0.5 * net_exit

        # Swap is counted on the feed's own bar clock (broker server time), the
        # same clock the backtest counts rollovers on.
        swap_r, bar_opened, bar_now = 0.0, pos.get("bar_opened"), self.feed.latest_closed_bar_time()
        if bar_now is not None and isinstance(bar_opened, str) and bar_opened:
            swap_r = swap_cost_r(direction, risk, [(1.0, bar_opened, bar_now)], self.costs)

        gross, net = round(gross, 2), round(net_exit + swap_r, 3)
        self.journal.close_last_open(self.feed.now(), outcome, gross,
                                     fill_exit=round(fill_exit, 3),
                                     swap_r=round(swap_r, 4), r_net=net)
        return gross, net

    def _check_open_position(self):
        pos = self.journal.get_open_position()
        if pos is None:
            return
        if not self.feed.data_is_fresh():
            return   # market closed: no real price to stop out, take profit or time out at
        price = self.feed.current_price()
        direction = pos["direction"]
        entry, sl, tp1, tp2 = pos["entry"], pos["sl"], pos["tp1"], pos["tp2"]
        tp1_hit = bool(int(pos["tp1_hit"]))
        risk = abs(entry - sl)
        trailing = self.exit_rule == "trail_1r"
        trail_sl = _num(pos.get("trail_sl"), entry)
        if not tp1_hit:
            effective_sl = sl                    # first target not reached yet
        elif trailing:
            # once TP1 is hit the stop follows 1R behind the best price, and
            # never sits worse than breakeven
            effective_sl = max(entry, trail_sl) if direction == "buy" else min(entry, trail_sl)
        else:
            effective_sl = entry                 # TP1 locks the stop to breakeven

        hit_stop = (price <= effective_sl) if direction == "buy" else (price >= effective_sl)
        hit_tp2 = (price >= tp2) if direction == "buy" else (price <= tp2)
        hit_tp1 = (price >= tp1) if direction == "buy" else (price <= tp1)

        # Max hold, same as the validated backtest -- without this, a forward
        # position can run indefinitely, which was never tested. The limit
        # itself lives in trade_accounting so the backtest's bar count and this
        # wall-clock check can't drift apart.
        opened_at = pd.Timestamp(pos["time_opened"])
        timed_out = (self.feed.now() - opened_at) >= pd.Timedelta(hours=self.max_hold_hours)

        # Realized R is computed from the actual observed price, not the ideal
        # stop/target level -- unlike backtest.py's simulate_trade (which
        # assumes a perfect fill exactly at the stop/target, a standard
        # backtest simplification). Live polling means price can move past a
        # level between checks, so this deliberately includes that slippage
        # rather than pretending it away. Expect paper results to run
        # slightly worse than the backtest numbers for this reason -- that's
        # real-world friction, not a bug.
        if hit_stop:
            outcome = "SL"
            if tp1_hit:
                outcome = "TRAIL" if (trailing and effective_sl != entry) else "BE"
            gross, net = self._close_position(pos, outcome, price)
            self._skip_candle_closed_during_trade()
            print(f"[{self.feed.now()}] CLOSED {direction.upper()} @ {outcome}  "
                  f"({gross:+.2f}R gross, {net:+.2f}R net)")
        elif hit_tp2:
            gross, net = self._close_position(pos, "TP2", price)
            self._skip_candle_closed_during_trade()
            print(f"[{self.feed.now()}] CLOSED {direction.upper()} @ TP2 "
                  f"({gross:+.2f}R gross, {net:+.2f}R net)")
        elif not tp1_hit and hit_tp1:
            self.journal.mark_tp1_hit()
            if trailing:
                self.journal.update_trail(entry)
            print(f"[{self.feed.now()}] {direction.upper()} hit TP1 -- stop moved to "
                  f"{'the 1R trail' if trailing else 'breakeven'}")
        elif timed_out:
            outcome, _ = timeout_exit(direction, entry, risk, price, tp1_hit)
            gross, net = self._close_position(pos, outcome, price, floor_at_zero=tp1_hit)
            print(f"[{self.feed.now()}] CLOSED {direction.upper()} @ {outcome} "
                  f"({self.max_hold_hours}h timeout, {gross:+.2f}R gross, {net:+.2f}R net)")
        elif trailing and tp1_hit:
            # ratchet last, never before the stop check -- exactly the order
            # backtest.simulate_trade uses for exit_rule="trail_1r"
            pulled = (price - risk) if direction == "buy" else (price + risk)
            better = pulled > effective_sl if direction == "buy" else pulled < effective_sl
            if better:
                self.journal.update_trail(pulled)
                print(f"[{self.feed.now()}] {direction.upper()} trail moved to {pulled:.5f}")

    def _skip_candle_closed_during_trade(self):
        """A stop or target is hit while a candle is still forming, so the newest
        CLOSED candle closed while the trade was open. The backtest never enters
        on such a candle (the next trade may start on the exit candle or later),
        so it is marked as decided instead of being traded late at a price that
        has moved on. Timeouts are left as they were: the round-number strategy's
        2-hour limit fires just after its exit candle has closed, and the
        backtest may enter on that very candle."""
        bar_t = self.feed.latest_closed_bar_time()
        if bar_t is not None:
            self._last_decision_bar = bar_t

    def _maybe_signal(self):
        if self.journal.has_open_position():
            return
        # Decide once per hour, on the HH:00 candle as soon as it has closed --
        # the exact data the backtest decides on. Re-running the engine on every
        # poll evaluated half-formed candles thousands of times a day.
        bar_t = self.feed.latest_closed_bar_time()
        if (bar_t is None
                or (bar_t.hour * 60 + bar_t.minute + self.decision_offset_minutes) % self.decision_minutes != 0
                or bar_t == self._last_decision_bar):
            return
        self._last_decision_bar = bar_t
        if not self.feed.data_is_fresh():
            print(f"[{self.feed.now()}] skipped the {bar_t} decision -- market closed / no live ticks")
            return
        m5 = self.feed.get_recent_5m(self.lookback_5m)
        if len(m5) < self.lookback_5m // 2 or m5.index[-1] != bar_t:
            return
        daily = self.feed.get_daily(250)
        data = {"1H": _resample(m5, "1h"), "15M": _resample(m5, "15min"),
                "5M": m5, "1D": daily}
        sig = self.signal_fn(data, symbol=self.symbol)
        if sig.verdict in ("BUY", "SELL"):
            direction = "buy" if sig.verdict == "BUY" else "sell"
            bid, ask = self.feed.current_quote()
            fill_entry = ask if direction == "buy" else bid   # you pay the other side
            self.journal.open_trade(self.feed.now(), direction, sig.plan,
                                     sig.confidence, sig.breakdown,
                                     bar_opened=bar_t, fill_entry=round(fill_entry, 3))
            print(f"[{self.feed.now()}] SIGNAL: {sig.verdict} conf={sig.confidence}% "
                  f"entry~{(sig.plan.entry_low+sig.plan.entry_high)/2:.2f} "
                  f"SL={sig.plan.sl:.2f} TP1={sig.plan.tp1:.2f} TP2={sig.plan.tp2:.2f}")

        # Observation hook, always last: the trade decision above is already
        # made and journaled, so nothing a listener does can influence it.
        if self.on_signal is not None:
            try:
                self.on_signal(sig)
            except Exception as e:                      # never let a UI break trading
                print(f"[{self.feed.now()}] on_signal listener raised (ignored): {e}")

    def tick(self):
        self._check_open_position()
        self._maybe_signal()

    def run_live(self, poll_seconds: int = 60):
        """For real deployment: polls the feed on an interval, forever. Keep it
        at or below 300s: the HH:00 candle is the latest closed candle only
        from HH:05 to HH:10, and a poll that misses that window skips the hour."""
        print(f"Paper trading {self.symbol} -- journal: {self.journal.path}")
        while True:
            try:
                self.tick()
            except Exception as e:
                print(f"[{pd.Timestamp.now()}] error this tick: {e}")
            time_module.sleep(poll_seconds)

    def run_replay(self, feed: ReplayDataFeed, step_bars: int = 1, verbose_every: int = 500):
        """For testing the runner itself against historical data. Ticks on
        every bar by default: positions are managed at each bar close, while
        decisions still happen only on HH:00 candles, like the backtest."""
        n = 0
        while True:
            if n % step_bars == 0:
                self.tick()
            if n % verbose_every == 0 and n > 0:
                print(f"  ...{n} bars replayed")
            n += 1
            if not feed.advance():
                break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "report":
        Journal(sys.argv[2] if len(sys.argv) > 2 else "paper_journal.csv").report()
    else:
        print("Usage:")
        print("  python3 paper_trader.py report [journal.csv]   -- print scoreboard")
        print("  (for a live/replay run, see run_replay_demo.py and the MT5DataFeed class)")
