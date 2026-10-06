"""
Forward test of the markets running beside the live gold trader.

Nothing here is a proven strategy — see `market_strategies.py` and
`experiments/RESULTS_BEST_PER_MARKET.md`. The point of this module is to build
the forward record that the history cannot provide: each market runs its best
candidate on the demo feed, decides on the candle it was tested on (the
closed HH:00 candle; every 5-minute candle for the gold round-number slot),
journals every fill, and is scored with that market's own measured costs.

It places NO orders, exactly like the gold runner, and it uses the same
PaperTrader the gold strategy uses, so the forward numbers mean the same thing
as the backtest numbers. Each market keeps its own journal file
(`forward_<KEY>.csv`) so nothing can contaminate `paper_journal.csv`.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime

import pandas as pd

from market_strategies import FORWARD_MARKETS, PLAYBOOK
from paper_trader import MT5DataFeed, PaperTrader
from trade_accounting import CostModel

HERE = os.path.dirname(os.path.abspath(__file__))
COSTS_MULTI = os.path.join(HERE, "broker_costs_multi.json")


def cost_model_for(key: str) -> CostModel:
    """This market's own spread and swap, measured from the broker. Falls back
    to zero costs with a loud source string rather than pretending gold's
    numbers apply to Bitcoin."""
    try:
        with open(COSTS_MULTI, encoding="utf-8") as f:
            snap = json.load(f)
        s = snap["symbols"][key]
        m = s.get("swap_model", {})
        per_unit = m.get("kind") == "price_per_unit"
        return CostModel(
            spread_price=float(s.get("spread_history", {}).get("median_price", 0.0)),
            swap_long_per_night=float(m.get("long", 0.0)) if per_unit else 0.0,
            swap_short_per_night=float(m.get("short", 0.0)) if per_unit else 0.0,
            triple_swap_weekday=int(s.get("triple_swap_weekday_pandas", 2)),
            point=float(s.get("point", 0.001)),
            source=f"{s.get('broker_symbol')} @ {snap.get('fetched_at')}",
        )
    except Exception as e:
        return CostModel(source=f"no costs for {key} ({e}) -- treated as 0")


_SUMMARY_CACHE: dict = {}


def journal_summary(path: str) -> dict:
    """Record so far: closed trades, win rate, net R, today's net R and the last
    closed trades. Never raises. The page asks every second, so the result is
    cached until the file changes (or the day changes)."""
    out = {"closed": 0, "open": 0, "win_rate": None, "total_r": None, "total_r_net": None,
           "today_closed": 0, "today_r_net": 0.0, "recent": []}
    try:
        if not os.path.exists(path):
            return out
        st = os.stat(path)
        key = (st.st_mtime_ns, st.st_size, datetime.now().date())
        hit = _SUMMARY_CACHE.get(path)
        if hit is not None and hit[0] == key:
            return dict(hit[1])
        df = pd.read_csv(path)
        if not df.empty:
            closed = df[df["status"] == "closed"]
            out["open"] = int((df["status"] == "open").sum())
            out["closed"] = int(len(closed))
            if len(closed):
                r = pd.to_numeric(closed["r_multiple"], errors="coerce")
                rn = pd.to_numeric(closed.get("r_net"), errors="coerce")
                if r.notna().any():
                    out["win_rate"] = round(float((r.dropna() > 0).mean()) * 100, 1)
                    out["total_r"] = round(float(r.sum()), 2)
                if rn.notna().any():
                    out["total_r_net"] = round(float(rn.sum()), 2)
                # today's result, on this computer's own clock
                local_tz = datetime.now().astimezone().tzinfo
                tc = pd.to_datetime(closed["time_closed"], utc=True, errors="coerce").dt.tz_convert(local_tz)
                today = (tc.dt.date == datetime.now().date()).to_numpy()
                net = rn.where(rn.notna(), r)
                out["today_closed"] = int(today.sum())
                out["today_r_net"] = round(float(net[today].sum()), 2)
                for i in range(len(closed) - 1, max(-1, len(closed) - 4), -1):
                    row = closed.iloc[i]
                    when = tc.iloc[i]
                    out["recent"].append({
                        "time": when.strftime("%d %b %H:%M") if pd.notna(when) else "",
                        "direction": str(row.get("direction", "")),
                        "outcome": str(row.get("outcome", "")),
                        "r_net": None if pd.isna(net.iloc[i]) else round(float(net.iloc[i]), 2)})
        _SUMMARY_CACHE[path] = (key, dict(out))
    except Exception:
        pass
    return out


class ForwardMarket:
    """One market: its feed, its strategy, its journal, its last snapshot."""

    def __init__(self, key: str, log=print, feed_factory=None, journal_dir: str = HERE):
        info = PLAYBOOK[key]
        self.key = key
        self.info = info
        self.log = log
        self.journal_path = os.path.join(journal_dir, f"forward_{key}.csv")
        self.error = None
        self.price = None
        self.digits = 5
        self.signal = None
        self.signal_at = None
        # What the strategy sees on the latest closed 5-minute candle, refreshed
        # every candle, FOR THE SCREEN ONLY. It never opens a trade: entries are
        # still taken only on the strategy's own decision candle, exactly as tested.
        self.preview = None
        self._preview_bar = None
        # feed_factory(broker_symbol) -> a feed with MT5DataFeed's methods (the cloud app uses OANDA / Binance)
        self.feed = (feed_factory(info["broker"]) if feed_factory is not None
                     else MT5DataFeed(info["broker"], auto_launch=True, status=log))
        if getattr(self.feed, "digits", None) is not None:
            self.digits = int(self.feed.digits)
        self.trader = PaperTrader(self.feed, self.journal_path, symbol=info["name"],
                                  on_signal=self._on_signal,
                                  costs=cost_model_for(info.get("cost_key", key)),
                                  signal_fn=info["fn"])
        self.trader.lookback_5m = info["lookback"]
        # each market brings its own management, so the app manages a Bitcoin
        # trade the way the Bitcoin study measured it, not the way gold is run
        self.trader.exit_rule = info.get("exit_rule", "breakeven")
        self.trader.max_hold_hours = float(info.get("max_hold", 24.0))
        self.trader.decision_minutes = int(info.get("decision_minutes", 60))
        self.trader.decision_offset_minutes = int(info.get("decision_offset", 0))
        try:
            import MetaTrader5 as mt5
            si = mt5.symbol_info(info["broker"])
            if si is not None:
                self.digits = int(si.digits)
        except Exception:
            pass
        cadence = ("every closed 5-minute candle" if self.trader.decision_minutes == 5
                   else f"when each {self.trader.decision_minutes // 60}h candle closes"
                   if self.trader.decision_offset_minutes else f"every {self.trader.decision_minutes} minutes"
                   if self.trader.decision_minutes != 60 else "on the closed HH:00 candle")
        log(f"forward market ready: {info['name']} ({info['broker']}) strategy "
            f"{info['strategy']} [{info['status']}] — {self.trader.exit_rule}, "
            f"max hold {self.trader.max_hold_hours:.0f}h, decides {cadence}")

    def _on_signal(self, sig):
        plan = None
        if sig.plan is not None:
            p = sig.plan
            d = self.digits
            plan = {"entry": round((p.entry_low + p.entry_high) / 2, d), "sl": round(p.sl, d),
                    "tp1": round(p.tp1, d), "tp2": round(p.tp2, d)}
        self.signal = {"verdict": sig.verdict, "reason": sig.reason,
                       "timeframes": dict(sig.timeframes or {}), "plan": plan}
        self.signal_at = datetime.now().isoformat(timespec="seconds")
        if sig.verdict in ("BUY", "SELL"):
            self.log(f"{self.info['name']}: {sig.verdict} entry {plan['entry']} "
                     f"SL {plan['sl']} TP1 {plan['tp1']} TP2 {plan['tp2']} "
                     f"[forward test, no order placed]")

    def _refresh_preview(self):
        """Runs the strategy on the newest data without trading, once per new
        5-minute candle, so the screen can say what the market looks like now
        instead of 'waiting' for up to an hour."""
        from paper_trader import _resample
        bar_t = self.feed.latest_closed_bar_time()
        if bar_t is None or bar_t == self._preview_bar:
            return
        m5 = self.feed.get_recent_5m(self.trader.lookback_5m)
        if len(m5) < 50:
            return
        data = {"1H": _resample(m5, "1h"), "15M": _resample(m5, "15min"), "5M": m5,
                "1D": self.feed.get_daily(250)}
        sig = self.info["fn"](data, symbol=self.info["name"])
        self._preview_bar = bar_t
        is_decision_candle = (bar_t.hour * 60 + bar_t.minute + self.trader.decision_offset_minutes
                              ) % self.trader.decision_minutes == 0
        reason = sig.reason
        if sig.verdict in ("BUY", "SELL") and not is_decision_candle:
            when = ("the HH:00 candle" if self.trader.decision_offset_minutes == 0
                    else f"the last candle of each {self.trader.decision_minutes // 60}h bar")
            reason = (f"{sig.verdict} setup on this candle — the tested rule only enters on "
                      f"{when}, so this one is not taken")
        elif sig.verdict in ("BUY", "SELL") and self.trader.journal.has_open_position():
            reason = (f"{sig.verdict} setup on this candle — a trade is already open, and the "
                      f"tested rule holds one position at a time")
        # Candle times are broker server time. Work out the server's offset from
        # UTC (the newest closed candle is only minutes old) so the screen can
        # show the user's own clock instead of the broker's.
        utc_now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        offset_h = round((bar_t - utc_now).total_seconds() / 3600)
        bar_utc = bar_t - pd.Timedelta(hours=offset_h)
        self.preview = {"verdict": sig.verdict, "reason": reason,
                        "timeframes": dict(sig.timeframes or {}),
                        "bar": bar_t.strftime("%H:%M"), "bar_utc": bar_utc.isoformat() + "Z",
                        "decision_candle": is_decision_candle}

    def tick(self):
        try:
            self.price = self.feed.current_price()
            self.trader.tick()
            self.error = None
        except Exception as e:
            self.error = str(e)
        try:
            self._refresh_preview()
        except Exception as e:                 # the preview must never break trading
            self.preview = {"verdict": None, "reason": f"preview unavailable: {e}",
                            "timeframes": {}, "bar": None, "decision_candle": False}

    def snapshot(self) -> dict:
        summary = journal_summary(self.journal_path)
        open_pos = None
        try:
            pos = self.trader.journal.get_open_position()
            if pos:
                d = self.digits
                open_pos = {"direction": pos["direction"],
                            "entry": round(float(pos["entry"]), d),
                            "sl": round(float(pos["sl"]), d),
                            "tp1": round(float(pos["tp1"]), d),
                            "tp2": round(float(pos["tp2"]), d),
                            "tp1_hit": str(pos.get("tp1_hit", "")).lower() == "true",
                            "opened": pos.get("time_opened")}
                # when the time limit closes it, in UTC so the screen can show local time
                t0 = pd.Timestamp(pos.get("time_opened"))
                if t0.tzinfo is None:
                    t0 = t0.tz_localize("UTC")
                open_pos["closes_at"] = (t0 + pd.Timedelta(hours=self.trader.max_hold_hours)
                                         ).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
        return {"key": self.key, "name": self.info["name"], "broker": self.info["broker"],
                "strategy": self.info["strategy"], "status": self.info["status"],
                "evidence": self.info["evidence"], "digits": self.digits,
                "decision_minutes": self.trader.decision_minutes,
                "no_target": bool(self.info.get("no_target")),
                "max_hold": self.trader.max_hold_hours,
                "price": self.price, "signal": self.signal, "signal_at": self.signal_at,
                "preview": self.preview,
                "open": open_pos, "record": summary, "error": self.error}


class NoiseMarket:
    """NAS100 day trade (nas100_noise.py). Its state is a pure function of the candles (decisions only on
    half-hour closes), so every new 5-minute candle the whole recent history is replayed; trades that close on
    or after the forward start are appended to forward_<KEY>.csv once. Paper only, no orders."""

    COLS = ["day", "direction", "entry_time", "entry", "exit_time", "exit", "points", "points_net", "outcome"]

    def __init__(self, key: str, log=print, feed_factory=None, journal_dir: str = HERE):
        import nas100_noise
        self._nn = nas100_noise
        info = PLAYBOOK[key]
        self.key, self.info, self.log = key, info, log
        self.journal_path = os.path.join(journal_dir, f"forward_{key}.csv")
        self.error = None
        self.price = None
        self.state = None
        self._bar = None
        self.digits = 2
        if feed_factory is not None:
            self.feed = feed_factory(info["broker"])
        else:
            self.feed = MT5DataFeed(info["broker"], auto_launch=True, status=log)
            try:
                import MetaTrader5 as mt5
                mt5.symbol_select(info["broker"], True)    # the symbol must be in Market Watch for fresh history
            except Exception:
                pass
        self.spread = cost_model_for(info.get("cost_key", key)).spread_price
        log(f"forward market ready: {info['name']} ({info['broker']}) — half-hour checks 10:00-15:30 New York, "
            f"out by 16:00, paper only")

    def _append_closed(self, trades):
        start = pd.Timestamp(self.info.get("forward_start", "2000-01-01"), tz="UTC")
        old = pd.read_csv(self.journal_path) if os.path.exists(self.journal_path) else pd.DataFrame(columns=self.COLS)
        seen = set(old["exit_time"].astype(str)) if len(old) else set()
        new = [t for t in trades if pd.Timestamp(t["exit_time"]) >= start and t["exit_time"] not in seen]
        if new:
            pd.concat([old, pd.DataFrame(new)[self.COLS]], ignore_index=True).to_csv(self.journal_path, index=False)
            for t in new:
                self.log(f"{self.info['name']}: {t['direction'].upper()} closed ({t['outcome']}) "
                         f"{t['points_net']:+.2f} points net [forward test, no order placed]")

    def tick(self):
        try:
            self.price = self.feed.current_price()
            bar_t = self.feed.latest_closed_bar_time()
            if bar_t is not None and bar_t != self._bar:
                m5 = self.feed.get_recent_5m(int(self.info["lookback"]))
                utc_now = pd.Timestamp.now(tz="UTC").tz_localize(None)
                offset_h = round((m5.index[-1] - utc_now).total_seconds() / 3600)   # broker server time -> UTC
                if offset_h > 0:
                    m5.index = m5.index - pd.Timedelta(hours=offset_h)
                res = self._nn.replay(m5, spread=self.spread)
                self._append_closed(res["trades"])
                self.state = res["state"]
                self._bar = bar_t
            self.error = None
        except Exception as e:
            self.error = str(e)

    def record(self) -> dict:
        out = {"closed": 0, "days": 0, "win_rate": None, "points_net": None, "today_closed": 0,
               "today_points_net": 0.0, "recent": [], "unit": "points"}
        try:
            if not os.path.exists(self.journal_path):
                return out
            df = pd.read_csv(self.journal_path)
            if df.empty:
                return out
            pn = pd.to_numeric(df["points_net"], errors="coerce")
            out.update(closed=int(len(df)), days=int(df["day"].nunique()),
                       win_rate=round(float((pn > 0).mean()) * 100, 1), points_net=round(float(pn.sum()), 2))
            local_tz = datetime.now().astimezone().tzinfo
            tc = pd.to_datetime(df["exit_time"], utc=True, errors="coerce").dt.tz_convert(local_tz)
            today = (tc.dt.date == datetime.now().date()).to_numpy()
            out["today_closed"] = int(today.sum())
            out["today_points_net"] = round(float(pn[today].sum()), 2)
            for i in range(len(df) - 1, max(-1, len(df) - 4), -1):
                out["recent"].append({"time": tc.iloc[i].strftime("%d %b %H:%M"),
                                      "direction": str(df["direction"].iloc[i]),
                                      "outcome": str(df["outcome"].iloc[i]),
                                      "points_net": round(float(pn.iloc[i]), 2)})
        except Exception:
            pass
        return out

    def snapshot(self) -> dict:
        return {"key": self.key, "kind": "noise", "name": self.info["name"], "broker": self.info["broker"],
                "strategy": self.info["strategy"], "status": self.info["status"],
                "evidence": self.info["evidence"], "digits": self.digits,
                "decision_minutes": 30, "max_hold": self.info["max_hold"], "no_target": True,
                "price": self.price, "signal": None, "preview": None, "noise": self.state,
                "open": None, "record": self.record(), "error": self.error}


class ForwardWorker(threading.Thread):
    """Ticks every forward market on the same cadence as the gold trader."""

    daemon = True

    def __init__(self, poll_seconds=lambda: 60, log=print, keys=None, feed_factory=None, journal_dir: str = HERE):
        super().__init__(name="forward-markets", daemon=True)
        self._stop = threading.Event()
        self._poll = poll_seconds
        self.log = log
        self.keys = tuple(keys) if keys is not None else FORWARD_MARKETS
        self.feed_factory = feed_factory
        self.journal_dir = journal_dir
        self.markets: dict[str, ForwardMarket] = {}
        self.failed: dict[str, str] = {}

    def stop(self):
        self._stop.set()

    def _build(self):
        for key in self.keys:
            if key in self.markets:
                continue
            try:
                cls = NoiseMarket if PLAYBOOK[key].get("kind") == "noise" else ForwardMarket
                self.markets[key] = cls(key, log=self.log, feed_factory=self.feed_factory,
                                        journal_dir=self.journal_dir)
                self.failed.pop(key, None)
            except Exception as e:
                # one missing symbol must not take the other markets down
                self.failed[key] = str(e)
                self.log(f"forward market {key} unavailable: {e}")

    def run(self):
        self._build()
        if not self.markets:
            self.log("no forward markets could be started")
            return
        self.log(f"forward test running on {', '.join(m.info['name'] for m in self.markets.values())}"
                 f" — each on its tested decision candle, paper only, no orders")
        while not self._stop.is_set():
            for m in list(self.markets.values()):
                if self._stop.is_set():
                    break
                m.tick()
            self._stop.wait(max(5, min(300, int(self._poll()))))

    def snapshots(self) -> list:
        rows = [m.snapshot() for m in self.markets.values()]
        for key, err in self.failed.items():
            info = PLAYBOOK[key]
            rows.append({"key": key, "name": info["name"], "broker": info["broker"],
                         "strategy": info["strategy"], "status": info["status"],
                         "evidence": info["evidence"],
                         "decision_minutes": int(info.get("decision_minutes", 60)),
                         "no_target": bool(info.get("no_target")),
                         "max_hold": float(info.get("max_hold", 24.0)),
                         "price": None, "signal": None,
                         "open": None, "record": journal_summary(
                             os.path.join(self.journal_dir, f"forward_{key}.csv")), "error": err})
        order = {k: i for i, k in enumerate(self.keys)}
        rows.sort(key=lambda r: order.get(r["key"], 99))
        return rows
