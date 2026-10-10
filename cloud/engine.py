"""
The always-on part of the cloud app: the SAME strategy code as the desktop app (paper_trader.PaperTrader for the gold
1-hour trend, forward_markets.ForwardMarket / NoiseMarket for the others), fed by OANDA / Binance instead of MT5.
Every 10 seconds it lets each strategy check its candles, then compares each strategy's state with the last one and
sends a push alert on a NEW trade, on TP1 and on every exit. Paper only: it places no orders.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cloud.push import Push            # noqa: E402
from cloud.persist import Persist      # noqa: E402

NAMES = {"MAIN": "Gold 1-hour trend", "XAUUSD_BO4H": "Gold 4-hour breakout", "XAUUSD_BO4H_BIG": "Gold 4-hour big target",
         "XAUUSD_BO4H_3R": "Gold 4-hour 3R", "XAUUSD_RC2": "Gold round numbers", "BTCUSD_BO1H": "BTC 1-hour breakout",
         "NAS100_NOISE": "NAS100 day trade", "USDJPY_BO4H_BIG": "USD/JPY 4-hour big target",
         "NAS100_NOISE_PT": "NAS100 + TradeCore trend (research)"}
SYMBOL = {"MAIN": "XAUUSD", "XAUUSD_BO4H": "XAUUSD", "XAUUSD_BO4H_BIG": "XAUUSD", "XAUUSD_BO4H_3R": "XAUUSD",
          "XAUUSD_RC2": "XAUUSD", "BTCUSD_BO1H": "BTCUSD", "NAS100_NOISE": "USTEC (NAS100)", "USDJPY_BO4H_BIG": "USDJPY",
          "NAS100_NOISE_PT": "USTEC (NAS100)"}
POLL_SECONDS = 10
# Exness Standard contract specs: (USD per 1.0 price move per 1 lot, smallest lot). USD/JPY's depends on the price.
LOT_SPECS = {"XAUUSD": (100.0, 0.01), "BTCUSD": (1.0, 0.01), "USTEC (NAS100)": (1.0, 0.05), "USDJPY": (None, 0.01)}
CENT_MARKETS = ("XAUUSD", "USDJPY")       # Exness Cent accounts trade forex + metals, not indices / crypto
USER_DEFAULTS = {"risk_usd": 1.0, "account": "standard", "muted": [], "daily_summary": True, "summary_utc_hour": 20,
                 "summary_sent": "", "news_alerts": True, "news_minutes": 30, "weekly_report": True, "weekly_sent": "",
                 "sound_on": True, "decision_minutes": 0}          # 0 = the server's MAIN_DECISION_MINUTES
# What each strategy did in its test (the "unseen" years), to compare with its live record.
#   win %, profit factor, average R per trade, trades per month, where the numbers come from
BACKTEST = {
    "XAUUSD_BO4H": (41.5, 1.30, 0.15, 2.2, "final test 2022-26, 118 trades"),
    "XAUUSD_BO4H_BIG": (24.0, 1.40, 0.25, 2.6, "final check 2022-26, 139 trades"),
    "XAUUSD_BO4H_3R": (64.2, 1.46, 0.16, 2.3, "final check 2022-26"),
    "XAUUSD_RC2": (65.4, 1.37, 0.08, 11.0, "final test 2022-26, 587 trades"),
    "BTCUSD_BO1H": (37.7, 1.39, 0.21, 6.4, "final test 2025-26, 130 trades"),
    "NAS100_NOISE": (40.2, 1.26, None, 35.0, "2004-26, 2,182 days (results in points)"),
    "USDJPY_BO4H_BIG": (22.8, 1.36, 0.21, 2.6, "final test 2022-26, 136 trades"),
    "NAS100_NOISE_PT": (49.5, 1.65, None, 33.0, "research 2020-23, 422 days (results in points; NOT proven)"),
}
# The worst losing stretch each strategy had in its tests (R; NAS100 in points at today's price), for the safety brake
# (PAUSE when a live stretch is 1.5x deeper). 2018-26 where the team study measured it, else the full history.
TEST_DROP = {"XAUUSD_RC2": 6.3, "XAUUSD_BO4H_BIG": 10.8, "XAUUSD_BO4H_3R": 6.6, "XAUUSD_BO4H": 15.0,
             "USDJPY_BO4H_BIG": 25.0, "BTCUSD_BO1H": 13.7, "NAS100_NOISE": 3100.0, "NAS100_NOISE_PT": 3100.0}
TEST_DROP_MAIN = {60: 13.2, 15: 25.2, 5: 36.0}        # gold 1-hour trend, by "decide every" (2015-26 test)
MAIN_BACKTEST = {5: (30.0, 1.17, 0.08, 10.0, "2015-26, decides every 5 min, 1,355 trades"),
                 60: (30.0, 1.42, 0.19, 4.0, "2015-26, decides every hour, 531 trades")}
# The steadiest mix an Exness cent account can trade (experiments/scoreboard_cent_teams.py, 2026-10-07). It was chosen
# AFTER seeing the scoreboard, so it is weaker evidence than a test. What the two did together, 2017-11 -> 2026-09:
# 2026-10-09 the user follows these five (alerts on, one trade per signal): the team card shows them together.
# What they did together 2018-03 -> 2026-09 at 1R each (big_team_study trades + the 60-minute gold 1-hour trend):
TEAM = ("XAUUSD_BO4H_BIG", "XAUUSD_RC2", "MAIN", "USDJPY_BO4H_BIG", "BTCUSD_BO1H")
TEAM_NAME = "My team (gold big target + round numbers + 1h trend + USD/JPY + BTC)"
TEAM_BACKTEST = {"months_green": 73.8, "avg_month_r": 5.31, "worst_month_r": -9.87, "max_dd_r": -20.7, "per_month": 23.6,
                 "per_trade_r": 0.226, "win": 44.8, "pf": 1.53, "years_green": "9 of 9",
                 "note": "2018-03 -> 2026-09, 2,417 trades; each member passed its own test, the team itself is "
                         "measured on years already seen"}
PRICE_MARKETS = {"XAUUSD": ("XAUUSDm", "Gold"), "USDJPY": ("USDJPYm", "USD/JPY"), "BTCUSD": ("BTCUSDm", "BTC"),
                 "NAS100": ("USTECm", "Nasdaq-100 index")}
NEWS_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_COUNTRIES = ("USD", "JPY")


WEEKEND_KEEP = ("BTCUSD_BO1H",)                     # crypto trades 24/7; everything else is closed at the weekend


def weekend_now(t: datetime = None) -> bool:
    """Gold, forex and the US indices are closed from Friday 22:00 UTC (after gold's last candle, summer and winter)
    to Sunday 21:00 UTC (just before forex / gold reopen): the server then runs BTC only."""
    t = t or datetime.utcnow()
    wd, h = t.weekday(), t.hour
    return (wd == 4 and h >= 22) or wd == 5 or (wd == 6 and h < 21)


def lot_text(key: str, entry, sl, user: dict) -> str:
    """'Lot 0.02 = risk $0.96' for the user's risk per trade and account type (cent lot = 0.01 standard lot)."""
    import math
    try:
        entry, dist = float(entry), abs(float(entry) - float(sl))
    except (TypeError, ValueError):
        return ""
    sym = SYMBOL.get(key, "")
    if dist <= 0 or sym not in LOT_SPECS:
        return ""
    per, min_lot = LOT_SPECS[sym]
    per = per if per is not None else 100000.0 / entry
    risk = float(user.get("risk_usd", 1.0))
    lots = risk / (dist * per)                          # standard lots
    if user.get("account") == "cent" and sym in CENT_MARKETS:
        c = math.floor(lots * 100 / 0.01 + 1e-9) * 0.01  # cent lots
        if c < 0.01:
            return f" | Lot: smallest 0.01 (cent) risks ${0.0001 * dist * per:.2f} - more than your ${risk:.2f}"
        return f" | Lot {c:.2f} on your Cent account = risk ${c / 100 * dist * per:.2f}"
    l = math.floor(lots / 0.01 + 1e-9) * 0.01
    note = " (Standard account)" if user.get("account") == "cent" else ""
    if l < min_lot:
        return f" | Lot: smallest {min_lot:.2f}{note} risks ${min_lot * dist * per:.2f} - more than your ${risk:.2f}"
    return f" | Lot {l:.2f}{note} = risk ${l * dist * per:.2f}"


def make_feed_factory(cfg: dict, data_dir: str):
    if cfg.get("PROVIDER") == "mt5":                    # local Windows test with the Exness terminal
        from paper_trader import MT5DataFeed
        feeds, lock = {}, threading.Lock()

        def mt5_factory(sym):
            with lock:
                if sym not in feeds:
                    feeds[sym] = MT5DataFeed(sym, auto_launch=True)
                return feeds[sym]
        return mt5_factory
    from cloud.feeds import FeedFactory
    return FeedFactory(cfg, os.path.join(data_dir, "candles"))


def _read_journal(path: str):
    """(trades, stats, open position) like the desktop app's journal reader."""
    try:
        if not os.path.exists(path):
            return [], None, None
        df = pd.read_csv(path)
        if df.empty:
            return [], None, None
        df = df.astype(object).where(pd.notna(df), None)
        trades = df.to_dict("records")
        closed = [t for t in trades if t.get("status") == "closed"]
        open_rows = [t for t in trades if t.get("status") == "open"]
        stats = None
        rs = [float(t["r_multiple"]) for t in closed if t.get("r_multiple") is not None]
        if rs:
            wins = [r for r in rs if r > 0]
            gl = -sum(r for r in rs if r <= 0)
            stats = {"closed": len(rs), "wins": len(wins), "losses": len(rs) - len(wins),
                     "win_rate": round(len(wins) / len(rs) * 100, 1), "total_r": round(sum(rs), 2),
                     "profit_factor": round(sum(wins) / gl, 2) if gl > 0 else None,
                     "equity": [round(v, 2) for v in pd.Series(rs).cumsum().tolist()]}
        return trades, stats, (open_rows[-1] if open_rows else None)
    except Exception:
        return [], None, None


class CloudEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_dir = cfg["DATA_DIR"]
        os.makedirs(self.data_dir, exist_ok=True)
        self.lock = threading.RLock()
        self.logs = collections.deque(maxlen=400)
        self.persist = Persist(cfg, self.data_dir, self.log)
        self.persist.pull()
        self._ensure_server_keys()
        self.push = Push(cfg, self.data_dir, self.log)
        self.keys = [k.strip() for k in cfg.get("MARKETS", "").split(",") if k.strip()]
        self.decision_minutes = int(cfg.get("MAIN_DECISION_MINUTES", "5"))
        self.running = True
        self.connected = False
        self.error = None
        self.main = None
        self.main_feed = None
        self.main_signal = None
        self.main_signal_at = None
        self.price = None
        self.markets = {}
        self.failed = {}
        self.last_tick_at = None
        self._seen = {}
        self.journal_path = os.path.join(self.data_dir, "paper_journal.csv")
        self.events_path = os.path.join(self.data_dir, "events.jsonl")
        self.events = []                         # every alert, numbered: the Android app asks for the new ones
        self._events_lock = threading.Lock()
        self.native_seen = 0.0                   # last time the Android app was listening (then: no Chrome alerts)
        self.user_path = os.path.join(self.data_dir, "user_settings.json")
        self.user = dict(USER_DEFAULTS)
        if os.path.exists(self.user_path):
            try:
                self.user.update(json.load(open(self.user_path, encoding="utf-8")))
            except Exception:
                pass
        if int(self.user.get("decision_minutes") or 0) in (5, 15, 60):     # chosen on the dashboard's Settings
            self.decision_minutes = int(self.user["decision_minutes"])
        if os.path.exists(self.events_path):
            for line in open(self.events_path, encoding="utf-8"):
                try:
                    e = json.loads(line)
                    e["id"] = len(self.events)
                    self.events.append(e)
                except Exception:
                    pass
        self._pa_lock = threading.Lock()
        threading.Thread(target=self._loop, name="tradecore-cloud", daemon=True).start()
        threading.Thread(target=self._side_loop, name="tradecore-alerts", daemon=True).start()

    def _side_loop(self):
        """Price alerts, news warnings, the daily summary and the Sunday report every 10 s - never held up by the strategy loop."""
        while True:
            for job in (self._check_price_alerts, self._check_news, self._maybe_summary, self._maybe_weekly,
                        self._check_drops):
                try:
                    job()
                except Exception as e:
                    self.log(f"{job.__name__}: {str(e)[:120]}")
            time.sleep(POLL_SECONDS)

    def _ensure_server_keys(self):
        """The login-cookie key and the push keys: from the settings if given, else made once here and kept in
        the private data store, so phones stay subscribed and logged in across restarts."""
        path = os.path.join(self.data_dir, "server_keys.json")
        keys = {}
        if os.path.exists(path):
            try:
                keys = json.load(open(path, encoding="utf-8"))
            except Exception:
                keys = {}
        if not keys.get("VAPID_PRIVATE_KEY"):
            import base64
            import secrets
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            b64 = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
            k = ec.generate_private_key(ec.SECP256R1())
            keys = {"SECRET_KEY": secrets.token_hex(32),
                    "VAPID_PRIVATE_KEY": b64(k.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                                             serialization.NoEncryption())),
                    "VAPID_PUBLIC_KEY": b64(k.public_key().public_bytes(serialization.Encoding.X962,
                                                                        serialization.PublicFormat.UncompressedPoint))}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(keys, f)
            self.log("made new push / login keys (kept in your private data store)")
        for k, v in keys.items():
            if not self.cfg.get(k):
                self.cfg[k] = v

    # ------------------------------------------------------------------ logging
    def log(self, msg: str):
        line = f"[{datetime.utcnow():%H:%M:%S} UTC] {msg}"
        print(line, flush=True)
        self.logs.append(line)

    # ------------------------------------------------------------------ build
    def _build(self):
        from paper_trader import PaperTrader, resolve_strategy
        from forward_markets import ForwardWorker
        factory = make_feed_factory(self.cfg, self.data_dir)
        self.factory = factory                   # the price alerts use the same feeds
        if self.main is None:
            try:
                self.main_feed = factory("XAUUSDm")
                self.main = PaperTrader(self.main_feed, self.journal_path, symbol="XAU/USD",
                                        on_signal=self._on_main_signal, signal_fn=resolve_strategy("indie_trend"))
                self.main.decision_minutes = self.decision_minutes
                self.log(f"gold 1-hour trend ready (decides every {self.decision_minutes} min)")
            except Exception as e:
                self.error = f"gold feed: {e}"
                self.log(self.error)
        fw = ForwardWorker(log=self.log, keys=self.keys, feed_factory=factory, journal_dir=self.data_dir)
        fw.markets = self.markets
        fw._build()
        self.failed = dict(fw.failed)
        self._worker = fw
        self.connected = self.main is not None or bool(self.markets)

    def _on_main_signal(self, sig):
        plan = None
        if sig.plan is not None:
            p = sig.plan
            plan = {"entry": round((p.entry_low + p.entry_high) / 2, 2), "sl": round(p.sl, 2),
                    "tp1": round(p.tp1, 2), "tp2": round(p.tp2, 2), "rr1": p.rr1, "rr2": p.rr2}
        self.main_signal = {"verdict": sig.verdict, "confidence": sig.confidence, "reason": sig.reason,
                            "timeframes": dict(sig.timeframes or {}), "plan": plan}
        self.main_signal_at = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------ loop
    def _loop(self):
        last_build = 0.0
        while True:
            try:
                missing = self.main is None or len(self.markets) < len(self.keys)
                if missing and time.time() - last_build > (10 if last_build == 0 else 120):
                    self.failed = {}                 # retry whatever could not start (no token yet, network)
                    self._build()
                    last_build = time.time()
                wk = weekend_now()
                if wk != getattr(self, "_weekend", None):
                    self._weekend = wk
                    self.log("weekend mode ON - markets closed, only BTC runs until Sunday 21:00 UTC" if wk
                             else "weekend mode OFF - all strategies run again")
                if self.running:
                    if self.main is not None and not wk:
                        try:
                            self.price = self.main_feed.current_price()
                            self.main.tick()
                            self.error = None
                        except Exception as e:
                            self.error = f"gold 1-hour trend: {e}"
                    for k, m in list(self.markets.items()):
                        if not wk or k in WEEKEND_KEEP:
                            m.tick()
                    self.last_tick_at = datetime.now().isoformat(timespec="seconds")
                    self._watch()
                self.persist.maybe_push()
            except Exception as e:
                self.error = str(e)
                self.log(f"loop error: {e}")
            time.sleep(POLL_SECONDS)

    # ------------------------------------------------------------------ alerts
    def _state_of(self, key):
        """(open id, tp1 hit, closed count, last closed row) for one strategy."""
        if key == "MAIN":
            trades, _, op = _read_journal(self.journal_path)
            closed = [t for t in trades if t.get("status") == "closed"]
            last = closed[-1] if closed else None
            return ((op or {}).get("time_opened"), bool(op and str(op.get("tp1_hit")).lower() in ("true", "1")),
                    len(closed), last, op)
        m = self.markets.get(key)
        if m is None:
            return None
        snap = m.snapshot()
        rec = snap.get("record") or {}
        last = (rec.get("recent") or [None])[0]
        if snap.get("kind") == "noise":
            pos = (snap.get("noise") or {}).get("position")
            return ((pos or {}).get("opened_utc"), False, int(rec.get("closed") or 0), last,
                    dict(pos, sl=(snap.get("noise") or {}).get("stop")) if pos else None)
        op = snap.get("open")
        return ((op or {}).get("opened"), bool(op and op.get("tp1_hit")), int(rec.get("closed") or 0), last, op)

    def _fmt(self, key, x):
        d = 3 if key.startswith("USDJPY") else 2
        try:
            return f"{float(x):,.{d}f}"
        except Exception:
            return str(x)

    def _watch(self):
        for key in ["MAIN"] + list(self.markets):
            st = self._state_of(key)
            if st is None:
                continue
            open_id, tp1, n_closed, last, op = st
            prev = self._seen.get(key)
            self._seen[key] = (open_id, tp1, n_closed)
            if prev is None:                         # first look after a start: remember, do not alert
                continue
            name, sym = NAMES.get(key, key), SYMBOL.get(key, "")
            if n_closed > prev[2] and last:
                if key.startswith("NAS100_NOISE"):
                    res = f"{float(last.get('points_net') or 0):+.1f} points"
                else:
                    r = last.get("r_net") if last.get("r_net") is not None else last.get("r_multiple")
                    res = f"{float(r):+.2f}R" if r is not None else ""
                self._alert(f"CLOSED {sym} - {name}", f"{last.get('outcome', '')} {res}".strip(), key)
            if open_id and open_id != prev[0] and op:
                d = str(op.get("direction", "")).upper()
                f = lambda v: self._fmt(key, v)

                def dist(level):                    # distance from the entry, to use on Exness's own price
                    try:
                        return self._fmt(key, abs(float(level) - float(op.get("entry"))))
                    except Exception:
                        return "?"
                if key.startswith("NAS100_NOISE"):
                    body = (f"Nasdaq-100 index {f(op.get('entry'))} | SL line {f(op.get('sl'))} = {dist(op.get('sl'))} "
                            f"points away (exit if a :00/:30 candle closes beyond it) | no TP, closes 16:00 New York. "
                            f"On Exness USTEC use the same distance in points.")
                elif key == "XAUUSD_RC2":
                    body = (f"Entry {f(op.get('entry'))} | SL {f(op.get('sl'))} ({dist(op.get('sl'))} away) | "
                            f"TP {f(op.get('tp1'))} ({dist(op.get('tp1'))} away) | closes after 2 hours")
                else:
                    half = "close HALF + " if key == "XAUUSD_BO4H_3R" else ""
                    body = (f"Entry {f(op.get('entry'))} | SL {f(op.get('sl'))} ({dist(op.get('sl'))} away) | "
                            f"TP1 {f(op.get('tp1'))} ({half}SL to entry) | TP2 {f(op.get('tp2'))} ({dist(op.get('tp2'))} away)")
                body += lot_text(key, op.get("entry"), op.get("sl"), self.user)
                self._alert(f"{d} {sym} - {name}", body, key)
            if tp1 and not prev[1] and open_id == prev[0]:
                what = "close HALF and move your SL to entry" if key == "XAUUSD_BO4H_3R" else "move your SL to entry"
                self._alert(f"TP1 HIT {sym} - {name}", what, key)

    def _alert(self, title: str, body: str, key: str) -> int:
        if key in (self.user.get("muted") or []):
            self.log(f"(muted by you) {title}: {body}")
            return 0
        self.log(f"ALERT {title}: {body}")
        e = {"at": datetime.utcnow().isoformat(timespec="seconds") + "Z", "key": key, "title": title, "body": body}
        with self._events_lock:
            try:
                with open(self.events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(e) + "\n")
            except Exception:
                pass
            e["id"] = len(self.events)
            self.events.append(e)
        if self.native_app_listening():
            self.log("  delivered to the TRADECORE Android app (Chrome alerts not needed)")
            return 0
        n = self.push.send(title, body, tag=key)
        self.log(f"  sent to {n} browser device(s) (the Android app is not listening right now)")
        return n

    def native_app_listening(self) -> bool:
        return time.time() - self.native_seen < 180

    # ------------------------------------------------------------------ the dashboard's Settings form (Save)
    def save_settings(self, new: dict) -> dict:
        """What the cloud can change from that form: the alarm sound and how often the gold 1-hour trend decides.
        (Strategy, symbol and check speed are fixed in the cloud; their rows are hidden there.)"""
        u = dict(self.user)
        try:
            if "sound_on" in new:
                u["sound_on"] = bool(new["sound_on"])
            if "decision_minutes" in new:
                dm = int(new["decision_minutes"])
                if dm not in (5, 15, 60):
                    raise ValueError("'Decide every' must be 5, 15 or 60 minutes")
                u["decision_minutes"] = dm
        except (TypeError, ValueError) as e:
            return {"ok": False, "msg": str(e)}
        self.user = u
        with open(self.user_path, "w", encoding="utf-8") as f:
            json.dump(u, f)
        dm = int(u.get("decision_minutes") or 0)
        if dm in (5, 15, 60) and dm != self.decision_minutes:
            self.decision_minutes = dm
            if self.main is not None:
                self.main.decision_minutes = dm
        self.log(f"settings saved: alarm sound {'ON' if u.get('sound_on', True) else 'off'}, "
                 f"gold 1-hour trend decides every {self.decision_minutes} min")
        return {"ok": True, "msg": "saved"}

    def event_view(self, e: dict) -> dict:
        """An alert as the phone / page gets it: alarm = ring the alarm sound (new trades, price alerts, the test)."""
        loud = str(e.get("title", "")).startswith(("BUY", "SELL", "PRICE ALERT", "TRADECORE test"))
        return dict(e, alarm=bool(self.user.get("sound_on", True)) and loud)

    # ------------------------------------------------------------------ your settings, history, daily summary
    def save_user(self, new: dict) -> dict:
        u = dict(self.user)
        try:
            if "risk_usd" in new:
                u["risk_usd"] = max(0.01, min(float(new["risk_usd"]), 100000.0))
            if new.get("account") in ("standard", "cent"):
                u["account"] = new["account"]
            if isinstance(new.get("muted"), list):
                u["muted"] = [k for k in new["muted"] if k in NAMES]
            if "daily_summary" in new:
                u["daily_summary"] = bool(new["daily_summary"])
            if "summary_utc_hour" in new:
                u["summary_utc_hour"] = int(new["summary_utc_hour"]) % 24
            if "news_alerts" in new:
                u["news_alerts"] = bool(new["news_alerts"])
            if "news_minutes" in new:
                u["news_minutes"] = max(5, min(int(new["news_minutes"]), 240))
            if "weekly_report" in new:
                u["weekly_report"] = bool(new["weekly_report"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"bad setting: {e}")
        self.user = u
        with open(self.user_path, "w", encoding="utf-8") as f:
            json.dump(u, f)
        self.log(f"your settings saved: risk ${u['risk_usd']:.2f}, {u['account']} account, "
                 f"{len(u['muted'])} strategy(ies) muted, daily summary {'on' if u['daily_summary'] else 'off'}")
        return u

    def history(self, limit: int = 300) -> list:
        with self._events_lock:
            return list(reversed(self.events[-limit:]))

    # ------------------------------------------------------------------ live results vs the test
    def performance(self) -> list:
        """Each strategy's live paper record next to its test numbers, with a plain verdict."""
        risk = float(self.user.get("risk_usd", 1.0))
        rows = []
        for key in ["MAIN"] + [k for k in self.keys]:
            bt = MAIN_BACKTEST.get(self.decision_minutes, MAIN_BACKTEST[5]) if key == "MAIN" else BACKTEST.get(key)
            if bt is None:
                continue
            path = self.journal_path if key == "MAIN" else os.path.join(self.data_dir, f"forward_{key}.csv")
            row = {"key": key, "name": NAMES.get(key, key), "bt_win": bt[0], "bt_pf": bt[1], "bt_r": bt[2],
                   "bt_per_month": bt[3], "bt_note": bt[4], "n": 0, "unit": "points" if key.startswith("NAS100_NOISE") else "R"}
            try:
                df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
            if len(df):
                if key.startswith("NAS100_NOISE"):
                    r = pd.to_numeric(df["points_net"], errors="coerce").dropna()
                    first = df["entry_time"].iloc[0] if "entry_time" in df else None
                else:
                    df = df[df.get("status") == "closed"] if "status" in df else df.iloc[0:0]
                    rn = pd.to_numeric(df.get("r_net"), errors="coerce") if "r_net" in df else None
                    rm = pd.to_numeric(df.get("r_multiple"), errors="coerce") if "r_multiple" in df else None
                    r = (rn.fillna(rm) if rn is not None and rm is not None else (rn if rn is not None else rm))
                    r = r.dropna() if r is not None else pd.Series(dtype=float)
                    first = df["time_opened"].iloc[0] if len(df) and "time_opened" in df else None
                if len(r):
                    w, l = r[r > 0].sum(), -r[r < 0].sum()
                    row.update(n=int(len(r)), win=round(100 * float((r > 0).mean()), 1), total=round(float(r.sum()), 2),
                               avg=round(float(r.mean()), 3), pf=round(float(w / l), 2) if l > 0 else None,
                               since=str(first)[:10] if first is not None else "")
                    if not key.startswith("NAS100_NOISE"):
                        row["usd"] = round(float(r.sum()) * risk, 2)
                    eq = r.cumsum()
                    peak = eq.cummax().clip(lower=0)
                    row["drop"] = round(float(min(0.0, eq.iloc[-1] - peak.iloc[-1])), 2)       # the stretch it is in NOW
                    row["worst_drop"] = round(float(min(0.0, (eq - peak).min())), 2)
            n = row["n"]
            if n < 20:
                row["verdict"] = f"too early - {n} closed trade(s); about 30 are needed to judge"
            elif key.startswith("NAS100_NOISE"):
                row["verdict"] = ("on track" if (row.get("pf") or 0) >= 1.1 else
                                  "weaker than the test" if row.get("total", 0) > 0 else "losing - watch it")
            else:
                row["verdict"] = ("on track" if row["avg"] >= 0.5 * bt[2] else
                                  "weaker than the test" if row["avg"] > 0 else "losing - watch it")
            # the safety brake: a live losing stretch 1.5x deeper than the worst one in the test -> pause and check
            td = TEST_DROP_MAIN.get(self.decision_minutes) if key == "MAIN" else TEST_DROP.get(key)
            if td:
                row["test_drop"] = -td
                if row.get("drop", 0) < -1.5 * td:
                    u = " points" if row["unit"] == "points" else "R"
                    row["verdict"] = (f"PAUSE - its losing stretch ({row['drop']:+.1f}{u}) is deeper than 1.5x the worst in "
                                      f"its test ({-td:+.1f}{u}); stop following it and check")
            rows.append(row)
        return rows

    def _check_drops(self):
        """Once an hour: a phone warning when a strategy first hits the safety brake (PAUSE), once per episode."""
        now = time.time()
        if now - getattr(self, "_drop_checked", 0) < 3600:
            return
        self._drop_checked = now
        warned = list(self.user.get("drop_warned") or [])
        changed = False
        for row in self.performance():
            paused = str(row.get("verdict", "")).startswith("PAUSE")
            if paused and row["key"] not in warned:
                self._alert(f"PAUSE CHECK: {row['name']}", row["verdict"].split(" - ", 1)[1], row["key"])
                warned.append(row["key"]); changed = True
            elif not paused and row["key"] in warned:
                warned.remove(row["key"]); changed = True
        if changed:
            self.user["drop_warned"] = warned
            self.save_user({})

    # ------------------------------------------------------------------ public health (no trades, no account)
    def health(self) -> dict:
        """Which price sources and strategies work - for /health. Keys in error texts are blanked."""
        import re
        now = time.time()
        if getattr(self, "_health", None) and now - self._health[0] < 20:
            return self._health[1]
        clean = lambda s: re.sub(r"(apikey|token|key|password)=[^&\s'\"]+", r"\1=***", str(s))[:220] if s else None
        feeds = {}
        f = getattr(self, "factory", None)
        for sym, fd in list((getattr(f, "feeds", None) or {}).items()):
            m5 = getattr(fd, "m5", None)
            n = int(len(m5)) if m5 is not None else 0
            feeds[sym] = {"source": type(fd).__name__.replace("Feed", ""), "candles": n,
                          "last_candle_utc": str(m5.index[-1]) if n else None,
                          "error": clean(getattr(fd, "_last_error", None) or getattr(fd, "last_error", None))}
        strategies = {"MAIN": {"name": NAMES["MAIN"], "ok": not self.error, "error": clean(self.error)}}
        for k in self.keys:
            m = self.markets.get(k)
            err = getattr(m, "error", None) if m is not None else self.failed.get(k, "not started yet")
            strategies[k] = {"name": NAMES.get(k, k), "ok": not err, "error": clean(err)}
        out = {"running": self.running, "weekend_mode": weekend_now(), "checked_utc": datetime.utcnow().isoformat(timespec="seconds"),
               "feeds": feeds, "strategies": strategies}
        self._health = (now, out)
        return out

    # ------------------------------------------------------------------ the gold team (round numbers + 4h 3R)
    def _closed_r(self, key) -> pd.DataFrame:
        """Closed paper trades of one strategy measured in R: closing time (UTC) and net R. Never raises."""
        path = self.journal_path if key == "MAIN" else os.path.join(self.data_dir, f"forward_{key}.csv")
        empty = pd.DataFrame({"at": pd.Series(dtype="datetime64[ns, UTC]"), "r": pd.Series(dtype=float)})
        try:
            df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
            if not len(df) or "status" not in df or "time_closed" not in df:
                return empty
            df = df[df["status"] == "closed"]
            r = pd.to_numeric(df["r_net"], errors="coerce") if "r_net" in df else pd.Series(float("nan"), index=df.index)
            if "r_multiple" in df:
                r = r.fillna(pd.to_numeric(df["r_multiple"], errors="coerce"))
            out = pd.DataFrame({"at": pd.to_datetime(df["time_closed"], utc=True, errors="coerce"), "r": r}).dropna()
            return out.sort_values("at").reset_index(drop=True)
        except Exception:
            return empty

    def team(self) -> dict:
        """The two gold strategies as one team: live paper result by month, next to what they did in the test."""
        risk = float(self.user.get("risk_usd", 1.0))
        now = pd.Timestamp.now(tz="UTC")
        members, frames = [], []
        for k in TEAM:
            d = self._closed_r(k)
            frames.append(d)
            seen = self._seen.get(k)
            members.append({"key": k, "name": NAMES.get(k, k), "n": int(len(d)), "r": round(float(d["r"].sum()), 2),
                            "usd": round(float(d["r"].sum()) * risk, 2), "open": bool(seen and seen[0]),
                            "muted": k in (self.user.get("muted") or [])})
        full_frames = [f for f in frames if len(f)]
        t = pd.concat(full_frames, ignore_index=True).sort_values("at").reset_index(drop=True) if full_frames else frames[0]
        cur = now.tz_localize(None).to_period("M")
        months = []
        if len(t):
            per = t["at"].dt.tz_localize(None).dt.to_period("M")
            g = t.groupby(per)["r"].agg(["sum", "count"])
            for p in pd.period_range(min(per.min(), cur), cur, freq="M"):
                r_, c_ = (float(g.loc[p, "sum"]), int(g.loc[p, "count"])) if p in g.index else (0.0, 0)
                months.append({"month": str(p), "r": round(r_, 2), "usd": round(r_ * risk, 2), "n": c_, "full": p != cur})
        full = [m for m in months if m["full"]]
        this = months[-1] if months else {"month": str(cur), "r": 0.0, "usd": 0.0, "n": 0, "full": False}
        wk = t[t["at"] >= now - pd.Timedelta(days=7)]
        n = int(len(t))
        avg = float(t["r"].mean()) if n else 0.0
        eq = t["r"].cumsum()
        dd = float(min(0.0, (eq - eq.cummax().clip(lower=0)).min())) if n else 0.0
        tb = TEAM_BACKTEST
        if n < 20:
            verdict = f"too early - {n} closed trade(s); about 20 are needed (about 2 months)"
        elif avg >= 0.5 * tb["per_trade_r"]:
            verdict = "on track"
        elif avg > 0:
            verdict = "weaker than the test"
        else:
            verdict = "losing - watch it"
        warn = ("The drop is already bigger than the worst drop in the test - stop following it and check."
                if dd < 1.5 * tb["max_dd_r"] else "")
        return {"members": members, "risk_usd": risk, "verdict": verdict, "warning": warn, "weekend": weekend_now(),
                "this_month": this, "months": list(reversed(months))[:24],
                "week": {"n": int(len(wk)), "r": round(float(wk["r"].sum()), 2), "usd": round(float(wk["r"].sum()) * risk, 2)},
                "total": {"n": n, "r": round(float(t["r"].sum()), 2), "usd": round(float(t["r"].sum()) * risk, 2),
                          "win": round(100 * float((t["r"] > 0).mean()), 1) if n else None, "avg": round(avg, 3),
                          "drop_r": round(dd, 2), "drop_usd": round(dd * risk, 2)},
                "full_months": len(full), "green_months": sum(m["r"] > 0 for m in full),
                "red_months": sum(m["r"] < 0 for m in full),
                "test": dict(tb, avg_month_usd=round(tb["avg_month_r"] * risk, 2),
                             worst_month_usd=round(tb["worst_month_r"] * risk, 2), max_dd_usd=round(tb["max_dd_r"] * risk, 2))}

    # ------------------------------------------------------------------ the Sunday report
    def weekly_text(self) -> str:
        risk = float(self.user.get("risk_usd", 1.0))
        m = lambda x: f"{'+' if x >= 0 else '-'}${abs(x):.2f}"
        tm = self.team()
        w, mo, tb = tm["week"], tm["this_month"], tm["test"]
        now = pd.Timestamp.now(tz="UTC")
        wk_all = [self._closed_r(k) for k in ["MAIN"] + [k for k in self.keys if not k.startswith("NAS100_NOISE")]]
        wk_all = [d[d["at"] >= now - pd.Timedelta(days=7)] for d in wk_all]
        n_all, r_all = sum(len(d) for d in wk_all), sum(float(d["r"].sum()) for d in wk_all)
        lines = [f"{TEAM_NAME}: this week {w['n']} closed, {w['r']:+.2f}R ({m(w['usd'])}); "
                 f"this month {mo['r']:+.2f}R ({m(mo['usd'])}). Test: about {m(tb['avg_month_usd'])} a month, "
                 f"{tb['months_green']:.0f}% of months in profit. Verdict: {tm['verdict'].split(' - ')[0]}.",
                 f"All strategies this week: {n_all} closed, {r_all:+.2f}R ({m(r_all * risk)}) at ${risk:.2f} a trade."]
        waiting = []
        for row in self.performance():
            if not row["n"]:
                waiting.append(row["name"])
                continue
            res = f"{row['total']:+.1f} points" if row["unit"] == "points" else f"{row['total']:+.2f}R ({m(row.get('usd', 0))})"
            lines.append(f"{row['name']}: {row['n']} trade(s), won {row['win']:.0f}% (test {row['bt_win']:.0f}%), {res} - "
                         f"{row['verdict'].split(' - ')[0]}")
        if waiting:
            lines.append("No closed trades yet: " + ", ".join(waiting) + ".")
        if tm["warning"]:
            lines.append("WARNING: " + tm["warning"])
        lines.append("Paper trading only - no real orders.")
        return "\n".join(lines)

    def _maybe_weekly(self, force: bool = False):
        u = self.user
        now = datetime.utcnow()
        iso = now.isocalendar()
        week = f"{iso[0]}-W{iso[1]:02d}"
        if not force and (not u.get("weekly_report", True) or now.weekday() != 6
                          or now.hour != int(u.get("summary_utc_hour", 20)) or u.get("weekly_sent") == week):
            return None
        body = self.weekly_text()
        if not force:
            u["weekly_sent"] = week
            self.save_user({})
        return self._alert("TRADECORE weekly report", body, "weekly")

    def send_weekly_now(self) -> dict:
        app = self.native_app_listening()
        n = self._maybe_weekly(force=True)
        return {"ok": True, "msg": "weekly report sent to the TRADECORE app" if app else
                f"weekly report sent to {n or 0} browser device(s) - it is also in History"}

    # ------------------------------------------------------------------ your price alerts
    def _price_alerts_path(self):
        return os.path.join(self.data_dir, "price_alerts.json")

    def price_alerts(self) -> list:
        p = self._price_alerts_path()
        try:
            return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else []
        except Exception:
            return []

    def _save_price_alerts(self, alerts: list):
        with open(self._price_alerts_path(), "w", encoding="utf-8") as f:
            json.dump(alerts, f)

    def add_price_alert(self, market: str, cond: str, level, note: str = "") -> dict:
        if market not in PRICE_MARKETS or cond not in ("above", "below"):
            raise ValueError("choose a market and above / below")
        level = float(level)
        if level <= 0:
            raise ValueError("the price must be above 0")
        with self._pa_lock:
            return self._add_price_alert(market, cond, level, note)

    def _add_price_alert(self, market, cond, level, note):
        alerts = self.price_alerts()
        a = {"id": max([x["id"] for x in alerts] + [0]) + 1, "market": market, "cond": cond, "level": level,
             "note": str(note)[:80], "created": datetime.utcnow().isoformat(timespec="seconds") + "Z", "triggered": ""}
        alerts.append(a)
        self._save_price_alerts(alerts)
        self.log(f"price alert added: {PRICE_MARKETS[market][1]} {cond} {level}")
        return a

    def delete_price_alert(self, aid: int):
        with self._pa_lock:
            self._save_price_alerts([a for a in self.price_alerts() if a["id"] != int(aid)])

    def prices(self) -> dict:
        out = {}
        f = getattr(self, "factory", None)
        wk = weekend_now()
        for m, (sym, _) in PRICE_MARKETS.items():
            if wk and m != "BTCUSD":                     # closed at the weekend: no price to ask for
                out[m] = None
                continue
            try:
                out[m] = float(f(sym).current_price()) if f else None
            except Exception:
                out[m] = None
        return out

    def _check_price_alerts(self):
        if not any(not a.get("triggered") for a in self.price_alerts()):
            return
        prices = self.prices()
        with self._pa_lock:
            self._fire_price_alerts(prices)

    def _fire_price_alerts(self, prices: dict):
        alerts = self.price_alerts()
        active = [a for a in alerts if not a.get("triggered")]
        changed = False
        for a in active:
            p = prices.get(a["market"])
            if p is None:
                continue
            if (a["cond"] == "above" and p >= a["level"]) or (a["cond"] == "below" and p <= a["level"]):
                a["triggered"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
                changed = True
                name = PRICE_MARKETS[a["market"]][1]
                self._alert(f"PRICE ALERT {name} {a['cond']} {a['level']:g}",
                            f"{name} is now {p:,.3f} - it reached your level {a['level']:g}." + (f" {a['note']}" if a["note"] else ""),
                            "price")
        if changed:
            self._save_price_alerts(alerts)

    # ------------------------------------------------------------------ news warnings
    def news(self) -> list:
        """High-impact USD / JPY news of this week (Forex Factory's public calendar), refreshed every hour."""
        now = time.time()
        if getattr(self, "_news_at", 0) and now - self._news_at < 3600:
            return self._news
        try:
            import requests
            raw = requests.get(NEWS_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).json()
            items = []
            for e in raw:
                if e.get("impact") != "High" or e.get("country") not in NEWS_COUNTRIES:
                    continue
                t = pd.Timestamp(e["date"]).tz_convert("UTC")
                items.append({"title": e.get("title", ""), "country": e["country"], "utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
            self._news = sorted(items, key=lambda x: x["utc"])
            self._news_at = now
        except Exception as ex:
            self.log(f"news calendar unavailable: {str(ex)[:80]}")
            self._news_at = now - 3000                # try again in about 10 minutes
            self._news = getattr(self, "_news", [])
        return self._news

    def _check_news(self):
        if not self.user.get("news_alerts", True):
            return
        lead = int(self.user.get("news_minutes", 30))
        warned_path = os.path.join(self.data_dir, "news_warned.json")
        try:
            warned = set(json.load(open(warned_path, encoding="utf-8"))) if os.path.exists(warned_path) else set()
        except Exception:
            warned = set()
        now = pd.Timestamp.now(tz="UTC")
        for e in self.news():
            t = pd.Timestamp(e["utc"])
            mins = (t - now).total_seconds() / 60
            k = e["utc"] + e["title"]
            if 0 < mins <= lead and k not in warned:
                warned.add(k)
                extra = (f" Forecast {e['forecast']}, previous {e['previous']}." if e["forecast"] or e["previous"] else "")
                self._alert(f"NEWS IN {int(round(mins))} MIN: {e['country']} {e['title']}",
                            f"High-impact news at {t:%H:%M} UTC.{extra} Spreads widen and prices can jump - better not to "
                            f"open a new trade just before it; open trades keep their SL.", "news")
                with open(warned_path, "w", encoding="utf-8") as f:
                    json.dump(sorted(warned)[-200:], f)

    def _maybe_summary(self):
        u = self.user
        now = datetime.utcnow()
        today = now.date().isoformat()
        if not u.get("daily_summary", True) or now.hour != int(u.get("summary_utc_hour", 20)) or u.get("summary_sent") == today:
            return
        import re
        evs = [e for e in self.history(500) if e.get("at", "").startswith(today) and e.get("key") not in ("test", "summary")]
        new = [e for e in evs if e["title"].startswith(("BUY", "SELL"))]
        closed = [e for e in evs if e["title"].startswith("CLOSED")]
        r_sum = sum(float(m.group(1)) for e in closed for m in [re.search(r"([+-][\d.]+)R", e["body"])] if m)
        pts = sum(float(m.group(1)) for e in closed for m in [re.search(r"([+-][\d.]+) points", e["body"])] if m)
        risk = float(u.get("risk_usd", 1.0))
        open_now = [NAMES.get(k, k) for k, v in self._seen.items() if v and v[0]]
        money = r_sum * risk
        body = (f"Today: {len(new)} new signal(s), {len(closed)} closed: {r_sum:+.2f}R (about "
                f"{'+' if money >= 0 else '-'}${abs(money):.2f} at "
                f"${risk:.2f} a trade)" + (f", NAS100 {pts:+.1f} points" if pts else "") + ". "
                + (f"Open now: {', '.join(open_now)}." if open_now else "No open trades."))
        u["summary_sent"] = today
        self.save_user({})
        self._alert("TRADECORE daily summary", body, "summary")

    def events_after(self, after: int) -> list:
        with self._events_lock:
            return [e for e in self.events[max(after + 1, 0):]]

    def last_event_id(self) -> int:
        return len(self.events) - 1

    # ------------------------------------------------------------------ page
    def get_state(self) -> dict:
        trades, stats, open_pos = _read_journal(self.journal_path)
        from forward_markets import journal_summary
        markets = []
        for k in self.keys:
            m = self.markets.get(k)
            if m is not None:
                try:
                    markets.append(m.snapshot())
                except Exception as e:
                    markets.append({"key": k, "name": NAMES.get(k, k), "error": str(e), "record": {}})
            elif k in self.failed:
                markets.append({"key": k, "name": NAMES.get(k, k), "error": self.failed[k], "record": {},
                                "price": None, "open": None, "signal": None, "status": "TEST-PASSED"})
        return {
            "running": self.running, "connected": self.connected,
            "symbol": "XAU/USD (" + {"mt5": "Exness MT5", "oanda": "OANDA"}.get(self.cfg.get("PROVIDER"), "Twelve Data") + ")",
            "price": self.price, "last_error": self.error, "last_tick_at": self.last_tick_at,
            "seconds_to_next": None, "signal": self.main_signal, "signal_at": self.main_signal_at,
            "account": {"login": "cloud", "server": "TRADECORE cloud", "demo": True},
            "alerts": [], "log": list(self.logs)[-120:],
            "settings": {"decision_minutes": self.decision_minutes, "poll_seconds": POLL_SECONDS,
                         "strategy": "indie_trend", "symbol": "XAUUSD",
                         "sound_on": bool(self.user.get("sound_on", True)), "toast_on": True,
                         "autostart": True, "forward_markets": True, "min_alert_confidence": 60},
            "trades": trades, "stats": stats, "open_position": open_pos, "journal_path": self.journal_path,
            "main_record": journal_summary(self.journal_path), "markets": markets,
            "cloud": {"push_devices": len(self.push.subs), "push_ready": self.push.ready},
        }

    def test_alert(self) -> dict:
        app = self.native_app_listening()
        n = self._alert("TRADECORE test", "Your phone alerts work. Signals will come like this.", "test")
        return {"ok": True, "msg": "test alert sent to the TRADECORE app - it shows within seconds" if app else
                f"test alert sent to {n} browser device(s)"}
