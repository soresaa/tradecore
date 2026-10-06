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
         "NAS100_NOISE": "NAS100 day trade", "USDJPY_BO4H_BIG": "USD/JPY 4-hour big target"}
SYMBOL = {"MAIN": "XAUUSD", "XAUUSD_BO4H": "XAUUSD", "XAUUSD_BO4H_BIG": "XAUUSD", "XAUUSD_BO4H_3R": "XAUUSD",
          "XAUUSD_RC2": "XAUUSD", "BTCUSD_BO1H": "BTCUSD", "NAS100_NOISE": "USTEC (NAS100)", "USDJPY_BO4H_BIG": "USDJPY"}
POLL_SECONDS = 10
# Exness Standard contract specs: (USD per 1.0 price move per 1 lot, smallest lot). USD/JPY's depends on the price.
LOT_SPECS = {"XAUUSD": (100.0, 0.01), "BTCUSD": (1.0, 0.01), "USTEC (NAS100)": (1.0, 0.05), "USDJPY": (None, 0.01)}
CENT_MARKETS = ("XAUUSD", "USDJPY")       # Exness Cent accounts trade forex + metals, not indices / crypto
USER_DEFAULTS = {"risk_usd": 1.0, "account": "standard", "muted": [], "daily_summary": True, "summary_utc_hour": 20,
                 "summary_sent": ""}


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
        if os.path.exists(self.events_path):
            for line in open(self.events_path, encoding="utf-8"):
                try:
                    e = json.loads(line)
                    e["id"] = len(self.events)
                    self.events.append(e)
                except Exception:
                    pass
        threading.Thread(target=self._loop, name="tradecore-cloud", daemon=True).start()

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
                if self.running:
                    if self.main is not None:
                        try:
                            self.price = self.main_feed.current_price()
                            self.main.tick()
                            self.error = None
                        except Exception as e:
                            self.error = f"gold 1-hour trend: {e}"
                    for m in list(self.markets.values()):
                        m.tick()
                    self.last_tick_at = datetime.now().isoformat(timespec="seconds")
                    self._watch()
                    self._maybe_summary()
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
                if key == "NAS100_NOISE":
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
                if key == "NAS100_NOISE":
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
                         "strategy": "indie_trend", "symbol": "XAUUSD", "sound_on": False, "toast_on": False,
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
