"""
Price feeds for the cloud app, with the same methods the strategies already use on MT5
(paper_trader.MT5DataFeed): now, current_price, current_quote, data_is_fresh, latest_closed_bar_time,
get_recent_5m, get_daily. Candles are BID prices on UTC time (like Exness MT5) and only CLOSED
5-minute candles are returned, exactly as the desktop app does.

  OandaFeed   gold, NAS100, USD/JPY, EUR/USD, GBP/USD from a free OANDA practice account (token needed)
  BinanceFeed BTC from Binance's public market data (no account)

Each feed keeps the candles it has already downloaded in memory and on disk, and only asks for
the new ones (a few kilobytes a minute). One feed per symbol is shared by every strategy on it.
"""
from __future__ import annotations

import os
import threading
import time

import pandas as pd
import requests

# Exness MT5 symbol (as in market_strategies.PLAYBOOK) -> (source, symbol there, price digits)
SYMBOLS = {
    "XAUUSDm": ("oanda", "XAU_USD", 3),
    "USTECm": ("oanda", "NAS100_USD", 2),
    "USDJPYm": ("oanda", "USD_JPY", 3),
    "EURUSDm": ("oanda", "EUR_USD", 5),
    "GBPUSDm": ("oanda", "GBP_USD", 5),
    "BTCUSDm": ("binance", "BTCUSDT", 2),
}
COLS = ["open", "high", "low", "close", "volume"]


class _CandleFeed:
    """Shared logic: an in-memory + on-disk cache of closed 5-minute candles."""
    FRESH_TICK_SECONDS = 120
    REFRESH_SECONDS = 5            # never ask the provider more often than this per symbol

    def __init__(self, symbol: str, cache_dir: str, digits: int, keep_bars: int = 80000):
        self.symbol = symbol
        self.digits = digits
        self.keep = keep_bars
        self.lock = threading.RLock()
        self.cache_path = os.path.join(cache_dir, f"candles_{symbol}.csv")
        self.m5 = pd.DataFrame(columns=COLS)
        self._last_refresh = 0.0
        self._quote = (None, None, 0.0)      # bid, ask, monotonic time
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path, index_col=0, parse_dates=True)
                self.m5 = df[COLS].astype(float).sort_index()
            except Exception:
                self.m5 = pd.DataFrame(columns=COLS)

    # ---- provider-specific
    def _fetch_closed(self, since: pd.Timestamp | None, need: int) -> pd.DataFrame:
        raise NotImplementedError

    def _fetch_quote(self) -> tuple:
        raise NotImplementedError

    def _fetch_daily(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

    # ---- shared
    def _refresh(self, need: int = 0, force: bool = False):
        with self.lock:
            if not force and time.monotonic() - self._last_refresh < self.REFRESH_SECONDS and len(self.m5) >= need:
                return
            since = self.m5.index[-1] if len(self.m5) else None
            new = self._fetch_closed(since, max(need - len(self.m5), 0))
            if len(new):
                df = pd.concat([self.m5, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                self.m5 = df.iloc[-self.keep:]
                self.m5.to_csv(self.cache_path)
            self._last_refresh = time.monotonic()

    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")

    def get_recent_5m(self, n_bars: int) -> pd.DataFrame:
        self._refresh(need=n_bars)
        with self.lock:
            if not len(self.m5):
                raise RuntimeError(f"no 5-minute candles for {self.symbol} yet")
            return self.m5.iloc[-n_bars:].copy()

    def latest_closed_bar_time(self):
        self._refresh()
        with self.lock:
            return self.m5.index[-1] if len(self.m5) else None

    def get_daily(self, n_days: int) -> pd.DataFrame:
        return self._fetch_daily(n_days)

    def current_quote(self) -> tuple:
        bid, ask, at = self._quote
        if bid is None or time.monotonic() - at > 2.0:
            bid, ask, tick_time = self._fetch_quote()
            self._quote = (bid, ask, time.monotonic())
            self._tick_time = tick_time
        return self._quote[0], self._quote[1]

    def current_price(self) -> float:
        return float(self.current_quote()[0])

    def data_is_fresh(self, sample_seconds: int = 10) -> bool:
        try:
            self.current_quote()
            t = getattr(self, "_tick_time", None)
            return t is not None and (pd.Timestamp.now(tz="UTC") - t).total_seconds() <= self.FRESH_TICK_SECONDS
        except Exception:
            return False


class OandaFeed(_CandleFeed):
    PAGE = 5000

    def __init__(self, instrument: str, token: str, env: str, cache_dir: str, digits: int):
        super().__init__(instrument, cache_dir, digits)
        host = "api-fxtrade.oanda.com" if env == "live" else "api-fxpractice.oanda.com"
        self.base = f"https://{host}/v3/instruments/{instrument}/candles"
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "UNIX"})

    def _get(self, params: dict) -> list:
        r = self.s.get(self.base, params=params, timeout=20)
        if r.status_code == 401:
            raise RuntimeError("OANDA refused the token (401) - check OANDA_TOKEN and OANDA_ENV in settings.env")
        r.raise_for_status()
        return r.json().get("candles", [])

    @staticmethod
    def _rows(candles: list, closed_only: bool = True) -> pd.DataFrame:
        rows = []
        for c in candles:
            if closed_only and not c.get("complete", False):
                continue
            b = c["bid"]
            rows.append((pd.Timestamp(float(c["time"]), unit="s"), float(b["o"]), float(b["h"]), float(b["l"]),
                         float(b["c"]), float(c.get("volume", 0))))
        if not rows:
            return pd.DataFrame(columns=COLS)
        df = pd.DataFrame(rows, columns=["time"] + COLS).set_index("time")
        return df

    def _fetch_closed(self, since, need):
        frames = []
        if since is None or need > 0:
            # backfill: pages of 5,000 candles going back in time
            to, got = None, 0
            target = max(need, 1)
            while got < target:
                p = {"granularity": "M5", "price": "B", "count": self.PAGE}
                if to is not None:
                    p["to"] = f"{to.timestamp():.0f}"
                df = self._rows(self._get(p))
                if not len(df):
                    break
                frames.append(df)
                got += len(df)
                to = df.index[0]
                if len(df) < self.PAGE // 2:
                    break
        if since is not None:
            p = {"granularity": "M5", "price": "B", "from": f"{since.timestamp():.0f}", "count": self.PAGE}
            frames.append(self._rows(self._get(p)))
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=COLS)

    def _fetch_quote(self):
        c = self.s.get(self.base, params={"granularity": "S5", "price": "BA", "count": 1}, timeout=15)
        c.raise_for_status()
        k = c.json()["candles"][-1]
        return float(k["bid"]["c"]), float(k["ask"]["c"]), pd.Timestamp(float(k["time"]), unit="s", tz="UTC")

    def _fetch_daily(self, n):
        cs = self._get({"granularity": "D", "price": "B", "count": min(n, 5000),
                        "dailyAlignment": 0, "alignmentTimezone": "UTC"})
        return self._rows(cs, closed_only=False)[["open", "high", "low", "close"]]


class BinanceFeed(_CandleFeed):
    BASE = "https://data-api.binance.vision/api/v3"
    PAGE = 1000

    def __init__(self, pair: str, cache_dir: str, digits: int):
        super().__init__(pair, cache_dir, digits)
        self.s = requests.Session()

    def _klines(self, interval: str, **kw) -> list:
        r = self.s.get(f"{self.BASE}/klines", params={"symbol": self.symbol, "interval": interval, **kw}, timeout=20)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _rows(k: list, closed_only: bool = True) -> pd.DataFrame:
        now_ms = time.time() * 1000
        rows = [(pd.Timestamp(int(x[0]), unit="ms"), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))
                for x in k if not closed_only or int(x[6]) < now_ms]
        if not rows:
            return pd.DataFrame(columns=COLS)
        return pd.DataFrame(rows, columns=["time"] + COLS).set_index("time")

    def _fetch_closed(self, since, need):
        frames = []
        if since is None or need > 0:
            end, got = None, 0
            while got < max(need, 1):
                kw = {"limit": self.PAGE}
                if end is not None:
                    kw["endTime"] = int(end.timestamp() * 1000) - 1
                df = self._rows(self._klines("5m", **kw))
                if not len(df):
                    break
                frames.append(df)
                got += len(df)
                end = df.index[0]
        if since is not None:
            frames.append(self._rows(self._klines("5m", startTime=int(since.timestamp() * 1000), limit=self.PAGE)))
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=COLS)

    def _fetch_quote(self):
        r = self.s.get(f"{self.BASE}/ticker/bookTicker", params={"symbol": self.symbol}, timeout=15)
        r.raise_for_status()
        j = r.json()
        return float(j["bidPrice"]), float(j["askPrice"]), pd.Timestamp.now(tz="UTC")

    def _fetch_daily(self, n):
        return self._rows(self._klines("1d", limit=min(n, 1000)), closed_only=False)[["open", "high", "low", "close"]]


def _resample_daily(m5: pd.DataFrame, n: int) -> pd.DataFrame:
    """UTC days built from the 5-minute candles (the same day boundaries as Exness MT5)."""
    d = m5.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return d.iloc[-n:]


class _LiveQuote:
    """A live price between candles, from a no-account source, asked at most every `every` seconds."""

    def __init__(self, fn, every: float = 15.0):
        self.fn, self.every = fn, every
        self.value, self.at = None, 0.0

    def get(self):
        if self.value is None or time.monotonic() - self.at > self.every:
            try:
                self.value = float(self.fn())
            except Exception:
                pass
            self.at = time.monotonic()
        return self.value


YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/130 Safari/537.36"}


def yahoo_price(symbol: str) -> float:
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"interval": "1m", "range": "1d"}, headers=YAHOO_HEADERS, timeout=15)
    r.raise_for_status()
    return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])


def gold_api_price() -> float:
    r = requests.get("https://api.gold-api.com/price/XAU", timeout=15)
    r.raise_for_status()
    return float(r.json()["price"])


class TwelveDataFeed(_CandleFeed):
    """Gold / forex 5-minute candles from Twelve Data's free plan (800 credits a day, 8 a minute): the history once
    (about 14 calls per symbol, paced), then ONE call per symbol each time a new 5-minute candle has closed
    (288 a day). The live price between candles comes from a no-account source (`live`)."""
    URL = "https://api.twelvedata.com/time_series"
    PAGE = 5000
    PACE = 8.0                      # seconds between calls -> never more than 8 a minute

    def __init__(self, symbol: str, key: str, cache_dir: str, digits: int, live=None):
        super().__init__(symbol.replace("/", ""), cache_dir, digits)
        self.td_symbol, self.key = symbol, key
        self.live = _LiveQuote(live) if live else None
        self._next_fetch = 0.0
        self._last_call = 0.0

    def _call(self, **params) -> pd.DataFrame:
        wait = self.PACE - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()
        r = requests.get(self.URL, params={"symbol": self.td_symbol, "interval": "5min", "timezone": "UTC",
                                           "apikey": self.key, **params}, timeout=30)
        j = r.json()
        if j.get("status") == "error":
            raise RuntimeError(f"Twelve Data: {j.get('message', '')[:160]}")
        rows = [(pd.Timestamp(v["datetime"]), float(v["open"]), float(v["high"]), float(v["low"]), float(v["close"]),
                 float(v.get("volume") or 0)) for v in j.get("values", [])]
        if not rows:
            return pd.DataFrame(columns=COLS)
        df = pd.DataFrame(rows, columns=["time"] + COLS).set_index("time").sort_index()
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return df[df.index + pd.Timedelta(minutes=5) <= now]          # closed candles only

    def _fetch_closed(self, since, need):
        frames = []
        if since is None or need > 0:
            end, got = None, 0
            while got < max(need, 1):
                kw = {"outputsize": self.PAGE}
                if end is not None:
                    kw["end_date"] = (end - pd.Timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                df = self._call(**kw)
                if not len(df):
                    break
                frames.append(df)
                got += len(df)
                end = df.index[0]
                if len(df) < self.PAGE // 2:
                    break
        if since is not None:
            frames.append(self._call(start_date=since.strftime("%Y-%m-%d %H:%M:%S"), outputsize=self.PAGE))
        frames = [f for f in frames if len(f)]
        return pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=COLS)

    def _refresh(self, need: int = 0, force: bool = False):
        # spend credits only when a new candle has closed (or history is missing)
        with self.lock:
            if not force and time.time() < self._next_fetch:
                if len(self.m5) >= need:
                    return
                if getattr(self, "_last_error", None):       # failing: answer at once, do not wait 8 s again
                    raise RuntimeError(self._last_error)
            try:
                super()._refresh(need=need, force=True)
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)[:200]
                self._next_fetch = time.time() + 60          # pause this feed a minute instead of slowing everything
                if len(self.m5) >= max(need, 1):
                    return                                   # keep using the candles we already have
                raise
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            expected = now.floor("5min") - pd.Timedelta(minutes=5)            # the newest candle that has closed
            last = self.m5.index[-1] if len(self.m5) else None
            if last is not None and now - last > pd.Timedelta(minutes=30):
                self._next_fetch = time.time() + 900                          # market closed (weekend): every 15 min
            elif last is not None and last < expected:
                self._next_fetch = time.time() + 20                           # candle not published yet: try again soon
            else:
                self._next_fetch = (now.floor("5min") + pd.Timedelta(minutes=5, seconds=15)).tz_localize("UTC").timestamp()

    def _fetch_quote(self):
        p = self.live.get() if self.live else None
        if p is None:
            with self.lock:
                p = float(self.m5["close"].iloc[-1]) if len(self.m5) else None
        if p is None:
            raise RuntimeError(f"no price for {self.td_symbol} yet")
        last = self.m5.index[-1] if len(self.m5) else None
        t = (last + pd.Timedelta(minutes=5)).tz_localize("UTC") if last is not None else None
        return p, p, t

    def data_is_fresh(self, sample_seconds: int = 10) -> bool:
        self._refresh()
        last = self.latest_closed_bar_time()
        return last is not None and (pd.Timestamp.now(tz="UTC").tz_localize(None) - last) <= pd.Timedelta(minutes=20)

    def _fetch_daily(self, n):
        self._refresh()
        with self.lock:
            return _resample_daily(self.m5, n)


class YahooFeed(_CandleFeed):
    """Index 5-minute candles from Yahoo Finance (no account; 60 days kept by Yahoo, enough for the NAS100 day
    trade, which needs 15 New York sessions). Used for NAS100 = the Nasdaq-100 index (^NDX), regular hours."""
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
    REFRESH_SECONDS = 30

    def __init__(self, symbol: str, cache_dir: str, digits: int):
        super().__init__(symbol.replace("^", "IDX_"), cache_dir, digits)
        self.y_symbol = symbol
        self._price = None

    def _fetch_closed(self, since, need):
        rng = "60d" if since is None or need > 0 else "5d"
        r = requests.get(self.URL + self.y_symbol, params={"interval": "5m", "range": rng},
                         headers=YAHOO_HEADERS, timeout=20)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        self._price = res["meta"].get("regularMarketPrice")
        ts = res.get("timestamp") or []
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                           "volume": q.get("volume")}, index=pd.to_datetime(ts, unit="s")).dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0)
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        df = df[df.index + pd.Timedelta(minutes=5) <= now]
        return df[~df.index.duplicated(keep="last")].astype(float)

    def _fetch_quote(self):
        self._refresh()
        p = self._price if self._price is not None else (float(self.m5["close"].iloc[-1]) if len(self.m5) else None)
        if p is None:
            raise RuntimeError(f"no price for {self.y_symbol} yet")
        last = self.m5.index[-1] if len(self.m5) else None
        return float(p), float(p), (last + pd.Timedelta(minutes=5)).tz_localize("UTC") if last is not None else None

    def data_is_fresh(self, sample_seconds: int = 10) -> bool:
        self._refresh()
        last = self.latest_closed_bar_time()
        return last is not None and (pd.Timestamp.now(tz="UTC").tz_localize(None) - last) <= pd.Timedelta(minutes=20)

    def _fetch_daily(self, n):
        self._refresh()
        with self.lock:
            return _resample_daily(self.m5, n)


# PROVIDER=free (default): no OANDA account needed
FREE_SYMBOLS = {
    "XAUUSDm": ("twelvedata", "XAU/USD", 2, gold_api_price),
    "USDJPYm": ("twelvedata", "USD/JPY", 3, lambda: yahoo_price("JPY=X")),
    "EURUSDm": ("twelvedata", "EUR/USD", 5, lambda: yahoo_price("EURUSD=X")),
    "GBPUSDm": ("twelvedata", "GBP/USD", 5, lambda: yahoo_price("GBPUSD=X")),
    "USTECm": ("yahoo", "^NDX", 2, None),
    "BTCUSDm": ("binance", "BTCUSDT", 2, None),
}


class FeedFactory:
    """broker symbol -> one shared feed per symbol."""

    def __init__(self, cfg: dict, cache_dir: str):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.feeds = {}
        self.lock = threading.Lock()

    def __call__(self, broker_symbol: str):
        with self.lock:
            if broker_symbol not in self.feeds:
                self.feeds[broker_symbol] = self._make(broker_symbol)
            return self.feeds[broker_symbol]

    def _make(self, broker_symbol: str):
        if self.cfg.get("PROVIDER", "free") == "oanda":
            src, sym, digits = SYMBOLS[broker_symbol]
            if src == "oanda":
                tok = self.cfg.get("OANDA_TOKEN", "")
                if not tok:
                    raise RuntimeError("OANDA_TOKEN is empty - add it to the Space secrets")
                return OandaFeed(sym, tok, self.cfg.get("OANDA_ENV", "practice"), self.cache_dir, digits)
            return BinanceFeed(sym, self.cache_dir, digits)
        src, sym, digits, live = FREE_SYMBOLS[broker_symbol]
        if src == "twelvedata":
            key = self.cfg.get("TWELVEDATA_KEY", "")
            if not key:
                raise RuntimeError("TWELVEDATA_KEY is empty - add your free Twelve Data API key to the Space secrets")
            return TwelveDataFeed(sym, key, self.cache_dir, digits, live=live)
        if src == "yahoo":
            return YahooFeed(sym, self.cache_dir, digits)
        return BinanceFeed(sym, self.cache_dir, digits)
