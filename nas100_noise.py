"""
NAS100 day trade "NOISE" (Zarattini, Aziz & Barbon 2024) for the app — the SAME candle-by-candle state machine as
tradingview/TRADECORE_nas100_day.pine, whose replay matched the backtest on 2,181 of 2,181 days
(experiments/intraday_momentum_test.py: 2,182 days, 43% won, +107.7% of price, PF 1.26, net of the spread).
experiments/verify_nas100_noise_app.py checks this file against that backtest.

Rules (New York time, regular session 09:30-16:00):
  sigma(t) = 14-day average of |close(t) / open(day) - 1| at the same time of day t.
  upper(t) = max(today's open, yesterday's 16:00 close) x (1 + sigma(t)); lower(t) = min(...) x (1 - sigma(t)).
  At 10:00, 10:30, ... 15:30 (candle closes): above upper -> BUY, below lower -> SELL.
  Exit at those same checks: a buy when the price is back below max(upper, VWAP), a sell above min(lower, VWAP)
  (and the opposite trade can start on the same check). Everything is closed at 16:00.
Paper only: it places no orders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
OPEN_HM, CLOSE_HM = 570, 960            # 09:30, 16:00 New York
N_DAYS = 14


def _slot_time(k: int) -> int:
    return 600 + 30 * k                 # candle END time in minutes (10:00 + 30 k)


def replay(m5: pd.DataFrame, spread: float = 0.0) -> dict:
    """m5: 5-minute candles indexed by UTC time (naive), columns open/high/low/close/volume.
    Returns {'trades': [...], 'state': {...}} — every closed trade and the state after the last candle."""
    idx = m5.index
    ny = idx.tz_localize("UTC").tz_convert(NY)
    hm = (ny.hour * 60 + ny.minute).to_numpy()
    dom = ny.day.to_numpy()
    date = ny.date
    o, h, l, c, v = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    insess = (hm >= OPEN_HM) & (hm < CLOSE_HM)
    slots = [[] for _ in range(12)]
    today = [None] * 12
    qual = have_day = False
    d_open = prev_close = last_close = None
    last_i = None
    cum_pv = cum_v = 0.0
    vw = None
    pos, entry, entry_i = 0, 0.0, None
    cur = None
    trades = []

    def close(i, px, why):
        nonlocal pos
        pts = (px - entry) * pos
        trades.append({"day": str(cur), "direction": "buy" if pos == 1 else "sell",
                       "entry_time": (idx[entry_i] + pd.Timedelta(minutes=5)).isoformat() + "Z", "entry": float(entry),
                       "exit_time": (idx[i] + pd.Timedelta(minutes=5)).isoformat() + "Z", "exit": float(px),
                       "points": round(float(pts), 2), "points_net": round(float(pts - spread), 2),
                       "outcome": why})
        pos = 0

    for i in range(len(c)):
        ins = insess[i]
        new_day = ins and (i == 0 or not insess[i - 1] or dom[i] != dom[i - 1])
        if new_day:
            if qual:
                if pos != 0:
                    close(last_i, last_close, "session ended early")
                for k in range(12):
                    if today[k] is not None:
                        slots[k].append(today[k])
                        slots[k] = slots[k][-N_DAYS:]
                prev_close = last_close
                have_day = True
            qual = hm[i] == OPEN_HM
            d_open = o[i]
            cum_pv = cum_v = 0.0
            vw = None
            pos = 0
            today = [None] * 12
            cur = date[i]
        if ins and qual:
            vol = v[i] + 1e-9
            cum_pv += (h[i] + l[i] + c[i]) / 3 * vol
            cum_v += vol
            vw = cum_pv / cum_v
            last_close, last_i = c[i], i
            te = hm[i] + 5
            if te % 30 == 0 and 600 <= te <= 930:
                k = (te - 600) // 30
                if have_day and len(slots[k]) >= N_DAYS:
                    sig = sum(slots[k]) / len(slots[k])
                    up = max(d_open, prev_close) * (1 + sig)
                    dn = min(d_open, prev_close) * (1 - sig)
                    px = c[i]
                    if pos == 1 and px < max(up, vw):
                        close(i, px, "back below band/VWAP")
                    elif pos == -1 and px > min(dn, vw):
                        close(i, px, "back above band/VWAP")
                    if pos == 0 and px > up:
                        pos, entry, entry_i = 1, px, i
                    elif pos == 0 and px < dn:
                        pos, entry, entry_i = -1, px, i
                today[k] = abs(c[i] / d_open - 1)
            if hm[i] == 955 and pos != 0:
                close(i, c[i], "16:00 close")

    # ---- state after the last candle (for the screen)
    n = len(c)
    st = {"ready": False, "position": None, "price": float(c[-1]) if n else None, "vwap": None,
          "in_session": False, "next_check_utc": None, "upper": None, "lower": None, "stop": None,
          "days_of_history": min(len(s) for s in slots) if slots else 0}
    if n == 0:
        return {"trades": trades, "state": st}
    last_ny = ny[-1] + pd.Timedelta(minutes=5)                    # end of the last closed candle
    end_hm = last_ny.hour * 60 + last_ny.minute
    same_day = cur is not None and last_ny.date() == cur and qual
    st["in_session"] = bool(same_day and OPEN_HM < end_hm <= CLOSE_HM)
    st["ready"] = have_day and len(slots[0]) >= N_DAYS
    if st["in_session"]:
        st["vwap"] = round(float(vw), 2) if vw is not None else None
        st["closes_utc"] = (pd.Timestamp(last_ny.date()).tz_localize(NY) + pd.Timedelta(minutes=CLOSE_HM)
                            ).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        nxt = [k for k in range(12) if _slot_time(k) > end_hm]       # the next half-hour check still to come
        if nxt:
            k = nxt[0]
            t = pd.Timestamp(last_ny.date()).tz_localize(NY) + pd.Timedelta(minutes=_slot_time(k))
            st["next_check_utc"] = t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            if len(slots[k]) >= N_DAYS and prev_close is not None:
                sig = sum(slots[k]) / len(slots[k])
                st["upper"] = round(float(max(d_open, prev_close) * (1 + sig)), 2)
                st["lower"] = round(float(min(d_open, prev_close) * (1 - sig)), 2)
        if pos != 0:
            st["position"] = {"direction": "buy" if pos == 1 else "sell", "entry": round(float(entry), 2),
                              "opened_utc": (idx[entry_i] + pd.Timedelta(minutes=5)).isoformat() + "Z"}
            if st["upper"] is not None and vw is not None:
                st["stop"] = round(float(max(st["upper"], vw) if pos == 1 else min(st["lower"], vw)), 2)
    else:
        # next session open, 09:30 New York on the next weekday after the last candle
        d = last_ny.normalize()
        if end_hm > OPEN_HM:
            d += pd.Timedelta(days=1)
        while d.weekday() >= 5:
            d += pd.Timedelta(days=1)
        st["next_open_utc"] = (d + pd.Timedelta(minutes=OPEN_HM)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"trades": trades, "state": st}
