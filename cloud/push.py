"""Web Push to the TRADECORE phone / computer app (no Telegram, no third-party app): the browser gives us a
subscription when you tap "Turn on alerts"; the server signs each message with its VAPID key and sends it."""
from __future__ import annotations

import json
import os
import threading


class Push:
    def __init__(self, cfg: dict, data_dir: str, log=print):
        self.pub = cfg.get("VAPID_PUBLIC_KEY", "")
        self.priv = cfg.get("VAPID_PRIVATE_KEY", "")
        self.email = cfg.get("VAPID_EMAIL", "mailto:tradecore@example.com")
        self.path = os.path.join(data_dir, "push_subs.json")
        self.log = log
        self.lock = threading.Lock()
        self.subs = []
        if os.path.exists(self.path):
            try:
                self.subs = json.load(open(self.path, encoding="utf-8"))
            except Exception:
                self.subs = []

    @property
    def ready(self) -> bool:
        return bool(self.pub and self.priv)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.subs, f)

    def add(self, sub: dict) -> int:
        if not isinstance(sub, dict) or "endpoint" not in sub or "keys" not in sub:
            raise ValueError("not a push subscription")
        with self.lock:
            self.subs = [s for s in self.subs if s.get("endpoint") != sub["endpoint"]] + [sub]
            self._save()
            return len(self.subs)

    def send(self, title: str, body: str, tag: str = "tradecore", url: str = "/") -> int:
        if not self.ready:
            self.log("push: VAPID keys missing - no phone alert sent")
            return 0
        from pywebpush import webpush, WebPushException
        data = json.dumps({"title": title, "body": body, "tag": tag, "url": url})
        sent, keep = 0, []
        with self.lock:
            subs = list(self.subs)
        for s in subs:
            try:
                webpush(subscription_info=s, data=data, vapid_private_key=self.priv,
                        vapid_claims={"sub": self.email}, ttl=3600)
                sent += 1
                keep.append(s)
            except WebPushException as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (404, 410):
                    self.log("push: a phone/browser unsubscribed - removed")
                else:
                    keep.append(s)
                    self.log(f"push failed ({code}): {str(e)[:120]}")
            except Exception as e:
                keep.append(s)
                self.log(f"push failed: {str(e)[:120]}")
        with self.lock:
            if len(keep) != len(self.subs):
                self.subs = keep
                self._save()
        return sent
