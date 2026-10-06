"""Optional: keeps the journals, the phone subscriptions and the alert log in a PRIVATE Hugging Face dataset, so a
restart of the free Space loses nothing. Off unless HF_TOKEN and DATA_REPO are set (Space secrets)."""
from __future__ import annotations

import glob
import os
import time

PATTERNS = ["paper_journal.csv", "forward_*.csv", "push_subs.json", "events.jsonl", "server_keys.json",
            "user_settings.json"]


class Persist:
    EVERY = 300

    def __init__(self, cfg: dict, data_dir: str, log=print):
        self.token = cfg.get("HF_TOKEN", "")
        self.repo = cfg.get("DATA_REPO", "")
        self.dir = data_dir
        self.log = log
        if self.token and not self.repo:                 # default: <your HF name>/tradecore-data
            try:
                from huggingface_hub import HfApi
                self.repo = f"{HfApi(token=self.token).whoami()['name']}/tradecore-data"
            except Exception as e:
                log(f"storage: HF_TOKEN not accepted ({str(e)[:80]})")
        self.on = bool(self.token and self.repo)
        self._last = 0.0
        self._sig = None

    def _files(self):
        out = []
        for p in PATTERNS:
            out += glob.glob(os.path.join(self.dir, p))
        return sorted(out)

    def _signature(self):
        return tuple((f, os.path.getsize(f), int(os.path.getmtime(f))) for f in self._files())

    def pull(self):
        if not self.on:
            self.log("storage: local only (set HF_TOKEN + DATA_REPO to keep journals across restarts)")
            return
        try:
            from huggingface_hub import HfApi, snapshot_download
            HfApi(token=self.token).create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
            snapshot_download(self.repo, repo_type="dataset", local_dir=self.dir, token=self.token,
                              allow_patterns=PATTERNS)
            self._sig = self._signature()
            self.log(f"storage: restored {len(self._files())} file(s) from {self.repo}")
        except Exception as e:
            self.log(f"storage: could not restore ({str(e)[:120]})")

    def maybe_push(self):
        if not self.on or time.time() - self._last < self.EVERY:
            return
        self._last = time.time()
        sig = self._signature()
        if sig == self._sig or not sig:
            return
        try:
            from huggingface_hub import HfApi
            HfApi(token=self.token).upload_folder(folder_path=self.dir, repo_id=self.repo, repo_type="dataset",
                                                  allow_patterns=PATTERNS, commit_message="tradecore state")
            self._sig = sig
        except Exception as e:
            self.log(f"storage: save failed ({str(e)[:120]})")
