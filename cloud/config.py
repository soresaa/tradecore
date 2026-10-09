"""Settings of the cloud app. On Hugging Face they are the Space's SECRETS / VARIABLES (environment variables);
for a local test they can also sit in cloud/settings.env (KEY=value lines). Nothing secret is ever written to the code."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULTS = {
    "PROVIDER": "free",               # free = Twelve Data (gold, USD/JPY) + Yahoo (NAS100 index) + Binance (BTC);
                                      # oanda = an OANDA API token; mt5 = local Windows test only
    "TWELVEDATA_KEY": "",             # free key from twelvedata.com (800 credits a day; the app uses about 600)
    "OANDA_TOKEN": "",
    "OANDA_ENV": "practice",
    "APP_PASSWORD": "",               # the password of your app (required)
    "SECRET_KEY": "",                 # random text that signs your login cookie (make_keys.py makes one)
    "VAPID_PUBLIC_KEY": "",           # push-notification keys (make_keys.py makes them)
    "VAPID_PRIVATE_KEY": "",
    "VAPID_EMAIL": "mailto:tradecore@example.com",
    "HF_TOKEN": "",                   # optional: keeps journals + phone subscriptions in a private HF dataset
    "DATA_REPO": "",                  # optional: e.g. yourname/tradecore-data
    "DATA_DIR": "/tmp/tradecore-data",
    "PORT": "7860",
    "MAIN_DECISION_MINUTES": "5",     # the gold 1h trend strategy: 5 = your desktop setting, 60 = the tested live rule
    "MARKETS": "XAUUSD_BO4H_BIG,XAUUSD_RC2,BTCUSD_BO1H,NAS100_NOISE,USDJPY_BO4H_BIG",   # 2026-10-09: the followed set only
    "SECURE_COOKIE": "1",             # 0 only for a local http test
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    path = os.path.join(HERE, "settings.env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    for k in DEFAULTS:
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg
