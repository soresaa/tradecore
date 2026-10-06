"""
TRADECORE cloud app: the web server. One password protects everything except the app icon files and /ping (the
keep-awake check). The dashboard is the desktop app's own page (app.html) with a small bridge that asks this server
instead of the desktop window, plus "Turn on alerts" (Web Push) and "install as an app" support.

Run:  python -m cloud.server          (Hugging Face starts it from the Dockerfile)
"""
from __future__ import annotations

import hmac
import os
import sys
import time
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, Response, jsonify, redirect, request, send_from_directory, session  # noqa: E402

from cloud import config  # noqa: E402

CFG = config.load()
STATIC = os.path.join(HERE, "static")
app = Flask(__name__, static_folder=None)
app.secret_key = CFG.get("SECRET_KEY") or os.urandom(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=CFG.get("SECURE_COOKIE", "1") == "1",
                  PERMANENT_SESSION_LIFETIME=timedelta(days=90))
ENGINE = None
_FAILS = []          # times of wrong passwords (simple brake against guessing)


def engine():
    global ENGINE
    if ENGINE is None:
        from cloud.engine import CloudEngine
        ENGINE = CloudEngine(CFG)
        if CFG.get("SECRET_KEY"):
            app.secret_key = CFG["SECRET_KEY"]
    return ENGINE


def logged_in() -> bool:
    return session.get("ok") is True


LOGIN_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TRADECORE AI</title>
<link rel="manifest" href="/manifest.webmanifest"><meta name="theme-color" content="#0b1020">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b1020;color:#e8ecf5;
font:16px system-ui,-apple-system,Segoe UI,Roboto,sans-serif}form{width:min(340px,90vw);background:#141b33;padding:28px;
border-radius:16px;box-shadow:0 10px 40px #0008}h1{margin:0 0 4px;font-size:22px}p{margin:0 0 18px;color:#9aa6c4;font-size:14px}
input,button{width:100%;box-sizing:border-box;padding:13px 14px;border-radius:10px;font-size:16px}
input{border:1px solid #2a355e;background:#0b1020;color:#e8ecf5;margin-bottom:12px}
button{border:0;background:#35d07f;color:#04210f;font-weight:700}.e{color:#ff6b78;font-size:14px;margin-top:10px}</style>
</head><body><form method="post"><h1>TRADECORE AI</h1><p>Your signals, paper only.</p>
<input type="password" name="password" placeholder="Password" autofocus autocomplete="current-password">
<button>Open</button>{err}</form></body></html>"""

BRIDGE = """
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<link rel="manifest" href="/manifest.webmanifest"><meta name="theme-color" content="#0b1020">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<style>#btnFolder{display:none!important}
@media (max-width:700px){ .sumrow{grid-template-columns:repeat(3,1fr)!important;gap:6px!important}
  .sum{padding:9px 10px!important} .sum .v{font-size:18px!important} .sum .s{font-size:10px!important}
  .sum .k{font-size:9px!important;letter-spacing:.4px!important} }
#tcCloud{position:sticky;top:0;z-index:50;display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 12px;
background:#141b33;border-bottom:1px solid #2a355e;font:13px system-ui,sans-serif;color:#cfd7ea}
#tcCloud button{border:0;border-radius:8px;padding:8px 12px;font-weight:700;background:#35d07f;color:#04210f}
#tcCloud .muted{color:#8a96b5}#tcCloud a{color:#8a96b5;margin-left:auto}</style>
<script>
(function () {
  const j = (u, o) => fetch(u, Object.assign({credentials: 'same-origin'}, o || {})).then(r => {
    if (r.status === 401) { location.href = '/login'; throw new Error('login'); } return r.json(); });
  const post = (u, b) => j(u, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b || {})});
  window.pywebview = { api: {
    get_state: () => j('/api/state'),
    start: () => post('/api/start'), stop: () => post('/api/stop'),
    test_alert: () => post('/api/test').then(r => { alert(r.msg); return r; }),
    open_folder: () => Promise.resolve({}),
    save_settings: s => post('/api/settings', s),
  }};
  const _si = window.setInterval.bind(window);
  window.setInterval = (fn, ms, ...a) => _si(fn, ms === 1000 ? 5000 : ms, ...a);   // refresh every 5 s, not 1 s
  window.addEventListener('load', () => window.dispatchEvent(new Event('pywebviewready')));
})();
</script>
"""

ALERTS_BAR = """
<div id="tcCloud"><button id="tcAlerts">Turn on alerts</button><span id="tcStatus" class="muted">checking...</span>
<a href="/tools" style="margin-left:auto;color:#35d07f;font-weight:700">History &amp; settings</a>
<a id="tcExness" href="tradecore://open?pkg=com.exness.android.pa" style="display:none;color:#5ab2ff;font-weight:700;margin-left:10px">Exness</a>
<a href="/logout" style="margin-left:10px">log out</a></div>
<script>
(function () {
  const st = document.getElementById('tcStatus'), btn = document.getElementById('tcAlerts');
  const b64 = s => { const p = '='.repeat((4 - s.length % 4) % 4); const r = atob((s + p).replace(/-/g, '+').replace(/_/g, '/'));
                     return Uint8Array.from([...r].map(c => c.charCodeAt(0))); };
  const standalone = matchMedia('(display-mode: standalone)').matches || navigator.standalone;
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent);
  async function refresh() {
    if (navigator.userAgent.includes('TradecoreApp')) {      // inside the Android app: it shows the alerts itself
      document.getElementById('tcExness').style.display = 'inline';
      st.textContent = 'alerts: handled by the TRADECORE app'; btn.textContent = 'Send test alert'; btn.dataset.on = '1'; return;
    }
    if (!('serviceWorker' in navigator)) { st.textContent = 'this browser cannot get alerts'; btn.style.display = 'none'; return; }
    const reg = await navigator.serviceWorker.register('/sw.js');
    if (ios && !standalone) { st.textContent = 'iPhone: tap Share > Add to Home Screen, then open the app from its icon'; return; }
    if (!('PushManager' in window)) { st.textContent = 'alerts not supported here'; btn.style.display = 'none'; return; }
    const sub = await reg.pushManager.getSubscription();
    if (sub && Notification.permission === 'granted') { st.textContent = 'alerts are ON on this device'; btn.textContent = 'Send test alert'; btn.dataset.on = '1'; }
    else st.textContent = standalone ? 'tap the button to get trade alerts' : 'tip: install this app (browser menu > Install app / Add to Home screen)';
  }
  btn.onclick = async () => {
    try {
      if (btn.dataset.on) { const r = await (await fetch('/api/test', {method: 'POST'})).json(); st.textContent = r.msg; return; }
      const reg = await navigator.serviceWorker.register('/sw.js');
      const perm = await Notification.requestPermission();
      if (perm !== 'granted') { st.textContent = 'alerts blocked - allow notifications for this site in the browser settings'; return; }
      const k = await (await fetch('/api/push/key')).json();
      if (!k.key) { st.textContent = 'the server has no push keys yet (VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY secrets)'; return; }
      const sub = await reg.pushManager.subscribe({userVisibleOnly: true, applicationServerKey: b64(k.key)});
      const r = await (await fetch('/api/push/subscribe', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                            body: JSON.stringify(sub)})).json();
      st.textContent = r.ok ? 'alerts are ON on this device' : ('could not turn on: ' + r.msg);
      if (r.ok) { btn.textContent = 'Send test alert'; btn.dataset.on = '1'; }
    } catch (e) { st.textContent = 'could not turn on alerts: ' + e; }
  };
  refresh().catch(e => st.textContent = String(e));
})();
</script>
"""


@app.after_request
def no_cache(resp):
    if request.path.startswith("/api/") or request.path in ("/", "/sw.js"):
        resp.headers["Cache-Control"] = "no-store"
    if ("gzip" in request.headers.get("Accept-Encoding", "") and resp.status_code == 200 and not resp.direct_passthrough
            and resp.mimetype in ("application/json", "text/html") and "Content-Encoding" not in resp.headers):
        import gzip
        data = resp.get_data()
        if len(data) > 800:
            resp.set_data(gzip.compress(data, 6))
            resp.headers["Content-Encoding"] = "gzip"
            resp.headers["Vary"] = "Accept-Encoding"
    return resp


@app.get("/ping")
def ping():
    engine()
    return "ok"


@app.route("/login", methods=["GET", "POST"])
def login():
    pw = CFG.get("APP_PASSWORD", "")
    if not pw:
        return Response(LOGIN_PAGE.replace("{err}", '<div class="e">Set the APP_PASSWORD secret first.</div>'), 503)
    err = ""
    if request.method == "POST":
        now = time.time()
        _FAILS[:] = [t for t in _FAILS if now - t < 900]
        if len(_FAILS) >= 8:
            err = '<div class="e">Too many wrong passwords - wait 15 minutes.</div>'
        elif hmac.compare_digest(request.form.get("password", "").encode(), pw.encode()):
            session.permanent = True
            session["ok"] = True
            return redirect("/")
        else:
            _FAILS.append(now)
            time.sleep(1.0)
            err = '<div class="e">Wrong password.</div>'
    return LOGIN_PAGE.replace("{err}", err)


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.get("/")
def index():
    if not logged_in():
        return redirect("/login")
    engine()
    html = open(os.path.join(ROOT, "app.html"), encoding="utf-8").read()
    html = html.replace("<head>", "<head>" + BRIDGE, 1)
    html = html.replace("on the Exness demo feed", "on live market prices (Twelve Data, Yahoo, Binance)")
    i = html.find("<body")
    j = html.find(">", i) + 1
    html = html[:j] + ALERTS_BAR + html[j:]
    return Response(html, mimetype="text/html")


def _need_login():
    return None if logged_in() else (jsonify({"error": "login"}), 401)


@app.get("/api/state")
def api_state():
    return _need_login() or jsonify(engine().get_state())


@app.get("/api/events")
def api_events():
    """The Android app's live line (long polling): waits up to `wait` seconds for alerts numbered above `after`
    (-1 = any). init=1 (the app's very first call): no alerts, only the newest number, so it never replays the past."""
    if (r := _need_login()):
        return r
    eng = engine()
    eng.native_seen = time.time()                    # the Android app is listening -> it gets the alerts, not Chrome
    if request.args.get("init") == "1":
        return jsonify({"events": [], "last": eng.last_event_id()})
    try:
        after = int(request.args.get("after", "-1"))
        wait = max(0, min(int(request.args.get("wait", "0")), 50))
    except ValueError:
        return jsonify({"error": "bad request"}), 400
    deadline = time.time() + wait
    while True:
        evs = eng.events_after(after)
        if evs or time.time() >= deadline:
            break
        time.sleep(1)
    eng.native_seen = time.time()
    return jsonify({"events": evs, "last": eng.last_event_id()})


@app.get("/tools")
def tools():
    if not logged_in():
        return redirect("/login")
    return send_from_directory(STATIC, "tools.html", mimetype="text/html")


@app.get("/api/history")
def api_history():
    return _need_login() or jsonify({"events": engine().history(300)})


@app.get("/api/performance")
def api_performance():
    return _need_login() or jsonify({"rows": engine().performance(), "risk_usd": engine().user.get("risk_usd", 1.0)})


@app.route("/api/price_alerts", methods=["GET", "POST"])
def api_price_alerts():
    if (r := _need_login()):
        return r
    eng = engine()
    if request.method == "POST":
        b = request.get_json(force=True) or {}
        try:
            a = eng.add_price_alert(b.get("market", ""), b.get("cond", ""), b.get("level"), b.get("note", ""))
            return jsonify({"ok": True, "alert": a})
        except (ValueError, TypeError) as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
    from cloud.engine import PRICE_MARKETS
    return jsonify({"alerts": list(reversed(eng.price_alerts())), "prices": eng.prices(),
                    "markets": {k: v[1] for k, v in PRICE_MARKETS.items()}})


@app.delete("/api/price_alerts/<int:aid>")
def api_price_alert_delete(aid):
    if (r := _need_login()):
        return r
    engine().delete_price_alert(aid)
    return jsonify({"ok": True})


@app.get("/api/news")
def api_news():
    if (r := _need_login()):
        return r
    eng = engine()
    return jsonify({"news": eng.news(), "on": eng.user.get("news_alerts", True), "minutes": eng.user.get("news_minutes", 30)})


@app.route("/api/user_settings", methods=["GET", "POST"])
def api_user_settings():
    if (r := _need_login()):
        return r
    from cloud.engine import NAMES
    eng = engine()
    if request.method == "POST":
        try:
            s = eng.save_user(request.get_json(force=True) or {})
            return jsonify({"ok": True, "settings": s})
        except ValueError as e:
            return jsonify({"ok": False, "msg": str(e)}), 400
    return jsonify({"settings": eng.user, "names": NAMES})


@app.post("/api/start")
def api_start():
    if (r := _need_login()):
        return r
    engine().running = True
    return jsonify({"ok": True, "msg": "running"})


@app.post("/api/stop")
def api_stop():
    if (r := _need_login()):
        return r
    engine().running = False
    return jsonify({"ok": True, "msg": "paused"})


@app.post("/api/test")
def api_test():
    return _need_login() or jsonify(engine().test_alert())


@app.post("/api/settings")
def api_settings():
    return _need_login() or jsonify({"ok": False, "msg": "The cloud app's settings are its Hugging Face secrets/variables."})


@app.get("/api/push/key")
def push_key():
    return _need_login() or jsonify({"key": CFG.get("VAPID_PUBLIC_KEY", "")})


@app.post("/api/push/subscribe")
def push_subscribe():
    if (r := _need_login()):
        return r
    try:
        n = engine().push.add(request.get_json(force=True))
        engine().log(f"phone alerts turned on ({n} device(s))")
        return jsonify({"ok": True, "devices": n})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 400


@app.get("/sw.js")
def sw():
    r = send_from_directory(STATIC, "sw.js", mimetype="application/javascript")
    r.headers["Service-Worker-Allowed"] = "/"
    return r


@app.get("/manifest.webmanifest")
def manifest():
    return send_from_directory(STATIC, "manifest.webmanifest", mimetype="application/manifest+json")


@app.get("/<name>.png")
def icons(name):
    if name not in ("icon-192", "icon-512", "apple-touch-icon"):
        return ("", 404)
    return send_from_directory(STATIC, f"{name}.png", mimetype="image/png")


def _keep_awake():
    import threading
    import requests
    url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("KEEP_AWAKE_URL")
    if not url:
        return

    def loop():
        while True:
            time.sleep(600)
            try:
                requests.get(url.rstrip("/") + "/ping", timeout=30)
            except Exception:
                pass
    threading.Thread(target=loop, name="keep-awake", daemon=True).start()


def main():
    from waitress import serve
    port = int(os.environ.get("PORT") or CFG.get("PORT", "7860"))
    engine()                                   # start the strategies right away, not on the first visit
    _keep_awake()
    print(f"TRADECORE cloud on port {port}", flush=True)
    serve(app, host="0.0.0.0", port=port, threads=8)


if __name__ == "__main__":
    main()
