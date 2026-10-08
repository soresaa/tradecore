package ai.tradecore.app;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.webkit.CookieManager;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Keeps one live line open to your TRADECORE server (no Chrome, no Google push): it asks "anything new?" and the
 * server answers the moment an alert happens (or after 50 s with nothing). Each new alert becomes a loud
 * notification. Runs in the background and after a phone restart.
 */
public class AlertService extends Service {
    static final String CH_SIGNALS = "signals";
    static final String CH_RUNNING = "running";
    static final String EXNESS = "com.exness.android.pa";
    private volatile boolean running = false;

    public static void start(Context c) {
        Intent i = new Intent(c, AlertService.class);
        try {
            if (Build.VERSION.SDK_INT >= 26) {
                c.startForegroundService(i);
            } else {
                c.startService(i);
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        makeChannels();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Notification n = builder(CH_RUNNING)
                .setContentTitle("TRADECORE is watching your strategies")
                .setContentText("Trade alerts will pop up here (paper only)")
                .setSmallIcon(R.drawable.ic_notify)
                .setOngoing(true)
                .setContentIntent(openApp())
                .build();
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        } else {
            startForeground(1, n);
        }
        if (!running) {
            running = true;
            new Thread(this::loop, "tradecore-alerts").start();
        }
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        running = false;
        super.onDestroy();
    }

    private Notification.Builder builder(String channel) {
        return Build.VERSION.SDK_INT >= 26 ? new Notification.Builder(this, channel) : new Notification.Builder(this);
    }

    private void makeChannels() {
        if (Build.VERSION.SDK_INT < 26) {
            return;
        }
        NotificationManager nm = getSystemService(NotificationManager.class);
        NotificationChannel signals = new NotificationChannel(CH_SIGNALS, "Trade signals", NotificationManager.IMPORTANCE_HIGH);
        signals.setDescription("New trades, TP1 and exits");
        signals.enableVibration(true);
        signals.setVibrationPattern(new long[]{0, 300, 150, 300, 150, 600});
        nm.createNotificationChannel(signals);
        NotificationChannel run = new NotificationChannel(CH_RUNNING, "App running", NotificationManager.IMPORTANCE_MIN);
        run.setDescription("Shows that TRADECORE is watching");
        nm.createNotificationChannel(run);
    }

    private PendingIntent openApp() {
        Intent i = new Intent(this, MainActivity.class);
        i.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(this, 0, i, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
    }

    @SuppressWarnings("deprecation")
    private void show(int id, String title, String body) {
        Notification.Builder b = builder(CH_SIGNALS)
                .setContentTitle(title)
                .setContentText(body)
                .setStyle(new Notification.BigTextStyle().bigText(body))
                .setSmallIcon(R.drawable.ic_notify)
                .setAutoCancel(true)
                .setContentIntent(openApp());
        if (Build.VERSION.SDK_INT < 26) {
            b.setPriority(Notification.PRIORITY_HIGH).setDefaults(Notification.DEFAULT_ALL);
        }
        Intent exness = getPackageManager().getLaunchIntentForPackage(EXNESS);
        if (exness != null && !title.contains("summary") && !title.contains("log in")) {   // a trade alert: one tap to Exness
            exness.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            PendingIntent pi = PendingIntent.getActivity(this, 7, exness, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
            b.addAction(new Notification.Action.Builder(
                    android.graphics.drawable.Icon.createWithResource(this, R.drawable.ic_notify), "Open Exness", pi).build());
        }
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        nm.notify(id, b.build());
    }

    /** The phone's alarm sound (alarm volume, also when the ringer is silent) for 8 seconds. */
    private void playAlarm() {
        new Thread(() -> {
            try {
                android.net.Uri u = android.media.RingtoneManager.getDefaultUri(android.media.RingtoneManager.TYPE_ALARM);
                if (u == null) {
                    u = android.media.RingtoneManager.getDefaultUri(android.media.RingtoneManager.TYPE_NOTIFICATION);
                }
                android.media.Ringtone r = android.media.RingtoneManager.getRingtone(getApplicationContext(), u);
                if (r == null) {
                    return;
                }
                r.setAudioAttributes(new android.media.AudioAttributes.Builder()
                        .setUsage(android.media.AudioAttributes.USAGE_ALARM)
                        .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build());
                r.play();
                Thread.sleep(8000);
                r.stop();
            } catch (Exception ignored) {
            }
        }, "tradecore-alarm").start();
    }

    private void loop() {
        SharedPreferences prefs = getSharedPreferences("tc", MODE_PRIVATE);
        long last = prefs.getLong("last_event", -1);
        int backoff = 5;
        boolean toldToLogIn = false;
        while (running) {
            try {
                String cookie = CookieManager.getInstance().getCookie(MainActivity.BASE);
                if (cookie == null || cookie.isEmpty()) {
                    Thread.sleep(30000);                 // not logged in yet
                    continue;
                }
                boolean init = !prefs.contains("last_event");     // first run: learn the newest number only
                long t0 = System.currentTimeMillis();
                URL url = new URL(MainActivity.BASE + "/api/events?" + (init ? "init=1" : "after=" + last + "&wait=50"));
                HttpURLConnection c = (HttpURLConnection) url.openConnection();
                c.setConnectTimeout(20000);
                c.setReadTimeout(90000);
                c.setRequestProperty("Cookie", cookie);
                c.setRequestProperty("User-Agent", "TradecoreApp/1");
                int code = c.getResponseCode();
                if (code == 401) {
                    c.disconnect();
                    if (!toldToLogIn) {
                        show(2, "TRADECORE: please log in", "Open the app and log in again to keep getting alerts.");
                        toldToLogIn = true;
                    }
                    Thread.sleep(60000);
                    continue;
                }
                if (code != 200) {
                    c.disconnect();
                    Thread.sleep(backoff * 1000L);
                    backoff = Math.min(backoff * 2, 120);
                    continue;
                }
                StringBuilder sb = new StringBuilder();
                try (BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream(), "UTF-8"))) {
                    String line;
                    while ((line = r.readLine()) != null) {
                        sb.append(line);
                    }
                }
                c.disconnect();
                JSONObject j = new JSONObject(sb.toString());
                JSONArray events = j.optJSONArray("events");
                int shown = 0;
                boolean ring = false;
                if (!init && events != null) {
                    for (int k = 0; k < events.length(); k++) {
                        JSONObject e = events.getJSONObject(k);
                        show(100 + (int) (e.optLong("id") % 1000), e.optString("title"), e.optString("body"));
                        ring = ring || e.optBoolean("alarm", false);
                        shown++;
                    }
                }
                if (ring) {
                    playAlarm();                         // "Alarm sound" is on and this is a trade / price alert
                }
                last = j.optLong("last", last);
                prefs.edit().putLong("last_event", last).apply();
                backoff = 5;
                toldToLogIn = false;
                if (!init && shown == 0 && System.currentTimeMillis() - t0 < 2000) {
                    Thread.sleep(5000);                  // the server answered at once with nothing: do not hammer it
                }
            } catch (InterruptedException ie) {
                return;
            } catch (Exception ex) {
                try {
                    Thread.sleep(backoff * 1000L);
                } catch (InterruptedException ie) {
                    return;
                }
                backoff = Math.min(backoff * 2, 120);
            }
        }
    }
}
