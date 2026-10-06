package ai.tradecore.app;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.provider.Settings;
import android.webkit.CookieManager;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

/** The TRADECORE screen: your server's dashboard, full screen, logged in once (the cookie is kept). */
public class MainActivity extends Activity {
    public static final String BASE = "https://tradecore-k8kr.onrender.com";
    private WebView web;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(Color.parseColor("#0b1020"));
        getWindow().setNavigationBarColor(Color.parseColor("#0b1020"));
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0b1020"));
        setContentView(web);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setUserAgentString(s.getUserAgentString() + " TradecoreApp/1");
        CookieManager.getInstance().setAcceptCookie(true);

        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri u = request.getUrl();
                if ("tradecore".equals(u.getScheme())) {         // tradecore://open?pkg=... -> open that app
                    openApp(MainActivity.this, u.getQueryParameter("pkg"));
                    return true;
                }
                String host = u.getHost();
                if (host != null && Uri.parse(BASE).getHost().equals(host)) {
                    return false;                        // your app's own pages stay inside the app
                }
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, u));
                } catch (Exception ignored) {
                }
                return true;
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                CookieManager.getInstance().flush();     // keep the login for the alert service
                AlertService.start(MainActivity.this);
            }
        });

        if (state == null) {
            web.loadUrl(BASE + "/");
        } else {
            web.restoreState(state);
        }
        if (Build.VERSION.SDK_INT >= 33) {
            requestPermissions(new String[]{"android.permission.POST_NOTIFICATIONS"}, 1);
        }
        askToIgnoreBatterySaver();
        AlertService.start(this);
    }

    /** Opens an installed app (Exness, MT5, TradingView); if it is missing, its Play Store page. */
    static void openApp(android.content.Context c, String pkg) {
        if (pkg == null || pkg.isEmpty()) {
            return;
        }
        Intent i = c.getPackageManager().getLaunchIntentForPackage(pkg);
        if (i == null) {
            i = new Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=" + pkg));
        }
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            c.startActivity(i);
        } catch (Exception ignored) {
        }
    }

    /** Once: ask Android not to put the alert service to sleep (otherwise phones like Xiaomi stop it). */
    private void askToIgnoreBatterySaver() {
        SharedPreferences p = getSharedPreferences("tc", MODE_PRIVATE);
        if (p.getBoolean("asked_battery", false)) {
            return;
        }
        p.edit().putBoolean("asked_battery", true).apply();
        PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
        if (pm != null && !pm.isIgnoringBatteryOptimizations(getPackageName())) {
            try {
                Intent i = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
                i.setData(Uri.parse("package:" + getPackageName()));
                startActivity(i);
            } catch (Exception ignored) {
            }
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle out) {
        super.onSaveInstanceState(out);
        web.saveState(out);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) {
            web.goBack();
        } else {
            moveTaskToBack(true);                        // keep running; the alerts keep coming
        }
    }
}
