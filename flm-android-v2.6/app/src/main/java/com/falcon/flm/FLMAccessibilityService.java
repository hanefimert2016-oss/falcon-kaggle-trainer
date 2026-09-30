package com.falcon.flm;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Path;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.text.TextUtils;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class FLMAccessibilityService extends AccessibilityService {
    private static volatile FLMAccessibilityService INSTANCE;

    private static final Map<String, String> PACKAGES = new HashMap<>();
    static {
        PACKAGES.put("chrome", "com.android.chrome");
        PACKAGES.put("browser", "com.android.chrome");
        PACKAGES.put("tarayıcı", "com.android.chrome");
        PACKAGES.put("tarayici", "com.android.chrome");
        PACKAGES.put("youtube", "com.google.android.youtube");
        PACKAGES.put("gmail", "com.google.android.gm");
        PACKAGES.put("photos", "com.google.android.apps.photos");
        PACKAGES.put("fotoğraflar", "com.google.android.apps.photos");
        PACKAGES.put("fotograflar", "com.google.android.apps.photos");
        PACKAGES.put("play store", "com.android.vending");
        PACKAGES.put("playstore", "com.android.vending");
        PACKAGES.put("settings", "com.android.settings");
        PACKAGES.put("ayarlar", "com.android.settings");
    }

    @Override protected void onServiceConnected() {
        super.onServiceConnected();
        INSTANCE = this;
    }

    @Override public void onAccessibilityEvent(AccessibilityEvent event) {}
    @Override public void onInterrupt() {}

    @Override public boolean onUnbind(Intent intent) {
        if (INSTANCE == this) INSTANCE = null;
        return super.onUnbind(intent);
    }

    @Override public void onDestroy() {
        if (INSTANCE == this) INSTANCE = null;
        super.onDestroy();
    }

    public static boolean isEnabled(Context context) {
        String enabled = Settings.Secure.getString(
                context.getContentResolver(),
                Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        );
        if (enabled == null) return false;
        ComponentName me = new ComponentName(context, FLMAccessibilityService.class);
        TextUtils.SimpleStringSplitter splitter = new TextUtils.SimpleStringSplitter(':');
        splitter.setString(enabled);
        while (splitter.hasNext()) {
            ComponentName cn = ComponentName.unflattenFromString(splitter.next());
            if (me.equals(cn)) return true;
        }
        return false;
    }

    public static boolean isConnected() {
        return INSTANCE != null;
    }

    public static JSONObject executeNow(JSONObject action) {
        FLMAccessibilityService service = INSTANCE;
        if (service == null) {
            return failStatic("accessibility_not_connected", "FLM Telefon Kontrolü etkin değil.");
        }
        return service.executeAction(action == null ? new JSONObject() : action);
    }

    public static JSONObject observeNow() {
        FLMAccessibilityService service = INSTANCE;
        if (service == null) {
            return failStatic("accessibility_not_connected", "FLM Telefon Kontrolü etkin değil.");
        }
        return service.observe();
    }

    private static JSONObject failStatic(String error, String message) {
        JSONObject out = new JSONObject();
        try {
            out.put("ok", false);
            out.put("error", error);
            out.put("message", message);
        } catch (Exception ignored) {}
        return out;
    }

    private JSONObject ok(String action) throws Exception {
        JSONObject r = new JSONObject();
        r.put("ok", true);
        r.put("action", action);
        return r;
    }

    private JSONObject fail(String error, String message) {
        return failStatic(error, message == null ? "" : message);
    }

    private String normalize(String x) {
        return x == null ? "" : x.trim().toLowerCase(new Locale("tr", "TR"));
    }

    private JSONObject executeAction(JSONObject a) {
        String action = a.optString("action", "");
        try {
            switch (action) {
                case "observe": {
                    JSONObject r = ok(action);
                    r.put("observation", observe());
                    return r;
                }
                case "open_app":
                    return openApp(a.optString("target", a.optString("package", "")));
                case "open_url":
                    return openUrl(a.optString("url", a.optString("target", "")));
                case "tap":
                    return tap(a.optInt("x", 0), a.optInt("y", 0));
                case "click":
                    return clickText(a.optString("target", ""));
                case "type_text":
                    return typeText(a.optString("text", ""));
                case "press_key":
                    return pressKey(a.optString("key", ""));
                case "scroll":
                    return scroll(a.optString("target", "down"), a.optInt("duration_ms", 450));
                case "wait":
                    Thread.sleep((long) (Math.max(0.0, a.optDouble("seconds", 0.5)) * 1000));
                    return ok("wait");
                case "inspect_and_resolve":
                    return clickText(a.optString("target", ""));
                case "close_app":
                    return fail("unsupported_rootless_action",
                            "Android, normal Accessibility uygulamalarına başka uygulamaları force-stop etme izni vermez.");
                default:
                    return fail("unsupported_action", action);
            }
        } catch (Exception e) {
            return fail(e.getClass().getSimpleName(), e.getMessage());
        }
    }

    private JSONObject openUrl(String url) throws Exception {
        if (url == null || url.trim().isEmpty()) return fail("url_required", "URL boş");
        if (!url.matches("(?i)^https?://.*")) url = "https://" + url;
        Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(i);
        JSONObject r = ok("open_url");
        r.put("url", url);
        return r;
    }

    private String resolvePackage(String target) {
        String t = normalize(target);
        if (PACKAGES.containsKey(t)) return PACKAGES.get(t);
        if (t.contains(".")) return t;
        try {
            PackageManager pm = getPackageManager();
            Intent launcher = new Intent(Intent.ACTION_MAIN, null);
            launcher.addCategory(Intent.CATEGORY_LAUNCHER);
            for (android.content.pm.ResolveInfo info : pm.queryIntentActivities(launcher, 0)) {
                String label = String.valueOf(info.loadLabel(pm));
                String pkg = info.activityInfo.packageName;
                String n = normalize(label);
                if (n.equals(t) || (!t.isEmpty() && n.contains(t))) return pkg;
            }
        } catch (Exception ignored) {}
        return null;
    }

    private JSONObject openApp(String target) throws Exception {
        String pkg = resolvePackage(target);
        if (pkg == null) return fail("package_not_found", target);
        Intent launch = getPackageManager().getLaunchIntentForPackage(pkg);
        if (launch == null) return fail("not_launchable", pkg);
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(launch);
        JSONObject r = ok("open_app");
        r.put("package", pkg);
        return r;
    }

    private JSONObject tap(int x, int y) throws Exception {
        if (x < 0 || y < 0) return fail("bad_coordinates", x + "," + y);
        boolean dispatched = dispatchTap(x, y, 60);
        if (!dispatched) return fail("gesture_failed", "tap");
        JSONObject r = ok("tap");
        r.put("x", x);
        r.put("y", y);
        return r;
    }

    private boolean dispatchTap(int x, int y, long durationMs) throws InterruptedException {
        Path p = new Path();
        p.moveTo(x, y);
        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(p, 0, Math.max(1, durationMs));
        GestureDescription gesture = new GestureDescription.Builder().addStroke(stroke).build();
        CountDownLatch latch = new CountDownLatch(1);
        final boolean[] success = new boolean[] { false };
        boolean accepted = dispatchGesture(gesture, new GestureResultCallback() {
            @Override public void onCompleted(GestureDescription gestureDescription) {
                success[0] = true;
                latch.countDown();
            }
            @Override public void onCancelled(GestureDescription gestureDescription) {
                latch.countDown();
            }
        }, null);
        if (!accepted) return false;
        latch.await(3, TimeUnit.SECONDS);
        return success[0];
    }

    private AccessibilityNodeInfo findByText(String target) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return null;
        String want = normalize(target);
        ArrayDeque<AccessibilityNodeInfo> q = new ArrayDeque<>();
        q.add(root);
        AccessibilityNodeInfo fuzzy = null;
        int visited = 0;
        while (!q.isEmpty() && visited++ < 700) {
            AccessibilityNodeInfo n = q.removeFirst();
            String text = n.getText() == null ? "" : n.getText().toString();
            String desc = n.getContentDescription() == null ? "" : n.getContentDescription().toString();
            String hay = normalize((text + " " + desc).trim());
            if (!hay.isEmpty()) {
                if (hay.equals(want)) return n;
                if (fuzzy == null && !want.isEmpty() && (hay.contains(want) || want.contains(hay))) {
                    fuzzy = n;
                }
            }
            for (int i = 0; i < n.getChildCount(); i++) {
                AccessibilityNodeInfo child = n.getChild(i);
                if (child != null) q.addLast(child);
            }
        }
        return fuzzy;
    }

    private boolean clickNode(AccessibilityNodeInfo n) throws Exception {
        AccessibilityNodeInfo current = n;
        for (int depth = 0; current != null && depth < 6; depth++) {
            if (current.isClickable() && current.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                return true;
            }
            current = current.getParent();
        }
        Rect b = new Rect();
        n.getBoundsInScreen(b);
        return !b.isEmpty() && dispatchTap(b.centerX(), b.centerY(), 60);
    }

    private JSONObject clickText(String target) throws Exception {
        if (target == null || target.trim().isEmpty()) return fail("target_required", "hedef boş");
        AccessibilityNodeInfo n = findByText(target);
        if (n == null) return fail("ui_target_not_found", target);
        Rect b = new Rect();
        n.getBoundsInScreen(b);
        boolean done = clickNode(n);
        if (!done) return fail("click_failed", target);
        JSONObject r = ok("click");
        r.put("target", target);
        r.put("x", b.centerX());
        r.put("y", b.centerY());
        return r;
    }

    private AccessibilityNodeInfo focusedEditable() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return null;
        AccessibilityNodeInfo focus = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT);
        if (focus != null && focus.isEditable()) return focus;

        ArrayDeque<AccessibilityNodeInfo> q = new ArrayDeque<>();
        q.add(root);
        int visited = 0;
        AccessibilityNodeInfo firstEditable = null;
        while (!q.isEmpty() && visited++ < 600) {
            AccessibilityNodeInfo n = q.removeFirst();
            if (n.isEditable()) {
                if (n.isFocused()) return n;
                if (firstEditable == null) firstEditable = n;
            }
            for (int i = 0; i < n.getChildCount(); i++) {
                AccessibilityNodeInfo child = n.getChild(i);
                if (child != null) q.addLast(child);
            }
        }
        return firstEditable;
    }

    private JSONObject typeText(String text) throws Exception {
        AccessibilityNodeInfo n = focusedEditable();
        if (n == null) return fail("no_editable_focus",
                "Önce bir metin alanına dokun veya FLM'den alanın etiketine tıklamasını iste.");
        if (!n.isFocused()) n.performAction(AccessibilityNodeInfo.ACTION_FOCUS);
        Bundle b = new Bundle();
        b.putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                text == null ? "" : text
        );
        boolean done = n.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, b);
        if (!done) return fail("set_text_failed", "ACTION_SET_TEXT reddedildi");
        JSONObject r = ok("type_text");
        r.put("characters", text == null ? 0 : text.length());
        return r;
    }

    private JSONObject pressKey(String key) throws Exception {
        String k = key == null ? "" : key.trim().toUpperCase(Locale.ROOT);
        int global = -1;
        if ("BACK".equals(k)) global = GLOBAL_ACTION_BACK;
        else if ("HOME".equals(k)) global = GLOBAL_ACTION_HOME;
        else if ("RECENTS".equals(k) || "OVERVIEW".equals(k)) global = GLOBAL_ACTION_RECENTS;

        if (global != -1) {
            boolean done = performGlobalAction(global);
            if (!done) return fail("global_action_failed", k);
            JSONObject r = ok("press_key");
            r.put("key", k);
            return r;
        }

        if ("ENTER".equals(k) || "RETURN".equals(k)) {
            AccessibilityNodeInfo n = focusedEditable();
            if (n == null) return fail("no_editable_focus", "Enter için odakta bir metin alanı gerekli.");
            if (Build.VERSION.SDK_INT >= 30) {
                boolean done = n.performAction(
                        AccessibilityNodeInfo.AccessibilityAction.ACTION_IME_ENTER.getId()
                );
                if (done) {
                    JSONObject r = ok("press_key");
                    r.put("key", "ENTER");
                    return r;
                }
            }
            return fail("ime_enter_unsupported",
                    "Açık uygulama Accessibility üzerinden Enter eylemini sunmadı.");
        }

        return fail("unsupported_rootless_key", k);
    }

    private AccessibilityNodeInfo findScrollable(AccessibilityNodeInfo root) {
        if (root == null) return null;
        ArrayDeque<AccessibilityNodeInfo> q = new ArrayDeque<>();
        q.add(root);
        int visited = 0;
        while (!q.isEmpty() && visited++ < 600) {
            AccessibilityNodeInfo n = q.removeFirst();
            if (n.isScrollable()) return n;
            for (int i = 0; i < n.getChildCount(); i++) {
                AccessibilityNodeInfo child = n.getChild(i);
                if (child != null) q.addLast(child);
            }
        }
        return null;
    }

    private JSONObject scroll(String direction, int durationMs) throws Exception {
        boolean up = "up".equalsIgnoreCase(direction)
                || "yukari".equalsIgnoreCase(direction)
                || "yukarı".equalsIgnoreCase(direction);

        AccessibilityNodeInfo scrollable = findScrollable(getRootInActiveWindow());
        if (scrollable != null) {
            int action = up
                    ? AccessibilityNodeInfo.ACTION_SCROLL_BACKWARD
                    : AccessibilityNodeInfo.ACTION_SCROLL_FORWARD;
            if (scrollable.performAction(action)) {
                JSONObject r = ok("scroll");
                r.put("direction", up ? "up" : "down");
                return r;
            }
        }

        android.util.DisplayMetrics dm = getResources().getDisplayMetrics();
        int x = dm.widthPixels / 2;
        int y1 = up ? (int) (dm.heightPixels * .35) : (int) (dm.heightPixels * .75);
        int y2 = up ? (int) (dm.heightPixels * .75) : (int) (dm.heightPixels * .35);

        Path p = new Path();
        p.moveTo(x, y1);
        p.lineTo(x, y2);
        GestureDescription.StrokeDescription stroke =
                new GestureDescription.StrokeDescription(p, 0, Math.max(180, durationMs));
        CountDownLatch latch = new CountDownLatch(1);
        final boolean[] success = new boolean[] { false };
        boolean accepted = dispatchGesture(
                new GestureDescription.Builder().addStroke(stroke).build(),
                new GestureResultCallback() {
                    @Override public void onCompleted(GestureDescription gestureDescription) {
                        success[0] = true;
                        latch.countDown();
                    }
                    @Override public void onCancelled(GestureDescription gestureDescription) {
                        latch.countDown();
                    }
                },
                null
        );
        if (!accepted) return fail("gesture_failed", "scroll");
        latch.await(3, TimeUnit.SECONDS);
        if (!success[0]) return fail("gesture_cancelled", "scroll");

        JSONObject r = ok("scroll");
        r.put("direction", up ? "up" : "down");
        return r;
    }

    private JSONObject observe() {
        JSONObject out = new JSONObject();
        try {
            out.put("ok", true);
            out.put("kind", "android_accessibility");
            AccessibilityNodeInfo root = getRootInActiveWindow();
            if (root == null) {
                out.put("package", "");
                out.put("nodes", new JSONArray());
                return out;
            }
            out.put("package", String.valueOf(root.getPackageName()));
            JSONArray nodes = new JSONArray();
            ArrayDeque<AccessibilityNodeInfo> q = new ArrayDeque<>();
            q.add(root);
            int visited = 0;

            while (!q.isEmpty() && visited++ < 140) {
                AccessibilityNodeInfo n = q.removeFirst();
                boolean password = n.isPassword();
                String text = password || n.getText() == null ? "" : n.getText().toString();
                String desc = password || n.getContentDescription() == null
                        ? ""
                        : n.getContentDescription().toString();

                if (!text.isEmpty() || !desc.isEmpty() || n.isEditable() || n.isClickable()) {
                    JSONObject item = new JSONObject();
                    if (!text.isEmpty()) item.put("text", trim(text, 160));
                    if (!desc.isEmpty()) item.put("desc", trim(desc, 160));
                    item.put("password", password);
                    item.put("class", String.valueOf(n.getClassName()));
                    item.put("clickable", n.isClickable());
                    item.put("editable", n.isEditable());
                    item.put("focused", n.isFocused());

                    Rect b = new Rect();
                    n.getBoundsInScreen(b);
                    JSONArray bounds = new JSONArray();
                    bounds.put(b.left);
                    bounds.put(b.top);
                    bounds.put(b.right);
                    bounds.put(b.bottom);
                    item.put("bounds", bounds);
                    nodes.put(item);
                }

                for (int i = 0; i < n.getChildCount(); i++) {
                    AccessibilityNodeInfo child = n.getChild(i);
                    if (child != null) q.addLast(child);
                }
            }
            out.put("nodes", nodes);
        } catch (Exception e) {
            try {
                out.put("ok", false);
                out.put("error", e.getClass().getSimpleName());
                out.put("message", e.getMessage());
            } catch (Exception ignored) {}
        }
        return out;
    }

    private String trim(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max);
    }
}
