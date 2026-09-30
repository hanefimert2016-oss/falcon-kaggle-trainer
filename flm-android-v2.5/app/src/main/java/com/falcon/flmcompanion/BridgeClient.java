package com.falcon.flmcompanion;

import android.content.Context;
import android.content.SharedPreferences;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public final class BridgeClient {
    public static final String PREFS = "flm_companion";
    public static final String KEY_SERVER = "server";
    public static final String KEY_TOKEN = "token";
    public static final String DEFAULT_SERVER = "http://127.0.0.1:8765";
    public static final String CLIENT_NAME = "FLM Android Companion";
    public static final String CLIENT_VERSION = "2.5";

    private BridgeClient() {}

    public static SharedPreferences prefs(Context c) {
        return c.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    public static String server(Context c) {
        return prefs(c).getString(KEY_SERVER, DEFAULT_SERVER);
    }

    public static String token(Context c) {
        return prefs(c).getString(KEY_TOKEN, "");
    }

    public static void savePair(Context c, String server, String token) {
        prefs(c).edit().putString(KEY_SERVER, normalizeServer(server)).putString(KEY_TOKEN, token).apply();
    }

    public static String normalizeServer(String s) {
        String x = (s == null || s.trim().isEmpty()) ? DEFAULT_SERVER : s.trim();
        while (x.endsWith("/")) x = x.substring(0, x.length() - 1);
        return x;
    }

    public static JSONObject pair(String server, String code) throws Exception {
        JSONObject body = new JSONObject();
        body.put("code", code == null ? "" : code.trim());
        body.put("client_name", CLIENT_NAME);
        body.put("client_version", CLIENT_VERSION);
        return request("POST", normalizeServer(server) + "/pair", "", body, 8000);
    }

    public static JSONObject chat(Context c, String prompt) throws Exception {
        JSONObject body = new JSONObject();
        body.put("prompt", prompt);
        return request("POST", server(c) + "/chat", token(c), body, 60000);
    }

    public static HttpResult next(Context c) throws Exception {
        return requestRaw("GET", server(c) + "/next", token(c), null, 25000);
    }

    public static JSONObject postResult(Context c, JSONObject body) throws Exception {
        return request("POST", server(c) + "/result", token(c), body, 12000);
    }

    public static JSONObject health(Context c) throws Exception {
        return request("GET", server(c) + "/health", "", null, 5000);
    }

    private static JSONObject request(String method, String url, String token, JSONObject body, int timeout) throws Exception {
        HttpResult r = requestRaw(method, url, token, body, timeout);
        if (r.code < 200 || r.code >= 300) {
            JSONObject err = new JSONObject();
            err.put("ok", false);
            err.put("http", r.code);
            err.put("body", r.body);
            return err;
        }
        if (r.body == null || r.body.trim().isEmpty()) return new JSONObject();
        return new JSONObject(r.body);
    }

    private static HttpResult requestRaw(String method, String urlText, String token, JSONObject body, int timeout) throws Exception {
        URL url = new URL(urlText);
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setRequestMethod(method);
        c.setConnectTimeout(Math.min(timeout, 6000));
        c.setReadTimeout(timeout);
        c.setUseCaches(false);
        c.setRequestProperty("Accept", "application/json");
        c.setRequestProperty("X-FLM-Client", CLIENT_NAME);
        c.setRequestProperty("X-FLM-Version", CLIENT_VERSION);
        if (token != null && !token.isEmpty()) c.setRequestProperty("Authorization", "Bearer " + token);
        if (body != null) {
            byte[] data = body.toString().getBytes(StandardCharsets.UTF_8);
            c.setDoOutput(true);
            c.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            c.setFixedLengthStreamingMode(data.length);
            try (OutputStream out = c.getOutputStream()) { out.write(data); }
        }
        int code = c.getResponseCode();
        InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
        String text = "";
        if (in != null) {
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
                String line;
                while ((line = br.readLine()) != null) sb.append(line);
            }
            text = sb.toString();
        }
        c.disconnect();
        return new HttpResult(code, text);
    }

    public static final class HttpResult {
        public final int code;
        public final String body;
        public HttpResult(int code, String body) { this.code = code; this.body = body == null ? "" : body; }
    }
}
