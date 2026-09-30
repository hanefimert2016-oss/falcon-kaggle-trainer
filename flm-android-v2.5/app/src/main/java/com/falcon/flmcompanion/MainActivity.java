package com.falcon.flmcompanion;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.provider.Settings;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private final ExecutorService io = Executors.newCachedThreadPool();
    private EditText serverInput;
    private EditText codeInput;
    private EditText chatInput;
    private TextView status;
    private TextView output;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        refreshStatus();
    }

    private TextView text(String value, int size) {
        TextView v = new TextView(this);
        v.setText(value);
        v.setTextColor(Color.WHITE);
        v.setTextSize(size);
        v.setPadding(4, 8, 4, 8);
        return v;
    }

    private Button button(String label, View.OnClickListener onClick) {
        Button b = new Button(this);
        b.setText(label);
        b.setOnClickListener(onClick);
        return b;
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(28, 30, 28, 30);
        root.setBackgroundColor(Color.rgb(12, 16, 26));

        TextView title = text("FLM Android v2.5", 26);
        title.setGravity(Gravity.CENTER_HORIZONTAL);
        root.addView(title);
        TextView desc = text("Root yok · ADB yok · Erişilebilirlik izniyle yerel telefon kontrolü", 14);
        desc.setTextColor(Color.rgb(170, 190, 220));
        root.addView(desc);

        serverInput = new EditText(this);
        serverInput.setHint("http://127.0.0.1:8765");
        serverInput.setText(BridgeClient.server(this));
        serverInput.setTextColor(Color.WHITE);
        serverInput.setHintTextColor(Color.GRAY);
        root.addView(serverInput);

        codeInput = new EditText(this);
        codeInput.setHint("FLM /phone pair kodu");
        codeInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        codeInput.setTextColor(Color.WHITE);
        codeInput.setHintTextColor(Color.GRAY);
        root.addView(codeInput);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        Button pair = button("Eşle", v -> pair());
        Button access = button("Erişilebilirliği Aç", v -> startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        row.addView(pair, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        row.addView(access, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        root.addView(row);

        status = text("Durum kontrol ediliyor…", 14);
        status.setTextColor(Color.rgb(100, 220, 160));
        root.addView(status);
        root.addView(button("Durumu Yenile", v -> refreshStatus()));

        TextView chatTitle = text("FLM'ye komut gönder", 18);
        root.addView(chatTitle);
        chatInput = new EditText(this);
        chatInput.setHint("Örn: Chrome'u aç sonra chatgpt.com adresine git");
        chatInput.setTextColor(Color.WHITE);
        chatInput.setHintTextColor(Color.GRAY);
        chatInput.setMinLines(2);
        root.addView(chatInput);
        root.addView(button("Gönder", v -> sendChat()));

        output = text("", 14);
        output.setTextColor(Color.rgb(225, 235, 250));
        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        root.addView(scroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));

        setContentView(root);
    }

    private void pair() {
        final String server = BridgeClient.normalizeServer(serverInput.getText().toString());
        final String code = codeInput.getText().toString().trim();
        status.setText("Eşleniyor…");
        io.submit(() -> {
            try {
                JSONObject r = BridgeClient.pair(server, code);
                if (r.optBoolean("ok", false) && !r.optString("token", "").isEmpty()) {
                    BridgeClient.savePair(this, server, r.getString("token"));
                    runOnUiThread(() -> status.setText("Eşlendi. Şimdi Erişilebilirlik ayarından FLM Telefon Kontrolü'nü aç."));
                } else {
                    runOnUiThread(() -> status.setText("Eşleme başarısız: " + r.toString()));
                }
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Eşleme hatası: " + e.getMessage()));
            }
        });
    }

    private void refreshStatus() {
        io.submit(() -> {
            try {
                JSONObject h = BridgeClient.health(this);
                boolean enabled = FLMAccessibilityService.isEnabled(this);
                String s = "Termux bridge: " + (h.optBoolean("ok", false) ? "ulaşılabilir" : "kapalı") +
                        " · Accessibility: " + (enabled ? "açık" : "kapalı") +
                        " · Controller: " + (h.optBoolean("connected", false) ? "bağlı" : "bekliyor");
                runOnUiThread(() -> status.setText(s));
            } catch (Exception e) {
                runOnUiThread(() -> status.setText("Termux bridge bulunamadı. Önce FLM v2.5 runtime'ını çalıştır: " + e.getMessage()));
            }
        });
    }

    private void sendChat() {
        final String prompt = chatInput.getText().toString().trim();
        if (prompt.isEmpty()) return;
        output.setText("FLM düşünüyor…");
        io.submit(() -> {
            try {
                JSONObject r = BridgeClient.chat(this, prompt);
                String ans = r.optString("answer", r.toString(2));
                if (r.has("execution")) ans += "\n\n" + r.optJSONObject("execution").toString(2);
                final String finalAns = ans;
                runOnUiThread(() -> output.setText(finalAns));
            } catch (Exception e) {
                runOnUiThread(() -> output.setText("İstek hatası: " + e.getMessage()));
            }
        });
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        io.shutdownNow();
    }
}
