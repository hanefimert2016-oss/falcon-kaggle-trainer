package com.falcon.flm;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.provider.Settings;
import android.text.InputType;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowInsetsController;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int BG = Color.rgb(9, 12, 18);
    private static final int SURFACE = Color.rgb(16, 21, 31);
    private static final int SURFACE_2 = Color.rgb(24, 31, 45);
    private static final int USER_BUBBLE = Color.rgb(45, 69, 95);
    private static final int TEXT = Color.rgb(242, 246, 252);
    private static final int MUTED = Color.rgb(152, 166, 188);
    private static final int ACCENT = Color.rgb(116, 231, 211);
    private static final int ACCENT_DARK = Color.rgb(29, 78, 75);
    private static final int ERROR = Color.rgb(255, 132, 132);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private LinearLayout chat;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView localBadge;
    private ProgressBar bootProgress;

    private volatile boolean flmReady = false;
    private volatile boolean initializing = false;

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        configureWindow();
        setContentView(buildRoot());
        addWelcome();
        initializeFLM();
    }

    private void configureWindow() {
        Window w = getWindow();
        w.setStatusBarColor(BG);
        w.setNavigationBarColor(BG);
        if (android.os.Build.VERSION.SDK_INT >= 30) {
            WindowInsetsController c = w.getInsetsController();
            if (c != null) {
                c.setSystemBarsAppearance(
                        0,
                        WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS
                                | WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
                );
            }
        }
    }

    private View buildRoot() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);

        root.addView(buildHeader(),
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));

        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        scroll.setPadding(dp(12), dp(10), dp(12), dp(14));

        chat = new LinearLayout(this);
        chat.setOrientation(LinearLayout.VERTICAL);
        chat.setPadding(0, dp(8), 0, dp(12));
        scroll.addView(chat,
                new ScrollView.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));

        root.addView(scroll,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        0,
                        1f
                ));

        root.addView(buildComposer(),
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));

        return root;
    }

    private View buildHeader() {
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(18), dp(14), dp(16), dp(12));

        TextView logo = new TextView(this);
        logo.setText("F");
        logo.setTextColor(BG);
        logo.setTextSize(18);
        logo.setTypeface(Typeface.DEFAULT_BOLD);
        logo.setGravity(Gravity.CENTER);
        logo.setBackground(round(ACCENT, 18));
        header.addView(logo, new LinearLayout.LayoutParams(dp(36), dp(36)));

        LinearLayout titleBox = new LinearLayout(this);
        titleBox.setOrientation(LinearLayout.VERTICAL);
        titleBox.setPadding(dp(10), 0, 0, 0);

        TextView title = new TextView(this);
        title.setText("FLM");
        title.setTextColor(TEXT);
        title.setTextSize(21);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        titleBox.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Yerel AI · Android");
        subtitle.setTextColor(MUTED);
        subtitle.setTextSize(12);
        titleBox.addView(subtitle);

        header.addView(titleBox,
                new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        LinearLayout statusBox = new LinearLayout(this);
        statusBox.setOrientation(LinearLayout.HORIZONTAL);
        statusBox.setGravity(Gravity.CENTER_VERTICAL);
        statusBox.setPadding(dp(10), dp(6), dp(10), dp(6));
        statusBox.setBackground(round(Color.rgb(19, 42, 42), 20));

        bootProgress = new ProgressBar(this);
        bootProgress.setIndeterminate(true);
        statusBox.addView(bootProgress, new LinearLayout.LayoutParams(dp(14), dp(14)));

        localBadge = new TextView(this);
        localBadge.setText(" HAZIRLANIYOR");
        localBadge.setTextColor(ACCENT);
        localBadge.setTextSize(10);
        localBadge.setTypeface(Typeface.DEFAULT_BOLD);
        statusBox.addView(localBadge);

        header.addView(statusBox);
        return header;
    }

    private View buildComposer() {
        LinearLayout outer = new LinearLayout(this);
        outer.setOrientation(LinearLayout.VERTICAL);
        outer.setPadding(dp(12), dp(8), dp(12), dp(12));
        outer.setBackgroundColor(BG);

        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.HORIZONTAL);
        box.setGravity(Gravity.BOTTOM);
        box.setPadding(dp(14), dp(8), dp(8), dp(8));
        box.setBackground(round(SURFACE_2, 25));

        input = new EditText(this);
        input.setHint("FLM'ye bir şey sor…");
        input.setHintTextColor(Color.rgb(115, 128, 148));
        input.setTextColor(TEXT);
        input.setTextSize(16);
        input.setBackgroundColor(Color.TRANSPARENT);
        input.setPadding(0, dp(4), dp(8), dp(4));
        input.setMinLines(1);
        input.setMaxLines(5);
        input.setSingleLine(false);
        input.setInputType(
                InputType.TYPE_CLASS_TEXT
                        | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
                        | InputType.TYPE_TEXT_FLAG_MULTI_LINE
        );
        input.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_SEND);
        input.setOnEditorActionListener((v, actionId, event) -> {
            boolean sendAction =
                    actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND
                            || (event != null
                            && event.getKeyCode() == KeyEvent.KEYCODE_ENTER
                            && !event.isShiftPressed());
            if (sendAction) {
                sendPrompt();
                return true;
            }
            return false;
        });

        box.addView(input,
                new LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        1f
                ));

        send = new Button(this);
        send.setText("➤");
        send.setTextColor(BG);
        send.setTextSize(19);
        send.setTypeface(Typeface.DEFAULT_BOLD);
        send.setGravity(Gravity.CENTER);
        send.setPadding(0, 0, 0, 0);
        send.setMinWidth(0);
        send.setMinimumWidth(0);
        send.setMinHeight(0);
        send.setMinimumHeight(0);
        send.setBackground(round(ACCENT, 22));
        send.setOnClickListener(v -> sendPrompt());

        box.addView(send, new LinearLayout.LayoutParams(dp(44), dp(44)));

        outer.addView(box,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));

        return outer;
    }

    private void addWelcome() {
        addAssistant(
                "Merhaba. Ben FLM. Bu sürümde Python ve AI çekirdeği doğrudan APK'nın içinde çalışıyor. "
                        + "Termux, sunucu adresi veya eşleme kodu gerekmiyor.",
                null,
                null
        );
    }

    private void initializeFLM() {
        if (initializing || flmReady) return;
        initializing = true;
        worker.submit(() -> {
            try {
                JSONObject info = EmbeddedFLM.initialize(this);
                flmReady = info.optBoolean("ok", false);
                String python = info.optString("python", "3.13");
                int facts = info.optInt("facts", 0);
                runOnUiThread(() -> {
                    initializing = false;
                    bootProgress.setVisibility(View.GONE);
                    localBadge.setText(" YEREL · PY " + python);
                    localBadge.setTextColor(ACCENT);
                    if (facts > 0) {
                        addSystemNote(String.format("%,d semantic fact hazır", facts));
                    }
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    initializing = false;
                    flmReady = false;
                    bootProgress.setVisibility(View.GONE);
                    localBadge.setText(" BAŞLATMA HATASI");
                    localBadge.setTextColor(ERROR);
                    addAssistant(
                            "Yerel FLM çekirdeği başlatılamadı: "
                                    + e.getClass().getSimpleName() + ": " + e.getMessage(),
                            null,
                            null
                    );
                });
            }
        });
    }

    private void sendPrompt() {
        String prompt = input.getText().toString().trim();
        if (prompt.isEmpty()) return;

        input.setText("");
        addUser(prompt);
        setBusy(true);

        worker.submit(() -> {
            try {
                JSONObject response = EmbeddedFLM.ask(this, prompt);
                runOnUiThread(() -> renderResponse(response));
            } catch (Exception e) {
                runOnUiThread(() -> addAssistant(
                        "İstek işlenemedi: " + e.getClass().getSimpleName()
                                + ": " + e.getMessage(),
                        null,
                        null
                ));
            } finally {
                runOnUiThread(() -> setBusy(false));
            }
        });
    }

    private void renderResponse(JSONObject response) {
        if (!response.optBoolean("ok", false)) {
            addAssistant(
                    "FLM hatası: " + response.optString("error", "bilinmeyen hata"),
                    null,
                    null
            );
            return;
        }

        String answer = response.optString("answer", "");
        String kind = response.optString("kind", "chat");

        if ("computer".equals(kind)) {
            JSONObject plan = response.optJSONObject("plan");
            JSONArray actionArray = plan == null ? null : plan.optJSONArray("actions");
            int count = actionArray == null ? 0 : actionArray.length();

            String label = FLMAccessibilityService.isEnabled(this)
                    ? "Çalıştır · " + count + " adım"
                    : "Telefon kontrolünü aç";

            addAssistant(answer, label, () -> {
                if (!FLMAccessibilityService.isEnabled(this)) {
                    startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS));
                    addSystemNote(
                            "Android yalnız telefon kontrolü izni için bu sistem ekranını zorunlu tutuyor. "
                                    + "FLM Telefon Kontrolü'nü açıp geri dön; başka ayar gerekmiyor."
                    );
                    return;
                }
                executePlan(plan);
            });
        } else {
            addAssistant(answer, null, null);
        }
    }

    private void executePlan(JSONObject plan) {
        if (plan == null) {
            addAssistant("Telefon kontrol planı boş.", null, null);
            return;
        }

        JSONArray actions = plan.optJSONArray("actions");
        if (actions == null || actions.length() == 0) {
            addAssistant("Çalıştırılacak adım bulunamadı.", null, null);
            return;
        }

        addSystemNote("Telefon kontrolü başladı · " + actions.length() + " adım");
        worker.submit(() -> {
            int completed = 0;
            JSONObject lastFailure = null;

            for (int i = 0; i < actions.length(); i++) {
                JSONObject action = actions.optJSONObject(i);
                if (action == null) continue;

                JSONObject result = FLMAccessibilityService.executeNow(action);
                if (!result.optBoolean("ok", false)) {
                    lastFailure = result;
                    break;
                }
                completed++;
            }

            final int done = completed;
            final JSONObject failure = lastFailure;
            runOnUiThread(() -> {
                if (failure == null) {
                    addAssistant(
                            "Telefon kontrolü tamamlandı. " + done + " adım uygulandı.",
                            null,
                            null
                    );
                } else {
                    addAssistant(
                            "Telefon kontrolü " + done + " adımdan sonra durdu: "
                                    + failure.optString("message",
                                    failure.optString("error", "bilinmeyen hata")),
                            null,
                            null
                    );
                }
            });
        });
    }

    private void setBusy(boolean busy) {
        send.setEnabled(!busy);
        send.setAlpha(busy ? 0.55f : 1f);
        input.setEnabled(!busy);
    }

    private void addUser(String message) {
        addBubble(message, true, null, null);
    }

    private void addAssistant(String message, String actionLabel, Runnable action) {
        addBubble(message, false, actionLabel, action);
    }

    private void addBubble(String message, boolean user, String actionLabel, Runnable action) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(user ? Gravity.END : Gravity.START);
        row.setPadding(0, dp(5), 0, dp(5));

        LinearLayout bubble = new LinearLayout(this);
        bubble.setOrientation(LinearLayout.VERTICAL);
        bubble.setPadding(dp(14), dp(11), dp(14), dp(11));
        bubble.setBackground(round(user ? USER_BUBBLE : SURFACE, user ? 20 : 18));

        TextView text = new TextView(this);
        text.setText(message == null || message.isEmpty() ? "…" : message);
        text.setTextColor(TEXT);
        text.setTextSize(15.5f);
        text.setLineSpacing(0, 1.08f);
        text.setTextIsSelectable(true);
        bubble.addView(text);

        if (actionLabel != null && action != null) {
            Button b = new Button(this);
            b.setText(actionLabel);
            b.setTextColor(ACCENT);
            b.setTextSize(13);
            b.setTypeface(Typeface.DEFAULT_BOLD);
            b.setAllCaps(false);
            b.setGravity(Gravity.CENTER);
            b.setMinHeight(0);
            b.setMinimumHeight(0);
            b.setPadding(dp(14), dp(8), dp(14), dp(8));
            b.setBackground(round(ACCENT_DARK, 16));
            b.setOnClickListener(v -> action.run());

            LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            bp.topMargin = dp(10);
            bubble.addView(b, bp);
        }

        int maxWidth = (int) (getResources().getDisplayMetrics().widthPixels * 0.86f);
        LinearLayout.LayoutParams bubbleLp = new LinearLayout.LayoutParams(
                Math.min(maxWidth, dp(520)),
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        if (message != null && message.length() < 55) {
            bubbleLp.width = ViewGroup.LayoutParams.WRAP_CONTENT;
        }
        row.addView(bubble, bubbleLp);

        chat.addView(row,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));
        scrollToBottom();
    }

    private void addSystemNote(String message) {
        TextView note = new TextView(this);
        note.setText(message);
        note.setTextColor(MUTED);
        note.setTextSize(11.5f);
        note.setGravity(Gravity.CENTER);
        note.setPadding(dp(12), dp(8), dp(12), dp(8));
        chat.addView(note,
                new LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                ));
        scrollToBottom();
    }

    private void scrollToBottom() {
        if (scroll == null) return;
        scroll.postDelayed(() -> scroll.fullScroll(View.FOCUS_DOWN), 60);
    }

    private GradientDrawable round(int color, int radiusDp) {
        GradientDrawable g = new GradientDrawable();
        g.setColor(color);
        g.setCornerRadius(dp(radiusDp));
        return g;
    }

    private int dp(float value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }
}
