package com.falcon.flm;

import android.content.Context;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.json.JSONObject;

public final class EmbeddedFLM {
    private EmbeddedFLM() {}

    public static synchronized void ensureStarted(Context context) {
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(context.getApplicationContext()));
        }
    }

    public static JSONObject initialize(Context context) throws Exception {
        ensureStarted(context);
        PyObject module = Python.getInstance().getModule("mobile_entry");
        String raw = module.callAttr("initialize").toString();
        return new JSONObject(raw);
    }

    public static JSONObject ask(Context context, String text) throws Exception {
        ensureStarted(context);
        PyObject module = Python.getInstance().getModule("mobile_entry");
        String raw = module.callAttr("ask_json", text).toString();
        return new JSONObject(raw);
    }

    public static JSONObject health(Context context) throws Exception {
        ensureStarted(context);
        PyObject module = Python.getInstance().getModule("mobile_entry");
        return new JSONObject(module.callAttr("health_json").toString());
    }
}
