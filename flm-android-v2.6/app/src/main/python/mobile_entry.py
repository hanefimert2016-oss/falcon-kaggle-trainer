import json
import pkgutil
import sys
import threading

from flm_tf.memory import SemanticMemory
from flm_tf.runtime import FLMRuntime
from flm_tf.semantic_ir import SemanticFact
from flm_tf.computer_use import ComputerUsePlanner

_lock = threading.RLock()
_runtime = None
_planner = ComputerUsePlanner()
_stats = {}


def _build_memory():
    raw = pkgutil.get_data("data", "core_semantics.jsonl")
    if raw is None:
        raise RuntimeError("FLM semantic data is missing from the APK")
    mem = SemanticMemory()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        mem.add(
            SemanticFact(
                x["subject"],
                x["relation"],
                x["object"],
                x.get("qualifiers", {}),
                x.get("source_text"),
            )
        )
    return mem


def initialize():
    global _runtime, _stats
    with _lock:
        if _runtime is None:
            mem = _build_memory()
            _runtime = FLMRuntime(memory=mem)
            _stats = _runtime.stats()
            _stats["python"] = sys.version.split()[0]
            _stats["embedded"] = True
            _stats["platform"] = "android"
    return json.dumps({"ok": True, **_stats}, ensure_ascii=False)


def ask_json(text):
    text = (text or "").strip()
    if not text:
        return json.dumps({"ok": False, "error": "empty_prompt"}, ensure_ascii=False)

    initialize()

    if _planner.can_plan(text):
        plan = _planner.plan(text, platform="android").to_dict()
        count = len(plan.get("actions", []))
        answer = (
            f"Telefon kontrol planı hazır: {count} adım. "
            "İstersen aşağıdaki Çalıştır düğmesiyle bu adımları onaylayabilirsin."
        )
        return json.dumps(
            {"ok": True, "kind": "computer", "answer": answer, "plan": plan},
            ensure_ascii=False,
        )

    answer = _runtime.answer(text)
    return json.dumps(
        {"ok": True, "kind": "chat", "answer": answer, "stats": _stats},
        ensure_ascii=False,
    )


def health_json():
    initialize()
    return json.dumps({"ok": True, **_stats}, ensure_ascii=False)
