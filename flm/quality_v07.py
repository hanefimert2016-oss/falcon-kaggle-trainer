from __future__ import annotations

import math
import re

from flm.tooling.protocol import extract_tool_calls


def natural_text(text: str, *, min_chars: int = 20) -> bool:
    text = str(text or "").strip()
    if len(text) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in text)
    weird = sum((ord(ch) < 32 and ch not in "\n\t") or ch == "\ufffd" for ch in text)
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}", text)
    if letters / max(1, len(text)) <= 0.45 or weird:
        return False
    if len(words) < 4:
        return False
    lowered = [w.lower() for w in words]
    unique_ratio = len(set(lowered)) / max(1, len(lowered))
    if len(lowered) >= 12 and unique_ratio < 0.30:
        return False
    return True


def _finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _check_text_metrics(kind: str, model: dict, errors: list[str]) -> None:
    pre = model.get("pretrain") or {}
    sft = model.get("sft") or {}
    for stage_name, stage in (("pretrain", pre), ("sft", sft)):
        if not stage:
            errors.append(f"missing {kind} {stage_name} metrics")
            continue
        for key in ("eval_loss", "best_eval_loss"):
            if not _finite_number(stage.get(key)):
                errors.append(f"{kind} {stage_name} {key} is not finite: {stage.get(key)!r}")
        if int(stage.get("best_step", -1)) <= 0:
            errors.append(f"{kind} {stage_name} did not select a best checkpoint")
    if _finite_number(pre.get("eval_loss")) and float(pre["eval_loss"]) >= 8.5:
        errors.append(f"{kind} pretrain heldout loss too high: {pre['eval_loss']}")
    if _finite_number(sft.get("eval_loss")) and float(sft["eval_loss"]) >= 8.0:
        errors.append(f"{kind} SFT heldout loss too high: {sft['eval_loss']}")
    cfg = pre.get("config") or {}
    if int(cfg.get("seq_len", 0)) < 4096:
        errors.append(f"{kind} context is below 4096 tokens: {cfg.get('seq_len')}")
    if cfg.get("position_encoding") != "rope":
        errors.append(f"{kind} is not using RoPE: {cfg.get('position_encoding')!r}")
    if cfg.get("core_memory") is not True:
        errors.append(f"{kind} CoreMemory architecture is disabled")
    if int(cfg.get("core_memory_order",0)) < 2:
        errors.append(f"{kind} CoreMemory order is invalid: {cfg.get('core_memory_order')}")


def validate_suite(suite: dict) -> list[str]:
    errors: list[str] = []
    if suite.get("pipeline_version") != "v0.7":
        errors.append("wrong pipeline version")

    data_validation = suite.get("data_validation") or {}
    if not data_validation.get("text") or not data_validation.get("computer"):
        errors.append("missing validated v0.7 input provenance")

    models = suite.get("models") or {}
    main_model = models.get("main") or {}
    coder_model = models.get("coder") or {}
    _check_text_metrics("main", main_model, errors)
    _check_text_metrics("coder", coder_model, errors)

    quality = suite.get("quality_samples") or {}
    main = quality.get("main") or []
    coder = quality.get("coder") or []
    tool = (quality.get("tool_call") or {}).get("output", "")

    if len(main) < 6:
        errors.append("missing main knowledge quality samples")
    else:
        for i, item in enumerate(main):
            if not natural_text(item.get("output", "")):
                errors.append(
                    f"main sample {i} is not coherent-looking text: "
                    f"{item.get('output','')[:120]!r}"
                )
        capital = next(
            (x.get("output", "") for x in main if "başkenti" in x.get("prompt", "")),
            "",
        )
        if "ankara" not in capital.lower():
            errors.append(f"capital QA failed: {capital!r}")
        english = next(
            (
                x.get("output", "")
                for x in main
                if x.get("prompt", "").startswith("Hello")
            ),
            "",
        )
        if not any(
            k in english.lower()
            for k in ("system", "computer", "software", "hardware", "resource", "program")
        ):
            errors.append(f"English OS answer looks off-topic: {english!r}")

        photo = next(
            (x.get("output", "") for x in main if "Fotosentez" in x.get("prompt", "")),
            "",
        )
        if sum(k in photo.lower() for k in ("ışık", "enerji", "glikoz", "oksijen", "karbondioksit")) < 2:
            errors.append(f"photosynthesis knowledge probe failed: {photo!r}")

        ram = next(
            (x.get("output", "") for x in main if x.get("prompt", "").startswith("What is the practical difference between RAM")),
            "",
        )
        if sum(k in ram.lower() for k in ("memory", "temporary", "volatile", "storage", "persistent")) < 2:
            errors.append(f"RAM/storage knowledge probe failed: {ram!r}")

    if len(coder) < 2:
        errors.append("missing coder quality samples")
    else:
        for i, item in enumerate(coder):
            out = item.get("output", "")
            if "def " not in out or "return" not in out:
                errors.append(
                    f"coder sample {i} missing Python function structure: {out[:180]!r}"
                )

    if not coder_model.get("initialized_from_main"):
        errors.append("Coder was not initialized from the trained Main checkpoint")

    normal = (quality.get("coder_normal") or {}).get("output", "")
    if not natural_text(normal, min_chars=60):
        errors.append(f"Coder normal-answer mode is not coherent: {normal[:180]!r}")
    if any(tag in normal for tag in ("<|tool_call|>", "<|plan|>", "<|plan_end|>")):
        errors.append(f"Coder normal-answer mode leaked agent protocol: {normal[:180]!r}")

    planning = (quality.get("coder_planning") or {}).get("output", "")
    if "<|plan|>" not in planning or "<|plan_end|>" not in planning:
        errors.append(f"Coder planning mode is missing plan boundaries: {planning[:240]!r}")
    if "<|final|>" not in planning:
        errors.append(f"Coder planning mode is missing a final answer: {planning[:240]!r}")
    plan_body = planning.split("<|plan|>", 1)[-1].split("<|plan_end|>", 1)[0]
    plan_words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2,}", plan_body)
    if len(plan_words) < 30:
        errors.append(f"Coder plan is too shallow: {planning[:240]!r}")
    planning_topics = (
        ("log", "kayıt"),
        ("config", "yapılandır"),
        ("test",),
        ("repro", "yeniden üret", "tekrar üret"),
        ("rollback", "geri al"),
    )
    lower_plan = plan_body.lower()
    if sum(any(k in lower_plan for k in group) for group in planning_topics) < 3:
        errors.append(f"Coder plan misses debugging stages: {planning[:240]!r}")

    try:
        calls = extract_tool_calls(tool)
        if not calls or calls[0].name != "add":
            errors.append(f"tool call did not select add: {tool!r}")
        else:
            args = calls[0].arguments
            if int(args.get("a", -1)) != 27 or int(args.get("b", -1)) != 15:
                errors.append(f"tool arguments incorrect: {args}")
    except Exception as exc:
        errors.append(
            f"tool call is not parseable: {type(exc).__name__}: {exc}; output={tool!r}"
        )

    cu = models.get("computer_use") or {}
    if cu:
        if not cu.get("task_embedding_initialized_from_main"):
            errors.append("ComputerUse task embedding was not initialized from Main")
        cfg = cu.get("config") or {}
        if int(cu.get("text_layers_initialized_from_main", 0)) < int(cfg.get("text_layers", 0)):
            errors.append(
                "ComputerUse text tower did not fully inherit Main-compatible layers: "
                f"{cu.get('text_layers_initialized_from_main')}/{cfg.get('text_layers')}"
            )
        action_counts = cu.get("action_counts") or {}
        minimum_actions = {
            "CLICK": 20_000, "KEY": 1_000, "TYPE": 1_000, "SCROLL": 750,
            "DRAG": 300, "RIGHT_CLICK": 200, "DOUBLE_CLICK": 200, "DONE": 500,
        }
        for op, minimum in minimum_actions.items():
            if int(action_counts.get(op, 0)) < minimum:
                errors.append(f"ComputerUse {op} training coverage too small: {action_counts.get(op,0)}/{minimum}")
        if not _finite_number(cu.get("train_loss")):
            errors.append(f"CU train loss is not finite: {cu.get('train_loss')}")
        if float(cu.get("op_accuracy", 0)) < 0.50:
            errors.append(f"CU held-out op_accuracy too low: {cu.get('op_accuracy')}")
        if float(cu.get("pointer_hit_0p1", 0)) < 0.15:
            errors.append(
                f"CU held-out pointer hit too low: {cu.get('pointer_hit_0p1')}"
            )
        if not (0 <= float(cu.get("pointer_mean_l2", 99)) < 0.50):
            errors.append(
                f"CU pointer L2 invalid/high: {cu.get('pointer_mean_l2')}"
            )
        by_class = cu.get("op_accuracy_by_class") or {}
        for op in ("KEY", "TYPE", "SCROLL", "DRAG"):
            item = by_class.get(op) or {}
            if int(item.get("count", 0)) >= 3 and float(item.get("accuracy", 0)) < 0.20:
                errors.append(f"CU held-out {op} accuracy too low: {item}")
    else:
        errors.append("missing ComputerUse metrics")

    return errors
