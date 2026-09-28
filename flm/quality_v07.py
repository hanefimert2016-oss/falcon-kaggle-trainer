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

    if len(main) < 4:
        errors.append("missing main quality samples")
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

    if len(coder) < 2:
        errors.append("missing coder quality samples")
    else:
        for i, item in enumerate(coder):
            out = item.get("output", "")
            if "def " not in out or "return" not in out:
                errors.append(
                    f"coder sample {i} missing Python function structure: {out[:180]!r}"
                )

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
    else:
        errors.append("missing ComputerUse metrics")

    return errors
