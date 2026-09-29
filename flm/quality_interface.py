from __future__ import annotations

from pathlib import Path

from flm.core import FLMCore, Operation, Program
from flm.interface_agent import (
    InterfaceAgent,
    _candidate_preserves_core,
    _has_long_verbatim_overlap,
)


_PROTOCOL_MARKERS=(
    "<|semantic_ir|>","<|semantic_end|>",
    "<|core_result|>","<|core_end|>",
)


def _clean(answer:str)->bool:
    return bool(answer.strip()) and not any(x in answer for x in _PROTOCOL_MARKERS)


def evaluate_interface_behavior(
    checkpoint:str|Path,
    *,
    tokenizer:str|Path|None=None,
    device:str|None=None,
)->dict:
    """Behavior gate for the trained single InterfaceTransformer.

    The Core supplies fixed verified meaning. This test checks that the learned
    renderer can express that meaning naturally instead of replaying protocol or
    source text. Deterministic fallback remains available at runtime, but it does
    not count as learned rendering success here.
    """
    agent=InterfaceAgent(
        checkpoint,tokenizer,
        device=device,
        render_temperature=0.72,
        render_top_k=50,
        render_top_p=0.92,
        repetition_penalty=1.08,
        max_new=512,
    )
    core=FLMCore()
    cases=[]
    failures=[]

    identity_prog=Program(operations=[Operation("IDENTITY",{})])
    identity=[]
    identity_neural=0
    for _ in range(4):
        result=core.execute(identity_prog)
        answer=agent.render_verified("Sen kimsin?",identity_prog,result)
        source=agent.last_render_source
        identity_neural+=int(source=="neural")
        identity.append(answer)
        cases.append({
            "case":"identity","answer":answer,"source":source,
            "attempts":agent.last_render_attempts,
        })
    if not all(_clean(x) and "flm" in x.casefold() for x in identity):
        failures.append("identity_not_faithful")
    if len(set(identity))<3:
        failures.append("identity_not_diverse")
    if identity_neural<2:
        failures.append("identity_neural_renderer_too_weak")

    synth_prompt=(
        "Kaynak bilgi: Ada bir kedidir. Ada bir memelidir. "
        "Bilgiyi aynen kopyalamadan, kendi cümlelerinle Ada hakkında cevap ver."
    )
    synth_prog=Program(operations=[Operation("SYNTHESIZE_FACTS",{
        "facts":[
            {"subject":"Ada","predicate":"type","object":"kedi"},
            {"subject":"Ada","predicate":"class","object":"memeli"},
        ]
    })])
    synth_result=core.execute(synth_prog)
    synth_answer=agent.render_verified(synth_prompt,synth_prog,synth_result)
    cases.append({
        "case":"fact_synthesis","answer":synth_answer,
        "source":agent.last_render_source,"attempts":agent.last_render_attempts,
    })
    if not _candidate_preserves_core(
        synth_answer,synth_prompt,synth_prog,synth_result,verbatim=False
    ):
        failures.append("fact_synthesis_core_drift")
    if _has_long_verbatim_overlap(synth_prompt,synth_answer,9):
        failures.append("fact_synthesis_verbatim_copy")
    if agent.last_render_source!="neural":
        failures.append("fact_synthesis_used_fallback")

    math_prompt=(
        "Çekirdek 27 ile 14'ün çarpımını hesapladı. "
        "Doğrulanmış sonucu doğal bir Türkçe cümleyle açıkla."
    )
    math_prog=Program(operations=[Operation("ARITHMETIC",{"expression":"27*14"})])
    math_result=core.execute(math_prog)
    math_answer=agent.render_verified(math_prompt,math_prog,math_result)
    cases.append({
        "case":"arithmetic_render","answer":math_answer,
        "source":agent.last_render_source,"attempts":agent.last_render_attempts,
    })
    if "378" not in math_answer or not _clean(math_answer):
        failures.append("arithmetic_render_wrong")
    if agent.last_render_source!="neural":
        failures.append("arithmetic_render_used_fallback")

    code_source="def iki_kat(x):\n    return x * 2\n"
    code_prompt=(
        "Aşağıdaki Python kodunun ne yaptığını kodu aynen tekrar etmeden açıkla:\n"
        +code_source
    )
    code_prog=Program(operations=[Operation("ANALYZE_CODE",{
        "language":"python","source":code_source,
    })])
    code_result=core.execute(code_prog)
    code_answer=agent.render_verified(code_prompt,code_prog,code_result)
    cases.append({
        "case":"code_explanation","answer":code_answer,
        "source":agent.last_render_source,"attempts":agent.last_render_attempts,
    })
    if "iki_kat" not in code_answer or not _clean(code_answer):
        failures.append("code_explanation_missing_symbol")
    if _has_long_verbatim_overlap(code_prompt,code_answer,9):
        failures.append("code_explanation_verbatim_copy")
    if agent.last_render_source!="neural":
        failures.append("code_explanation_used_fallback")

    neural=sum(1 for x in cases if x["source"]=="neural")
    return {
        "ok":not failures,
        "failures":failures,
        "cases":cases,
        "identity_unique":len(set(identity)),
        "identity_neural":identity_neural,
        "neural_cases":neural,
        "total_cases":len(cases),
    }
