from __future__ import annotations

import difflib
import json
import random
import re
from pathlib import Path

from flm.core import FLMCore, Program


TR_NAMES=("Ali","Ayşe","Mehmet","Zeynep","Deniz","Ece","Mert","Selin")
EN_NAMES=("Alice","Bob","Charlie","Diana","Ethan","Grace","Henry","Iris")
TYPES=(
    ("kedi","cat","Cat"),
    ("köpek","dog","Dog"),
    ("kuş","bird","Bird"),
    ("memeli","mammal","Mammal"),
    ("araç","vehicle","Vehicle"),
    ("bilgisayar","computer","Computer"),
    ("program","program","Program"),
    ("dosya","file","File"),
)
RELATIONS=(
    ("daha uzundur","is taller than","TallerThan"),
    ("daha hızlıdır","is faster than","FasterThan"),
    ("içindedir","is in","LocatedIn"),
)

SEMANTIC_MODES=16

TR_IDENTITY_ANSWERS=(
    "Ben FLM'im. Dili tek bir arayüz modeliyle işler, asıl muhakemeyi Semantic Core üzerinden yürütürüm.",
    "Adım FLM. İstekleri anlayan bir arayüz Transformer'ım var; planlama, doğrulama ve mantık ise çekirdekte çalışır.",
    "Ben FLM adlı bir yapay zekâ asistanıyım. Dil üretimini arayüz modeli, doğrulanabilir muhakemeyi ise Semantic Core üstlenir.",
    "FLM'im. Amacım isteğini anlamak, çekirdekte gerekli işlemleri yürütmek ve sonucu doğal bir dille sana aktarmak.",
    "Bana FLM diyebilirsin. Tek bir eğitilebilir dil arayüzü ile eğitim gerektirmeyen Semantic Core'u birlikte kullanırım.",
    "Ben FLM; dil arayüzü, semantik hafıza, planlayıcı ve doğrulayıcı çekirdeği birlikte kullanan bir yapay zekâ asistanıyım.",
)
EN_IDENTITY_ANSWERS=(
    "I'm FLM. One trainable interface model handles language while the Semantic Core performs the structured reasoning.",
    "My name is FLM. I use a single language interface Transformer, with planning and verification handled by my Semantic Core.",
    "I'm an AI assistant called FLM. Language is handled by one interface model and structured reasoning lives in the Core.",
    "I'm FLM. I interpret requests through one language model, then use a Semantic Core for reasoning, tools, and verification.",
    "You can call me FLM. I combine one trainable language interface with a training-free Semantic Core.",
    "I'm FLM, an AI assistant built around a language interface, semantic memory, planning, and verification.",
)


def _normalized_words(text:str)->str:
    text=re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü]+"," ",str(text or "").casefold())
    return " ".join(text.split())


def copy_similarity(source:str,answer:str)->float:
    a=_normalized_words(source)
    b=_normalized_words(answer)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None,a,b).ratio()


def _assert_not_copy(user:str,final:str)->None:
    # Tiny answers such as "Evet." are semantic labels rather than copied prose.
    if len(_normalized_words(final))<18:
        return
    score=copy_similarity(user,final)
    if score>=0.88:
        raise RuntimeError(
            f"semantic curriculum answer copies input too closely score={score:.3f} "
            f"user={user!r} final={final!r}"
        )


def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"),sort_keys=True)


def atom(pred,*args):
    return {"predicate":pred,"args":list(args)}


def semantic_ir(*,facts=None,rules=None,queries=None,operations=None,source="synthetic-semantic-curriculum-v2"):
    return {
        "facts":facts or [],
        "rules":rules or [],
        "queries":queries or [],
        "operations":operations or [],
        "metadata":{"source":source},
    }


def operation(kind,**args):
    return {"kind":kind,"args":args}


def _core_payload(prog:dict)->dict:
    core=FLMCore()
    result=core.execute(Program.from_dict(prog))
    return {
        "results":[
            {
                "answer":x.answer,
                "query":x.query.to_dict(),
                "proof":x.proof,
                "rounds":x.rounds,
                "derived_facts":x.derived_facts,
            }
            for x in result.results
        ],
        "operation_results":result.operation_results,
        "memory_facts":result.memory_facts,
        "memory_rules":result.memory_rules,
    }


def _relation_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    b=names[(i*3+1)%len(names)]
    tr_rel,en_rel,pred=RELATIONS[(i//3)%len(RELATIONS)]
    user=f"{a} {b}'den {tr_rel}." if tr else f"{a} {en_rel} {b}."
    prog=semantic_ir(facts=[atom(pred,a,b)])
    final="İlişki kaydedildi." if tr else "The relation was recorded."
    return user,prog,final


def _transitive_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    b=names[(i*3+1)%len(names)]
    c=names[(i*5+2)%len(names)]
    user=(
        f"{a}, {b}'den uzundur. {b}, {c}'den uzundur. {a}, {c}'den uzun mudur?"
        if tr else
        f"{a} is taller than {b}. {b} is taller than {c}. Is {a} taller than {c}?"
    )
    rule={
        "name":"taller_transitive",
        "premises":[atom("TallerThan","?x","?y"),atom("TallerThan","?y","?z")],
        "conclusion":atom("TallerThan","?x","?z"),
    }
    prog=semantic_ir(
        facts=[atom("TallerThan",a,b),atom("TallerThan",b,c)],
        rules=[rule],
        queries=[{"atom":atom("TallerThan",a,c)}],
    )
    return user,prog,("Evet." if tr else "Yes.")


def _rule_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    user=(
        f"Her memeli sıcakkanlıdır. {a} bir memelidir. {a} sıcakkanlı mıdır?"
        if tr else
        f"Every mammal is warm-blooded. {a} is a mammal. Is {a} warm-blooded?"
    )
    prog=semantic_ir(
        facts=[atom("IsA",a,"Mammal")],
        rules=[{
            "name":"mammal_warm_blooded",
            "premises":[atom("IsA","?x","Mammal")],
            "conclusion":atom("WarmBlooded","?x"),
        }],
        queries=[{"atom":atom("WarmBlooded",a)}],
    )
    return user,prog,("Evet, sıcakkanlıdır." if tr else "Yes, it is warm-blooded.")


def _arithmetic_case(i:int,tr:bool):
    a=17+(i%83)
    b=3+((i*7)%29)
    mul=2+(i%5)
    expr=f"({a}+{b})*{mul}"
    value=(a+b)*mul
    user=(
        f"({a}+{b})×{mul} işlemini hesapla."
        if tr else
        f"Calculate ({a}+{b})*{mul}."
    )
    prog=semantic_ir(operations=[operation("ARITHMETIC",expression=expr)])
    final=(f"Sonuç {value}." if tr else f"The result is {value}.")
    return user,prog,final


def _code_case(i:int,tr:bool):
    name=f"helper_{i%97}"
    source=(
        "import os\n"
        f"def {name}(x):\n"
        "    return print(x)\n"
    )
    user=(
        f"Bu Python kodunu yapısal olarak analiz et:\n\n{source}"
        if tr else
        f"Structurally analyze this Python code:\n\n{source}"
    )
    prog=semantic_ir(operations=[operation("ANALYZE_CODE",language="python",source=source)])
    final=(
        f"Kodda {name} adlı bir fonksiyon var; os içe aktarılıyor ve fonksiyon print çağırıyor."
        if tr else
        f"The code defines {name}, imports os, and the function calls print."
    )
    return user,prog,final


def _syntax_error_case(i:int,tr:bool):
    source=f"def broken_{i%71}(x)\n    return x+1\n"
    user=(
        f"Bu Python kodundaki sözdizimi sorununu analiz et:\n{source}"
        if tr else
        f"Analyze the syntax problem in this Python code:\n{source}"
    )
    prog=semantic_ir(operations=[operation("ANALYZE_CODE",language="python",source=source)])
    final=(
        "Kodda fonksiyon tanımından sonra iki nokta eksik; analiz SyntaxError döndürüyor."
        if tr else
        "The function definition is missing a colon; the analysis reports a SyntaxError."
    )
    return user,prog,final


def _state_plan_case(i:int,tr:bool):
    user=(
        "Belge kapalı. Önce açıp sonra kaydedilmiş duruma getir."
        if tr else
        "The document is closed. Plan how to open it and reach a saved state."
    )
    prog=semantic_ir(operations=[operation(
        "STATE_PLAN",
        start=["closed"],
        goal=["saved"],
        actions=[
            {
                "name":"open",
                "preconditions":["closed"],
                "add_effects":["open"],
                "delete_effects":["closed"],
            },
            {
                "name":"save",
                "preconditions":["open"],
                "add_effects":["saved"],
            },
        ],
    )])
    final=(
        "Plan: önce belgeyi aç, ardından kaydet."
        if tr else
        "Plan: open the document first, then save it."
    )
    return user,prog,final


def _ui_save_case(i:int,tr:bool):
    goal="Dosyayı kaydet" if tr else "Save the file"
    prog=semantic_ir(operations=[operation(
        "UI_PLAN",
        goal=goal,
        app="editor",
        elements=[
            {"id":"save","role":"button","name":"Kaydet" if tr else "Save","x":0.91,"y":0.08},
            {"id":"editor","role":"editor","name":"main.py","x":0.50,"y":0.50},
        ],
    )])
    final=(
        "Kaydet düğmesine tıklanmalı."
        if tr else
        "The Save button should be clicked."
    )
    return goal,prog,final


def _ui_type_case(i:int,tr:bool):
    text=("merhaba dünya" if tr else "hello world")+f" {i%23}"
    goal=(f'Kutucuğa "{text}" yaz.' if tr else f'Type "{text}" into the textbox.')
    prog=semantic_ir(operations=[operation(
        "UI_PLAN",
        goal=goal,
        app="browser",
        elements=[
            {"id":"query","role":"textbox","name":"Arama" if tr else "Search","x":0.45,"y":0.15},
        ],
    )])
    final=(
        f"Metin kutusuna tıklayıp “{text}” yazılmalı."
        if tr else
        f"Click the textbox and type “{text}”."
    )
    return goal,prog,final


def _verify_case(i:int,tr:bool):
    expected=i%7
    actual=expected if i%3 else expected+1
    user=(
        f"Kontrol et: beklenen değer {expected}, gerçek değer {actual}."
        if tr else
        f"Verify this check: expected {expected}, actual {actual}."
    )
    prog=semantic_ir(operations=[operation(
        "VERIFY",
        checks=[{"name":"value","expected":expected,"actual":actual}],
    )])
    ok=actual==expected
    final=(
        ("Kontrol geçti." if ok else "Kontrol başarısız oldu.")
        if tr else
        ("The check passed." if ok else "The check failed.")
    )
    return user,prog,final


def _negative_query_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    user=(
        f"{a} bir memelidir. {a} bir bilgisayar mıdır?"
        if tr else
        f"{a} is a mammal. Is {a} a computer?"
    )
    prog=semantic_ir(
        facts=[atom("IsA",a,"Mammal")],
        queries=[{"atom":atom("IsA",a,"Computer")}],
    )
    return user,prog,("Hayır." if tr else "No.")


def _long_chain_case(i:int,tr:bool,names):
    a,b,c,d=(names[(i+j)%len(names)] for j in range(4))
    user=(
        f"{a}, {b}'den; {b}, {c}'den; {c}, {d}'den uzundur. {a}, {d}'den uzun mudur?"
        if tr else
        f"{a} is taller than {b}; {b} than {c}; {c} than {d}. Is {a} taller than {d}?"
    )
    rule={
        "name":"taller_transitive",
        "premises":[atom("TallerThan","?x","?y"),atom("TallerThan","?y","?z")],
        "conclusion":atom("TallerThan","?x","?z"),
    }
    prog=semantic_ir(
        facts=[
            atom("TallerThan",a,b),
            atom("TallerThan",b,c),
            atom("TallerThan",c,d),
        ],
        rules=[rule],
        queries=[{"atom":atom("TallerThan",a,d)}],
    )
    return user,prog,("Evet; ilişki üç adımda çıkarılabilir." if tr else "Yes; it follows through three relation steps.")


def _identity_case(i:int,tr:bool):
    user=(
        ("Sen kimsin?" if i%4==0 else "Kendini tanıt.")
        if tr else
        ("Who are you?" if i%4==1 else "Introduce yourself.")
    )
    prog=semantic_ir(operations=[operation("IDENTITY")],source="synthetic-semantic-curriculum-v3")
    answers=TR_IDENTITY_ANSWERS if tr else EN_IDENTITY_ANSWERS
    final=answers[(i//SEMANTIC_MODES)%len(answers)]
    return user,prog,final


def _fact_synthesis_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    if tr:
        user=(
            f"Kaynak bilgi: {a} bir kedidir. {a} bir memelidir. "
            f"Bu bilgileri aynen kopyalamadan {a} hakkında doğal bir cevap üret."
        )
        facts=[
            {"subject":a,"predicate":"type","object":"kedi"},
            {"subject":a,"predicate":"class","object":"memeli"},
        ]
        variants=(
            f"{a}, kedi türünde bir memelidir.",
            f"{a} hem bir kedidir hem de memeliler sınıfına aittir.",
            f"{a}'yı bir kedi ve dolayısıyla bir memeli olarak tanımlayabiliriz.",
            f"{a}, memeliler grubundaki bir kedidir.",
        )
    else:
        user=(
            f"Source facts: {a} is a cat. {a} is a mammal. "
            f"Answer naturally about {a} without copying those sentences."
        )
        facts=[
            {"subject":a,"predicate":"type","object":"cat"},
            {"subject":a,"predicate":"class","object":"mammal"},
        ]
        variants=(
            f"{a} is a cat that belongs to the mammal class.",
            f"{a} can be described as both a cat and a mammal.",
            f"{a} belongs to the mammals and is specifically a cat.",
            f"In short, {a} is a mammalian cat.",
        )
    prog=semantic_ir(
        operations=[operation("SYNTHESIZE_FACTS",facts=facts)],
        source="synthetic-semantic-curriculum-v3",
    )
    return user,prog,variants[(i//SEMANTIC_MODES)%len(variants)]


def _answer_style_case(i:int,tr:bool):
    value=40+(i%53)
    user=(
        f"Çekirdek sonucu {value}. Cevabı veri satırını tekrar etmeden doğal Türkçe ile söyle."
        if tr else
        f"The Core result is {value}. Answer naturally without repeating the data row verbatim."
    )
    prog=semantic_ir(
        operations=[operation("ARITHMETIC",expression=str(value))],
        source="synthetic-semantic-curriculum-v3",
    )
    if tr:
        variants=(
            f"Hesabın sonucu {value}.",
            f"Sonuç olarak {value} elde ediliyor.",
            f"Bu işlem {value} değerini veriyor.",
            f"Doğrulanan sonuç {value}.",
        )
    else:
        variants=(
            f"The calculation comes out to {value}.",
            f"The verified result is {value}.",
            f"This evaluates to {value}.",
            f"The result of the calculation is {value}.",
        )
    return user,prog,variants[(i//SEMANTIC_MODES)%len(variants)]


def _code_explain_case(i:int,tr:bool):
    name=f"double_{i%71}"
    source=f"def {name}(x):\n    return x * 2\n"
    user=(
        f"Bu kodun ne yaptığını kodu aynen tekrar etmeden açıkla:\n\n```python\n{source}```"
        if tr else
        f"Explain what this code does without repeating it verbatim:\n\n```python\n{source}```"
    )
    prog=semantic_ir(
        operations=[operation("ANALYZE_CODE",language="python",source=source)],
        source="synthetic-semantic-curriculum-v3",
    )
    variants=(
        (
            f"{name} adlı fonksiyon, aldığı sayının iki katını döndürüyor.",
            f"Bu fonksiyon girdiyi 2 ile çarpıp sonucu geri veriyor; adı {name}.",
            f"{name}, tek bir değer alıp onu iki katına çıkaran basit bir fonksiyon.",
        )
        if tr else
        (
            f"The function {name} takes one value and returns twice that value.",
            f"{name} simply doubles its input and returns the result.",
            f"This defines {name}, a small function that multiplies the input by two.",
        )
    )
    return user,prog,variants[(i//SEMANTIC_MODES)%len(variants)]


def messages_for(i:int):
    tr=(i%2==0)
    names=TR_NAMES if tr else EN_NAMES
    mode=i%SEMANTIC_MODES
    if mode==0:
        a=names[i%len(names)]
        tr_type,en_type,sym=TYPES[(i//4)%len(TYPES)]
        user=f"{a} bir {tr_type}dir." if tr else f"{a} is a {en_type.lower()}."
        prog=semantic_ir(facts=[atom("IsA",a,sym)])
        final=f"{a} bir {tr_type} olarak kaydedildi." if tr else f"{a} was recorded as a {en_type.lower()}."
    elif mode==1:
        user,prog,final=_relation_case(i,tr,names)
    elif mode==2:
        user,prog,final=_transitive_case(i,tr,names)
    elif mode==3:
        user,prog,final=_rule_case(i,tr,names)
    elif mode==4:
        user,prog,final=_arithmetic_case(i,tr)
    elif mode==5:
        user,prog,final=_code_case(i,tr)
    elif mode==6:
        user,prog,final=_syntax_error_case(i,tr)
    elif mode==7:
        user,prog,final=_state_plan_case(i,tr)
    elif mode==8:
        user,prog,final=_ui_save_case(i,tr)
    elif mode==9:
        user,prog,final=_ui_type_case(i,tr)
    elif mode==10:
        user,prog,final=_verify_case(i,tr)
    elif mode==11:
        user,prog,final=(
            _negative_query_case(i,tr,names)
            if (i//SEMANTIC_MODES)%2==0
            else _long_chain_case(i,tr,names)
        )
    elif mode==12:
        user,prog,final=_identity_case(i,tr)
    elif mode==13:
        user,prog,final=_fact_synthesis_case(i,tr,names)
    elif mode==14:
        user,prog,final=_answer_style_case(i,tr)
    else:
        user,prog,final=_code_explain_case(i,tr)

    _assert_not_copy(user,final)
    payload=_core_payload(prog)
    ir="<|semantic_ir|>"+compact(prog)+"<|semantic_end|>"
    core="<|core_result|>"+compact(payload)+"<|core_end|>"
    system=(
        "You are the single FLM Interface Transformer. Convert language/code/UI requests "
        "into FLM Semantic IR. Never perform hidden reasoning that FLM Core can execute. "
        "After receiving a verified core result, use it as semantic meaning rather than text "
        "to copy. Produce a fresh natural answer in the user's language, preserve the Core "
        "facts exactly, avoid mirroring the source wording, and vary phrasing across equivalent "
        "examples without changing the conclusion."
    )
    return [
        {"role":"system","content":system},
        {"role":"user","content":user},
        {"role":"assistant","content":ir},
        {"role":"tool","content":core},
        {"role":"assistant","content":"<|final|>\n"+final},
    ]


def append_semantic_curriculum(path:Path,rows:int=600_000,seed:int=7071)->dict:
    rng=random.Random(seed)
    indices=list(range(rows))
    rng.shuffle(indices)
    mode_counts={str(i):0 for i in range(SEMANTIC_MODES)}
    with path.open("a",encoding="utf-8") as fh:
        for i in indices:
            messages=messages_for(i)
            fh.write(json.dumps({
                "messages":messages,
                "source":"synthetic:flm-semantic-curriculum-v3",
            },ensure_ascii=False)+"\n")
            mode_counts[str(i%SEMANTIC_MODES)]+=1
    return {
        "rows":rows,
        "modes":SEMANTIC_MODES,
        "mode_counts":mode_counts,
        "source":"synthetic:flm-semantic-curriculum-v3",
        "core_executed":True,
        "anti_copy_verified":True,
        "identity_variants_tr":len(TR_IDENTITY_ANSWERS),
        "identity_variants_en":len(EN_IDENTITY_ANSWERS),
    }
