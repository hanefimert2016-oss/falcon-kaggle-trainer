from __future__ import annotations

import json
import random
from pathlib import Path


TR_NAMES=("Ali","Ayşe","Mehmet","Zeynep","Deniz","Ece","Mert","Selin")
EN_NAMES=("Alice","Bob","Charlie","Diana","Ethan","Grace","Henry","Iris")
TYPES=(
    ("kedi","cat","Cat"),("köpek","dog","Dog"),("kuş","bird","Bird"),("memeli","mammal","Mammal"),
    ("araç","vehicle","Vehicle"),("bilgisayar","computer","Computer"),
    ("program","program","Program"),("dosya","file","File"),
)
RELATIONS=(
    ("daha uzundur","is taller than","TallerThan"),
    ("daha hızlıdır","is faster than","FasterThan"),
    ("içindedir","is in","LocatedIn"),
)


def compact(obj):
    return json.dumps(obj,ensure_ascii=False,separators=(",",":"),sort_keys=True)


def semantic_ir(facts=None,rules=None,queries=None,operations=None):
    return {
        "facts":facts or [],
        "rules":rules or [],
        "queries":queries or [],
        "operations":operations or [],
        "metadata":{"source":"synthetic-semantic-curriculum-v2"},
    }


def atom(pred,*args):
    return {"predicate":pred,"args":list(args)}


def messages_for(i:int):
    tr=(i%2==0)
    names=TR_NAMES if tr else EN_NAMES
    a=names[i%len(names)]
    b=names[(i*3+1)%len(names)]
    c=names[(i*5+2)%len(names)]
    mode=i%9

    if mode==0:
        tr_type,en_type,sym=TYPES[(i//9)%len(TYPES)]
        sentence=f"{a} bir {tr_type}dir." if tr else f"{a} is a {en_type}."
        prog=semantic_ir(facts=[atom("IsA",a,sym)])
        core_payload={"queries":[],"operation_results":[]}
        final=f"{a} bir {tr_type} olarak kaydedildi." if tr else f"{a} was recorded as a {en_type}."

    elif mode==1:
        tr_rel,en_rel,pred=RELATIONS[(i//9)%len(RELATIONS)]
        sentence=f"{a} {b}'den {tr_rel}." if tr else f"{a} {en_rel} {b}."
        prog=semantic_ir(facts=[atom(pred,a,b)])
        core_payload={"queries":[],"operation_results":[]}
        final="İlişki kaydedildi." if tr else "The relation was recorded."

    elif mode==2:
        sentence=(
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
        core_payload={"queries":[{"answer":True}],"operation_results":[]}
        final="Evet." if tr else "Yes."

    elif mode==3:
        sentence=(
            f"Bir memeli sıcakkanlıdır. {a} bir memelidir. {a} sıcakkanlı mıdır?"
            if tr else
            f"Every mammal is warm-blooded. {a} is a mammal. Is {a} warm-blooded?"
        )
        rule={
            "name":"mammal_warm_blooded",
            "premises":[atom("IsA","?x","Mammal")],
            "conclusion":atom("WarmBlooded","?x"),
        }
        prog=semantic_ir(
            facts=[atom("IsA",a,"Mammal")],
            rules=[rule],
            queries=[{"atom":atom("WarmBlooded",a)}],
        )
        core_payload={"queries":[{"answer":True}],"operation_results":[]}
        final="Evet, sıcakkanlıdır." if tr else "Yes, it is warm-blooded."

    elif mode==4:
        x=17+(i%71)
        y=3+(i%19)
        expression=f"({x}*{y})+{y}"
        value=(x*y)+y
        sentence=(
            f"{expression} işlemini hesapla. Sonucu tahmin etme; çekirdeği kullan."
            if tr else
            f"Calculate {expression}. Do not guess; use the core."
        )
        prog=semantic_ir(operations=[{
            "kind":"ARITHMETIC",
            "args":{"expression":expression},
        }])
        core_payload={"queries":[],"operation_results":[{"kind":"ARITHMETIC","ok":True,"value":value}]}
        final=f"Sonuç {value}." if tr else f"The result is {value}."

    elif mode==5:
        fn=f"compute_{i%97}"
        source=f"def {fn}(x):\n    return abs(x) + 1\n"
        sentence=(
            f"Şu Python kodunu yapısal olarak analiz et:\n{source}"
            if tr else
            f"Structurally analyze this Python code:\n{source}"
        )
        prog=semantic_ir(operations=[{
            "kind":"ANALYZE_CODE",
            "args":{"language":"python","source":source},
        }])
        core_payload={
            "queries":[],
            "operation_results":[{
                "kind":"ANALYZE_CODE","ok":True,"language":"python",
                "symbols":[{"name":fn,"kind":"function","line":1}],
                "calls":[[fn,"abs"]],"imports":[],"diagnostics":[],
            }],
        }
        final=(
            f"{fn} adlı bir fonksiyon var ve abs çağrısı yapıyor."
            if tr else
            f"There is a function named {fn}, and it calls abs."
        )

    elif mode==6:
        sentence=(
            "Başlangıçta dosya açık. Hedef dosyanın kaydedilmiş olması. Uygun adımları planla."
            if tr else
            "The file is open initially. Plan steps until the file is saved."
        )
        actions=[
            {
                "name":"save_file",
                "preconditions":["file_open"],
                "add_effects":["file_saved"],
                "delete_effects":[],
                "cost":1.0,
            }
        ]
        prog=semantic_ir(operations=[{
            "kind":"STATE_PLAN",
            "args":{"start":["file_open"],"goal":["file_saved"],"actions":actions},
        }])
        core_payload={
            "queries":[],
            "operation_results":[{
                "kind":"STATE_PLAN","ok":True,"actions":["save_file"],
                "final_state":["file_open","file_saved"],"explored":1,
            }],
        }
        final="Plan: save_file." if not tr else "Plan: save_file."

    elif mode==7:
        sentence=(
            "Editörde belgeyi kaydet. Ekranda Save düğmesi var."
            if tr else
            "Save the document in the editor. A Save button is visible."
        )
        elements=[
            {
                "id":"save_button","role":"button","name":"Save","value":"",
                "enabled":True,"visible":True,"x":0.91,"y":0.05,
            }
        ]
        prog=semantic_ir(operations=[{
            "kind":"UI_PLAN",
            "args":{"app":"editor","goal":sentence,"elements":elements},
        }])
        core_payload={
            "queries":[],
            "operation_results":[{
                "kind":"UI_PLAN","ok":True,
                "actions":[{"op":"CLICK","target_id":"save_button","payload":""}],
            }],
        }
        final="Save düğmesine tıklanmalı." if tr else "Click the Save button."

    else:
        sentence=(
            "İki kontrolü doğrula: derleme başarılı ve test sayısı 12."
            if tr else
            "Verify two checks: the build succeeded and the test count is 12."
        )
        checks=[
            {"name":"build","actual":"success","expected":"success"},
            {"name":"tests","actual":12,"expected":12},
        ]
        prog=semantic_ir(operations=[{"kind":"VERIFY","args":{"checks":checks}}])
        core_payload={
            "queries":[],
            "operation_results":[{
                "kind":"VERIFY","ok":True,
                "checks":[
                    {"name":"build","ok":True,"actual":"success","expected":"success"},
                    {"name":"tests","ok":True,"actual":12,"expected":12},
                ],
            }],
        }
        final="Tüm kontroller geçti." if tr else "All checks passed."

    ir="<|semantic_ir|>"+compact(prog)+"<|semantic_end|>"
    core="<|core_result|>"+compact(core_payload)+"<|core_end|>"
    system=(
        "You are the single FLM Interface Transformer. Convert user language into "
        "FLM Semantic IR. Do not perform hidden reasoning when FLM Core can execute it."
    )
    return [
        {"role":"system","content":system},
        {"role":"user","content":sentence},
        {"role":"assistant","content":ir},
        {"role":"tool","content":core},
        {"role":"assistant","content":"<|final|>\n"+final},
    ]


def append_semantic_curriculum(path:Path,rows:int=120_000,seed:int=7071)->dict:
    rng=random.Random(seed)
    indices=list(range(rows))
    rng.shuffle(indices)
    with path.open("a",encoding="utf-8") as fh:
        for i in indices:
            fh.write(json.dumps({
                "messages":messages_for(i),
                "source":"synthetic:flm-semantic-curriculum-v1",
            },ensure_ascii=False)+"\n")
    return {"rows":rows,"source":"synthetic:flm-semantic-curriculum-v1"}
