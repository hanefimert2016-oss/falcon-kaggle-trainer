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

SEMANTIC_MODES=20

TR_IDENTITY_ANSWERS=(
    "Ben FLM'im. Dili tek bir arayüz modeliyle işler, asıl muhakemeyi Semantic Core üzerinden yürütürüm.",
    "Adım FLM. İstekleri anlayan bir arayüz Transformer'ım var; planlama, doğrulama ve mantık ise çekirdekte çalışır.",
    "Ben FLM adlı bir yapay zekâ asistanıyım. Dil üretimini arayüz modeli, doğrulanabilir muhakemeyi ise Semantic Core üstlenir.",
    "FLM'im. Amacım isteğini anlamak, çekirdekte gerekli işlemleri yürütmek ve sonucu doğal bir dille sana aktarmak.",
    "Bana FLM diyebilirsin. Tek bir eğitilebilir dil arayüzü ile eğitim gerektirmeyen Semantic Core'u birlikte kullanırım.",
    "Ben FLM; dil arayüzü, semantik hafıza, planlayıcı ve doğrulayıcı çekirdeği birlikte kullanan bir yapay zekâ asistanıyım.",
    "Kısaca FLM'im. Dili anlamak için tek bir arayüz modeli kullanırım; sonuçları ise çekirdekteki mantık ve araçlarla doğrularım.",
    "Benim adım FLM. Sorunu dile çevirmek yerine önce anlam yapısına dönüştürür, ardından Semantic Core ile işlerim.",
    "FLM olarak çalışıyorum: arayüz modeli ne istediğini çözer, çekirdeğim hesaplama ve doğrulamayı yürütür.",
    "Ben FLM adlı bir AI asistanıyım. Cevaplarımı sadece ezberden değil, Semantic Core'un doğruladığı sonuçlardan oluştururum.",
)
EN_IDENTITY_ANSWERS=(
    "I'm FLM. One trainable interface model handles language while the Semantic Core performs the structured reasoning.",
    "My name is FLM. I use a single language interface Transformer, with planning and verification handled by my Semantic Core.",
    "I'm an AI assistant called FLM. Language is handled by one interface model and structured reasoning lives in the Core.",
    "I'm FLM. I interpret requests through one language model, then use a Semantic Core for reasoning, tools, and verification.",
    "You can call me FLM. I combine one trainable language interface with a training-free Semantic Core.",
    "I'm FLM, an AI assistant built around a language interface, semantic memory, planning, and verification.",
    "In short, I'm FLM. One interface model understands language, while the Core handles structured reasoning and checks.",
    "I'm called FLM. I turn requests into semantic structure and let the Core perform the calculations and verification.",
    "I'm FLM: the interface understands what you mean, while my Semantic Core carries out the structured work.",
    "I'm an AI assistant named FLM. I form answers from Core-verified results rather than simply replaying stored text.",
)


TR_IDENTITY_PARTS={
    "open":(
        "Ben FLM'im.",
        "Adım FLM.",
        "Bana FLM diyebilirsin.",
        "FLM adlı bir yapay zekâ asistanıyım.",
        "Kısaca, ben FLM'im.",
        "Benim adım FLM.",
    ),
    "architecture":(
        "Dili tek bir arayüz modeliyle işler, doğrulanabilir muhakemeyi Semantic Core'da yürütürüm.",
        "Bir arayüz Transformer'ı dili yorumlar; planlama, mantık ve doğrulama çekirdekte çalışır.",
        "Dil katmanım isteği semantik yapıya çevirir, asıl işlemleri eğitim gerektirmeyen çekirdeğim yapar.",
        "Tek eğitilebilir dil arayüzünü semantik hafıza, planlayıcı ve doğrulayıcı çekirdekle birleştiririm.",
        "Metni anlamak için arayüz modelini, sonuçları hesaplamak ve kontrol etmek için Semantic Core'u kullanırım.",
        "İstekleri Semantic IR'ye dönüştürüp mantık, araç ve doğrulamayı çekirdekte yürütürüm.",
    ),
    "purpose":(
        "Amacım doğru sonucu üretip bunu doğal bir dille anlatmak.",
        "Cevabı ezberden basmak yerine doğrulanmış anlamdan yeniden oluştururum.",
        "Aynı gerçeği korurken cümleleri bağlama uygun biçimde yeniden kurabilirim.",
        "Bilgiyi kaynak cümleyi kopyalamadan kullanıp yararlı bir cevap haline getiririm.",
        "Hedefim anlamı koruyarak doğal, tutarlı ve doğrulanabilir cevaplar vermek.",
        "Sonuçları çekirdekten alır, kullanıcıya uygun yeni bir anlatımla sunarım.",
    ),
}
EN_IDENTITY_PARTS={
    "open":(
        "I'm FLM.",
        "My name is FLM.",
        "You can call me FLM.",
        "I'm an AI assistant named FLM.",
        "In short, I'm FLM.",
        "I go by FLM.",
    ),
    "architecture":(
        "One interface model handles language while my Semantic Core performs verifiable reasoning.",
        "A single interface Transformer interprets language, while planning, logic, and verification run in the Core.",
        "My language layer converts requests into semantic structure and the training-free Core performs the actual operations.",
        "I combine one trainable language interface with semantic memory, planning, and verification in the Core.",
        "The interface handles text, while the Semantic Core calculates, plans, and checks results.",
        "I turn requests into Semantic IR and let the Core handle logic, tools, and verification.",
    ),
    "purpose":(
        "My goal is to produce correct results and explain them naturally.",
        "I build answers from verified meaning instead of replaying memorized source text.",
        "I can vary the wording while keeping the underlying conclusion unchanged.",
        "I use information without simply copying the sentence it came from.",
        "The aim is to keep answers natural, consistent, and grounded in verified meaning.",
        "I take Core results and express them in a fresh form suited to the user.",
    ),
}


def identity_surface(i:int,tr:bool)->str:
    parts=TR_IDENTITY_PARTS if tr else EN_IDENTITY_PARTS
    # 6×6×6 = 216 surface combinations per language before neural sampling.
    idx=max(0,int(i))
    a=parts["open"][idx%len(parts["open"])]
    b=parts["architecture"][(idx//len(parts["open"]))%len(parts["architecture"])]
    c=parts["purpose"][(idx//(len(parts["open"])*len(parts["architecture"])))%len(parts["purpose"])]
    return f"{a} {b} {c}"


TR_IDENTITY_OPENERS=(
    "Ben FLM'im.",
    "Adım FLM.",
    "Bana FLM diyebilirsin.",
    "Kısaca FLM adlı bir yapay zekâ asistanıyım.",
    "Benim adım FLM; bir yapay zekâ asistanıyım.",
    "FLM olarak çalışan bir yapay zekâ asistanıyım.",
)
TR_IDENTITY_ARCH=(
    "Dili tek bir eğitilebilir arayüz Transformer'ı ile işlerim, asıl muhakemeyi Semantic Core yürütür.",
    "Doğal dili bir arayüz modeli anlar; planlama, hesaplama ve doğrulama çekirdekte yapılır.",
    "Dil üretimi neural arayüzde, yapılandırılmış reasoning ise eğitim gerektirmeyen Semantic Core'da çalışır.",
    "Tek bir dil modeliyle iletişim kurar, mantık ve araç kullanımını Core tarafında gerçekleştiririm.",
    "İstekleri Semantic IR'ye çevirir, doğrulanabilir işlemleri Core içinde yürütürüm.",
    "Arayüz Transformer'ı dil için kullanılır; hafıza, planlama ve doğrulama ayrı Semantic Core bileşenleridir.",
)
TR_IDENTITY_PURPOSE=(
    "Amacım isteğini anlayıp doğrulanmış sonucu doğal biçimde aktarmak.",
    "Böylece cevap üretirken yalnızca ezberlenmiş metni tekrar etmek yerine doğrulanmış anlamı kullanırım.",
    "Görevim sorunu anlamak, uygun çekirdek işlemlerini çalıştırmak ve sonucu anlaşılır biçimde sunmak.",
    "Bu yapı bilgi, kodlama ve planlama görevlerinde sonucu önce doğrulayıp sonra ifade etmeme yardımcı olur.",
    "Cevaplarımı mümkün olduğunca Core'un doğruladığı anlamdan yeniden oluştururum.",
    "Yeni bilgileri ham cümle olarak ezberlemek yerine yapılandırılmış anlam üzerinden kullanmayı hedeflerim.",
)
EN_IDENTITY_OPENERS=(
    "I'm FLM.",
    "My name is FLM.",
    "You can call me FLM.",
    "In short, I'm an AI assistant called FLM.",
    "I'm an AI assistant named FLM.",
    "I operate as FLM, an AI assistant.",
)
EN_IDENTITY_ARCH=(
    "One trainable interface Transformer handles language while the Semantic Core performs the structured reasoning.",
    "A language interface interprets requests, while planning, calculation, and verification run in the Core.",
    "Neural generation handles language, while training-free Semantic Core components handle structured reasoning.",
    "I use one language model for communication and keep logic and tool execution in the Core.",
    "I translate requests into Semantic IR and execute verifiable operations inside the Core.",
    "The interface Transformer handles language; memory, planning, and verification live in separate Core components.",
)
EN_IDENTITY_PURPOSE=(
    "My goal is to understand the request and express a verified result naturally.",
    "That lets me answer from verified meaning instead of merely replaying memorized wording.",
    "My job is to understand the problem, run the appropriate Core operations, and present the result clearly.",
    "This structure lets me verify information, coding, and planning results before wording the answer.",
    "I try to reconstruct answers from Core-verified meaning rather than copying stored text.",
    "The design aims to use new information as structured meaning instead of memorizing raw sentences.",
)


def _compose_identity(i:int,tr:bool)->str:
    cycle=max(0,i//(SEMANTIC_MODES*2))
    if tr:
        a=TR_IDENTITY_OPENERS[cycle%len(TR_IDENTITY_OPENERS)]
        b=TR_IDENTITY_ARCH[(cycle//len(TR_IDENTITY_OPENERS))%len(TR_IDENTITY_ARCH)]
        d=TR_IDENTITY_PURPOSE[
            (cycle//(len(TR_IDENTITY_OPENERS)*len(TR_IDENTITY_ARCH)))%len(TR_IDENTITY_PURPOSE)
        ]
    else:
        a=EN_IDENTITY_OPENERS[cycle%len(EN_IDENTITY_OPENERS)]
        b=EN_IDENTITY_ARCH[(cycle//len(EN_IDENTITY_OPENERS))%len(EN_IDENTITY_ARCH)]
        d=EN_IDENTITY_PURPOSE[
            (cycle//(len(EN_IDENTITY_OPENERS)*len(EN_IDENTITY_ARCH)))%len(EN_IDENTITY_PURPOSE)
        ]
    return f"{a} {b} {d}"


def _normalized_words(text:str)->str:
    text=re.sub(r"[^0-9A-Za-zÇĞİÖŞÜçğıöşü]+"," ",str(text or "").casefold())
    return " ".join(text.split())


def copy_similarity(source:str,answer:str)->float:
    a=_normalized_words(source)
    b=_normalized_words(answer)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None,a,b).ratio()


def has_long_verbatim_overlap(source:str,answer:str,min_words:int=9)->bool:
    a=_normalized_words(source).split()
    b=_normalized_words(answer).split()
    if len(a)<min_words or len(b)<min_words:
        return False
    grams={tuple(a[i:i+min_words]) for i in range(len(a)-min_words+1)}
    return any(tuple(b[i:i+min_words]) in grams for i in range(len(b)-min_words+1))


def _assert_not_copy(user:str,final:str)->None:
    # Tiny answers such as "Evet." are semantic labels rather than copied prose.
    if len(_normalized_words(final))<18:
        return
    score=copy_similarity(user,final)
    if has_long_verbatim_overlap(user,final):
        raise RuntimeError(
            f"semantic curriculum contains long verbatim span user={user!r} final={final!r}"
        )
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
        ("Sen kimsin?" if (i//(SEMANTIC_MODES*2))%2==0 else "Kendini tanıt.")
        if tr else
        ("Who are you?" if (i//(SEMANTIC_MODES*2))%2==0 else "Introduce yourself.")
    )
    prog=semantic_ir(operations=[operation("IDENTITY")],source="synthetic-semantic-curriculum-v5")
    final=identity_surface(i//(SEMANTIC_MODES*2),tr)
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
            f"Özetle {a}, memeli sınıfına ait bir kedidir.",
            f"{a} için iki bilgi birlikte geçerli: kedi olması ve memeli sınıfında yer alması.",
            f"{a}'nın türü kedidir; biyolojik sınıf olarak da memeliler içinde değerlendirilir.",
            f"Doğrulanan bilgilere göre {a}, memeli olan bir kedidir.",
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
            f"The verified information places {a} among mammals, specifically as a cat.",
            f"Two facts apply to {a}: it is a cat and it belongs to the mammal class.",
            f"{a}'s specific type is cat, while its broader biological class is mammal.",
            f"Taken together, the facts describe {a} as a cat within the mammals.",
        )
    prog=semantic_ir(
        operations=[operation("SYNTHESIZE_FACTS",facts=facts)],
        source="synthetic-semantic-curriculum-v5",
    )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]


def _answer_style_case(i:int,tr:bool):
    value=40+(i%53)
    user=(
        f"Çekirdek sonucu {value}. Cevabı veri satırını tekrar etmeden doğal Türkçe ile söyle."
        if tr else
        f"The Core result is {value}. Answer naturally without repeating the data row verbatim."
    )
    prog=semantic_ir(
        operations=[operation("ARITHMETIC",expression=str(value))],
        source="synthetic-semantic-curriculum-v5",
    )
    if tr:
        variants=(
            f"Hesabın sonucu {value}.",
            f"Sonuç olarak {value} elde ediliyor.",
            f"Bu işlem {value} değerini veriyor.",
            f"Doğrulanan sonuç {value}.",
            f"İşlemi hesapladığımızda sonuç {value} oluyor.",
            f"Çekirdeğin doğruladığı değer {value}.",
            f"Kısaca cevap {value}.",
            f"Hesaplama {value} sonucuna ulaşıyor.",
        )
    else:
        variants=(
            f"The calculation comes out to {value}.",
            f"The verified result is {value}.",
            f"This evaluates to {value}.",
            f"The result of the calculation is {value}.",
            f"Evaluating the expression gives {value}.",
            f"The Core verifies the value as {value}.",
            f"In short, the answer is {value}.",
            f"The calculation reaches a result of {value}.",
        )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]


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
        source="synthetic-semantic-curriculum-v5",
    )
    variants=(
        (
            f"{name} adlı fonksiyon, aldığı sayının iki katını döndürüyor.",
            f"Bu fonksiyon girdiyi 2 ile çarpıp sonucu geri veriyor; adı {name}.",
            f"{name}, tek bir değer alıp onu iki katına çıkaran basit bir fonksiyon.",
            f"{name} girdiyi ikiyle çarpar; yani verilen değerin iki katını üretir.",
            f"Bu kod {name} fonksiyonunu tanımlar ve fonksiyon kendisine verilen değeri iki katına çıkarır.",
            f"Fonksiyonun davranışı basit: {name}, aldığı x değerinin iki katını sonuç olarak verir.",
        )
        if tr else
        (
            f"The function {name} takes one value and returns twice that value.",
            f"{name} simply doubles its input and returns the result.",
            f"This defines {name}, a small function that multiplies the input by two.",
            f"{name} multiplies its argument by two, so it produces twice the supplied value.",
            f"The code defines {name}; its behavior is simply to double the value it receives.",
            f"The function {name} returns two times its input value.",
        )
    )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]



def _multi_fact_synthesis_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    if tr:
        user=(
            f"Bilgi notları: {a} bir kedidir; memelidir; evde yaşayan bir hayvandır. "
            "Bu notları kopyalamadan tek bir doğal cevap oluştur."
        )
        facts=[
            {"subject":a,"predicate":"type","object":"kedi"},
            {"subject":a,"predicate":"class","object":"memeli"},
            {"subject":a,"predicate":"habitat","object":"ev"},
        ]
        variants=(
            f"{a}, ev ortamında yaşayan memeli bir kedidir.",
            f"{a}'yı evde yaşayan bir kedi ve memeli olarak tanımlayabiliriz.",
            f"Özetle {a}, memeliler sınıfında yer alan ve evde yaşayan bir kedidir.",
            f"{a}, kedi türündedir; memelidir ve yaşam alanı evdir.",
        )
    else:
        user=(
            f"Knowledge notes: {a} is a cat; a mammal; and an animal that lives at home. "
            "Create one natural answer without copying the notes."
        )
        facts=[
            {"subject":a,"predicate":"type","object":"cat"},
            {"subject":a,"predicate":"class","object":"mammal"},
            {"subject":a,"predicate":"habitat","object":"home"},
        ]
        variants=(
            f"{a} is a mammalian cat that lives at home.",
            f"{a} can be described as a cat, a mammal, and a home-dwelling animal.",
            f"In short, {a} is a cat in the mammal class whose habitat is the home.",
            f"{a} belongs to the mammals, is specifically a cat, and lives at home.",
        )
    prog=semantic_ir(
        operations=[operation("SYNTHESIZE_FACTS",facts=facts)],
        source="synthetic-semantic-curriculum-v5",
    )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]


def _identity_working_case(i:int,tr:bool):
    user=(
        ("Nasıl çalışıyorsun?" if (i//(SEMANTIC_MODES*2))%2==0 else "Cevaplarını nasıl oluşturuyorsun?")
        if tr else
        ("How do you work?" if (i//(SEMANTIC_MODES*2))%2==0 else "How do you form your answers?")
    )
    prog=semantic_ir(operations=[operation("IDENTITY")],source="synthetic-semantic-curriculum-v5")
    return user,prog,identity_surface((i//(SEMANTIC_MODES*2))+37,tr)


def _relation_synthesis_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    b=("Ankara" if tr else "London")
    country=("Türkiye" if tr else "United Kingdom")
    if tr:
        user=(
            f"Kaynak: {a} {b}'dadır. {b}, {country} içindedir. "
            "Bilgiyi liste gibi tekrar etmeden doğal bir özet yaz."
        )
        facts=[
            {"subject":a,"predicate":"located_in","object":b},
            {"subject":b,"predicate":"located_in","object":country},
        ]
        variants=(
            f"{a} {b}'da bulunuyor; {b} da {country} sınırları içinde.",
            f"Konum bilgisine göre {a}, {country}'deki {b} şehrinde.",
            f"{a}'nın bulunduğu yer {b}; bu şehir {country} içindedir.",
        )
    else:
        user=(
            f"Source: {a} is in {b}. {b} is in the {country}. "
            "Write a natural summary instead of repeating the source lines."
        )
        facts=[
            {"subject":a,"predicate":"located_in","object":b},
            {"subject":b,"predicate":"located_in","object":country},
        ]
        variants=(
            f"{a} is located in {b}, which is in the {country}.",
            f"The location information places {a} in {b} within the {country}.",
            f"{a}'s location is {b}, a city in the {country}.",
        )
    prog=semantic_ir(
        operations=[operation("SYNTHESIZE_FACTS",facts=facts)],
        source="synthetic-semantic-curriculum-v5",
    )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]


def _knowledge_answer_case(i:int,tr:bool,names):
    a=names[i%len(names)]
    if tr:
        user=(
            f"Yeni bilgi olarak {a}'nın bir kedi ve memeli olduğunu hafızana ekle. "
            f"Ardından {a} hakkında bildiklerini kaynak cümleyi kopyalamadan anlat."
        )
        variants=(
            f"{a}, memeliler sınıfına ait bir kedidir.",
            f"Hafızamdaki doğrulanmış bilgilere göre {a} hem kedi hem de memelidir.",
            f"{a} için kayıtlı sınıflandırma onu memeli bir kedi olarak gösteriyor.",
            f"Özetle {a}, kedi türünde ve memeliler grubunda yer alıyor.",
            f"{a} hakkındaki semantik kayıtlarda kedi ve memeli sınıflandırmaları birlikte bulunuyor.",
        )
    else:
        user=(
            f"Store the new facts that {a} is a cat and a mammal. Then describe what "
            f"you know about {a} without copying the source sentence."
        )
        variants=(
            f"{a} is a cat that belongs to the mammal class.",
            f"According to verified memory, {a} is both a cat and a mammal.",
            f"The stored classification describes {a} as a mammalian cat.",
            f"In short, {a} is recorded as a cat within the mammals.",
            f"Semantic memory associates {a} with both the cat and mammal classes.",
        )
    prog=semantic_ir(
        facts=[atom("IsA",a,"Cat"),atom("IsA",a,"Mammal")],
        operations=[operation("DESCRIBE_ENTITY",entity=a)],
        source="synthetic-semantic-curriculum-v5",
    )
    return user,prog,variants[(i//(SEMANTIC_MODES*2))%len(variants)]

def messages_for(i:int):
    mode=i%SEMANTIC_MODES
    cycle=i//SEMANTIC_MODES
    # Every semantic mode appears in both languages instead of mode parity
    # accidentally locking a task to only Turkish or only English.
    tr=(cycle%2==0)
    names=TR_NAMES if tr else EN_NAMES
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
            if (i//(SEMANTIC_MODES*2))%2==0
            else _long_chain_case(i,tr,names)
        )
    elif mode==12:
        user,prog,final=_identity_case(i,tr)
    elif mode==13:
        user,prog,final=_fact_synthesis_case(i,tr,names)
    elif mode==14:
        user,prog,final=_answer_style_case(i,tr)
    elif mode==15:
        user,prog,final=_code_explain_case(i,tr)
    elif mode==16:
        user,prog,final=_multi_fact_synthesis_case(i,tr,names)
    elif mode==17:
        user,prog,final=_identity_working_case(i,tr)
    elif mode==18:
        user,prog,final=_relation_synthesis_case(i,tr,names)
    else:
        user,prog,final=_knowledge_answer_case(i,tr,names)

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
    # Each language gets both compiler-supervised and renderer-only examples.
    # Renderer-only rows deliberately hide the raw source wording so the model
    # must compose from Semantic IR + verified Core state rather than copy input.
    cycle=i//SEMANTIC_MODES
    train_ir=((cycle//2)%2)==0
    if train_ir:
        model_user=user
    elif tr:
        model_user=(
            "Doğrulanmış Semantic IR ve Core sonucunu kullanarak kendi cümlelerinle doğal "
            "bir Türkçe cevap üret. Ham kaynak metin gizlidir; onu yeniden kurmaya çalışma."
        )
    else:
        model_user=(
            "Use the verified Semantic IR and Core result to write a fresh natural English "
            "answer. The raw source wording is hidden; do not try to reconstruct it."
        )
    return [
        {"role":"system","content":system},
        {"role":"user","content":model_user},
        {"role":"assistant","content":ir,"train":train_ir},
        {"role":"tool","content":core},
        {"role":"assistant","content":"<|final|>\n"+final,"train":True},
    ]


def append_semantic_curriculum(path:Path,rows:int=1_200_000,seed:int=7071)->dict:
    rng=random.Random(seed)
    indices=list(range(rows))
    rng.shuffle(indices)
    mode_counts={str(i):0 for i in range(SEMANTIC_MODES)}
    language_counts={"tr":0,"en":0}
    render_focused_rows=0
    with path.open("a",encoding="utf-8") as fh:
        for i in indices:
            messages=messages_for(i)
            fh.write(json.dumps({
                "messages":messages,
                "source":"synthetic:flm-semantic-curriculum-v5",
            },ensure_ascii=False)+"\n")
            mode_counts[str(i%SEMANTIC_MODES)]+=1
            language_counts["tr" if ((i//SEMANTIC_MODES)%2==0) else "en"]+=1
            if not bool(messages[2].get("train",True)):
                render_focused_rows+=1
    return {
        "rows":rows,
        "modes":SEMANTIC_MODES,
        "mode_counts":mode_counts,
        "language_counts":language_counts,
        "source":"synthetic:flm-semantic-curriculum-v5",
        "core_executed":True,
        "anti_copy_verified":True,
        "anti_copy_phrase_words":9,
        "render_focused_rows":render_focused_rows,
        "ir_supervision_policy":"1/2 compiler+render, 1/2 source-hidden render-only",
        "source_hidden_rendering":True,
        "identity_variants_tr":216,
        "identity_variants_en":216,
        "response_synthesis_modes":[13,14,15,16,17,18,19],
    }
