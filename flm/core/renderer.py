from __future__ import annotations

import random

from .runtime import CoreResponse


def _is_turkish(text: str) -> bool:
    low=str(text or "").casefold()
    return any(x in low for x in (
        " mı"," mi"," mu"," mü"," bir ","hesapla","kodu","kodunu","analiz","sıcakkanlı",
        "kaydet","dosya","nedir","kaç","işlemi","memeli","bilgisayar","hayvan","araç",
        "sen ","kimsin","kendini tanıt","kendini tanit","nesin",
    )) or any(ch in low for ch in "çğıöşü")


class DeterministicRenderer:
    """Training-free renderer for verified FLM Core results.

    Semantic content stays fixed while selected conversational outputs may vary
    in wording. Variation never changes the verified Core facts.
    """

    def __init__(self):
        self._recent: dict[str,list[str]] = {}

    def _choose_nonrepeating(self, key:str, variants:tuple[str,...])->str:
        history=self._recent.get(key,[])
        # Exhaust the entire surface bank before allowing any wording to repeat.
        # Meaning remains fixed by Core; only phrasing changes.
        used=set(history)
        choices=[x for x in variants if x not in used]
        if not choices:
            history=[]
            choices=list(variants)
        answer=random.SystemRandom().choice(choices)
        history.append(answer)
        self._recent[key]=history
        return answer

    @staticmethod
    def _identity_variants(tr: bool) -> tuple[str,...]:
        if tr:
            openers=(
                "Ben FLM'im.",
                "Adım FLM.",
                "Bana FLM diyebilirsin.",
                "Kısaca, ben FLM adlı bir yapay zekâ asistanıyım.",
                "Benim adım FLM.",
                "FLM olarak çalışan bir yapay zekâ asistanıyım.",
            )
            architecture=(
                "Metni tokenizer ve Semantic Compiler ile işler, doğrulanabilir muhakemeyi Semantic Core'da yürütürüm.",
                "Doğal dili kural tabanlı Semantic Compiler yorumlar; planlama, mantık ve doğrulama çekirdekte çalışır.",
                "İstekleri Semantic IR'ye dönüştürür, hesaplama ve yapılandırılmış reasoning'i Core'a bırakırım.",
                "Tokenizer, semantik hafıza, planlayıcı ve doğrulayıcı çekirdeği birlikte kullanırım.",
                "Metni anlam yapısına çevirmek için Semantic Compiler'ı, sonuçları hesaplayıp kontrol etmek için Semantic Core'u kullanırım.",
                "Dil katmanım isteği anlam yapısına çevirir; asıl yapılandırılmış işlemleri çekirdeğim yapar.",
            )
            purpose=(
                "Amacım doğrulanmış sonucu doğal ve yararlı bir cevap halinde sunmak.",
                "Cevabı ezberlenmiş bir satırdan kopyalamak yerine doğrulanmış anlamdan yeniden kurarım.",
                "Aynı gerçeği korurken ifadeyi bağlama uygun biçimde değiştirebilirim.",
                "Bilgiyi ham metin olarak tekrarlamak yerine semantik içeriğini kullanırım.",
                "Hedefim tutarlı, doğrulanabilir ve doğal cevaplar üretmek.",
                "Core sonucunu kullanıcıya uygun yeni bir anlatımla ifade ederim.",
            )
        else:
            openers=(
                "I'm FLM.",
                "My name is FLM.",
                "You can call me FLM.",
                "In short, I'm an AI assistant called FLM.",
                "I go by FLM.",
                "I operate as FLM, an AI assistant.",
            )
            architecture=(
                "A tokenizer and Semantic Compiler handle text while verifiable reasoning runs in my Semantic Core.",
                "A rule-based Semantic Compiler interprets supported language; planning, logic, and verification run in the Core.",
                "I translate requests into Semantic IR and leave structured computation and reasoning to the Core.",
                "I combine a tokenizer with semantic memory, planning, verification, and a training-free response composer.",
                "The Semantic Compiler handles text while the Core calculates and checks structured results.",
                "My language layer converts requests into semantic structure and the Core performs the actual operations.",
            )
            purpose=(
                "My goal is to turn verified results into natural and useful answers.",
                "I reconstruct answers from verified meaning instead of copying a memorized source line.",
                "I can vary the wording while keeping the underlying conclusion unchanged.",
                "I use the semantic content of information instead of replaying raw text.",
                "The aim is to produce natural, consistent, and verifiable responses.",
                "I express Core results in fresh wording suited to the user.",
            )
        return tuple(f"{a} {b} {d}" for a in openers for b in architecture for d in purpose)

    def render(self, prompt: str, response: CoreResponse) -> str:
        tr=_is_turkish(prompt)

        if response.results:
            item=response.results[0]
            return ("Evet." if item.answer else "Hayır.") if tr else ("Yes." if item.answer else "No.")

        if response.operation_results:
            item=response.operation_results[0]
            kind=item.get("kind")
            if kind=="VERIFY":
                checks=item.get("checks") or []
                failed=[x for x in checks if not x.get("ok",False)]
                if failed:
                    names=", ".join(str(x.get("name") or "check") for x in failed)
                    return (
                        f"Doğrulama geçmedi; başarısız kontroller: {names}."
                        if tr else
                        f"Verification failed; failing checks: {names}."
                    )
                if item.get("ok",False):
                    return "Tüm kontroller geçti." if tr else "All checks passed."

            if not item.get("ok",False):
                msg=item.get("message") or item.get("error") or "unknown error"
                return (f"İşlem başarısız: {msg}" if tr else f"Operation failed: {msg}")

            if kind=="IDENTITY":
                variants=self._identity_variants(tr)
                return self._choose_nonrepeating("identity-tr" if tr else "identity-en",variants)

            if kind=="DESCRIBE_ENTITY":
                entity=str(item.get("entity") or "Bu varlık")
                facts=item.get("facts") or []
                if not facts:
                    return (
                        f"{entity} hakkında hafızamda doğrulanmış bir bilgi yok."
                        if tr else
                        f"I don't have any verified facts about {entity} in memory."
                    )
                pieces=[]
                for fact in facts:
                    pred=str(fact.get("predicate") or "")
                    args=[str(x) for x in (fact.get("args") or [])]
                    if pred=="IsA" and len(args)>=2:
                        value=args[1]
                        if tr:
                            tr_types={
                                "Cat":"kedi","Dog":"köpek","Bird":"kuş","Mammal":"memeli",
                                "Vehicle":"araç","Computer":"bilgisayar","Program":"program",
                                "File":"dosya","Human":"insan","Animal":"hayvan",
                            }
                            value=tr_types.get(value,value)
                            pieces.append(f"{entity} bir {value}")
                        else:
                            pieces.append(f"{entity} is a {value}")
                    elif pred=="WarmBlooded":
                        pieces.append(
                            f"{entity} sıcakkanlıdır" if tr else f"{entity} is warm-blooded"
                        )
                    elif pred=="LocatedIn" and len(args)>=2:
                        pieces.append(
                            f"{entity}, {args[1]} içindedir" if tr else f"{entity} is in {args[1]}"
                        )
                    elif pred=="TallerThan" and len(args)>=2:
                        pieces.append(
                            f"{entity}, {args[1]}'den uzundur" if tr else f"{entity} is taller than {args[1]}"
                        )
                    elif len(args)>=2:
                        pieces.append(f"{pred}({', '.join(args)})")
                if not pieces:
                    return (
                        f"{entity} hakkında {len(facts)} doğrulanmış kayıt var."
                        if tr else
                        f"There are {len(facts)} verified records about {entity}."
                    )
                if tr:
                    variants=(
                        f"{entity} hakkında bildiğim şu: " + "; ".join(pieces) + ".",
                        f"Hafızamdaki doğrulanmış bilgilere göre " + "; ".join(pieces) + ".",
                        f"{entity} için kayıtlı bilgiler özetle şöyle: " + "; ".join(pieces) + ".",
                    )
                    return self._choose_nonrepeating("describe-tr:"+entity,variants)
                variants=(
                    f"Here is what I know about {entity}: " + "; ".join(pieces) + ".",
                    f"According to my verified memory, " + "; ".join(pieces) + ".",
                    f"The stored facts about {entity} can be summarized as: " + "; ".join(pieces) + ".",
                )
                return self._choose_nonrepeating("describe-en:"+entity,variants)

            if kind=="SYNTHESIZE_FACTS":
                facts=item.get("facts") or []
                if not facts:
                    return "Kullanılabilir bilgi yok." if tr else "No usable facts."
                subject=str(facts[0].get("subject") or ("Bu varlık" if tr else "This entity"))
                tr_values={
                    "cat":"kedi","Cat":"kedi","dog":"köpek","Dog":"köpek",
                    "bird":"kuş","Bird":"kuş","mammal":"memeli","Mammal":"memeli",
                    "animal":"hayvan","Animal":"hayvan","computer":"bilgisayar",
                    "Computer":"bilgisayar","vehicle":"araç","Vehicle":"araç",
                    "home":"ev","File":"dosya","Program":"program",
                }
                classes=[]
                locations=[]
                other=[]
                for fact in facts:
                    pred=str(fact.get("predicate") or "").casefold()
                    value=str(fact.get("object") or "").strip()
                    if not value:
                        continue
                    if pred in {"type","class","isa","is_a"}:
                        classes.append(tr_values.get(value,value) if tr else value.lower())
                    elif pred in {"habitat","located_in","location"}:
                        locations.append(tr_values.get(value,value) if tr else value)
                    else:
                        other.append((pred,value))
                classes=list(dict.fromkeys(classes))
                locations=list(dict.fromkeys(locations))
                if tr:
                    clauses=[]
                    if classes:
                        if len(classes)==1:
                            clauses.append(f"{subject} bir {classes[0]}")
                        else:
                            clauses.append(f"{subject} hem {', hem '.join(classes)}")
                    if locations:
                        clauses.append(f"{subject} {', '.join(locations)} ortamında bulunur")
                    for pred,value in other:
                        clauses.append(f"{subject} için {pred} bilgisi {tr_values.get(value,value)}")
                    meaning="; ".join(clauses) or f"{subject} hakkında doğrulanmış bilgiler mevcut"
                    variants=(
                        meaning+".",
                        f"Hafızamdaki doğrulanmış bilgilere göre {meaning[0].lower()+meaning[1:]}.",
                        f"Özetle, {meaning[0].lower()+meaning[1:]}.",
                        f"{subject} hakkındaki semantik kayıtların ortak sonucu şu: {meaning[0].lower()+meaning[1:]}.",
                        f"Bilgileri birlikte değerlendirdiğimde {meaning[0].lower()+meaning[1:]}.",
                    )
                    return self._choose_nonrepeating("synth-tr:"+subject,variants)
                clauses=[]
                if classes:
                    clauses.append(
                        f"{subject} is a {classes[0]}" if len(classes)==1
                        else f"{subject} is both " + " and ".join(classes)
                    )
                if locations:
                    clauses.append(f"{subject} is associated with {', '.join(locations)}")
                for pred,value in other:
                    clauses.append(f"{subject}'s {pred} is {value}")
                meaning="; ".join(clauses) or f"verified information is stored about {subject}"
                variants=(
                    meaning+".",
                    f"According to verified memory, {meaning[0].lower()+meaning[1:]}.",
                    f"In short, {meaning[0].lower()+meaning[1:]}.",
                    f"The semantic records about {subject} jointly indicate that {meaning[0].lower()+meaning[1:]}.",
                    f"Putting the verified facts together, {meaning[0].lower()+meaning[1:]}.",
                )
                return self._choose_nonrepeating("synth-en:"+subject,variants)

            if kind=="ARITHMETIC":
                return (f"Sonuç: {item.get('value')}." if tr else f"Result: {item.get('value')}.")

            if kind=="COMPARE_VALUES":
                left=item.get("left"); right=item.get("right"); relation=item.get("relation")
                if tr:
                    variants={
                        "greater":(
                            f"{left}, {right}'den büyüktür.",
                            f"Karşılaştırmada büyük olan değer {left}; diğer değer {right}.",
                            f"{left} değeri {right} değerini aşıyor.",
                        ),
                        "less":(
                            f"{left}, {right}'den küçüktür.",
                            f"Karşılaştırmada {right} daha büyük; {left} daha küçük kalıyor.",
                            f"{left} değeri {right} değerinin altında.",
                        ),
                        "equal":(
                            f"{left} ile {right} eşittir.",
                            f"İki değer de aynı: {left}.",
                            f"Karşılaştırma eşitlik veriyor; her ikisi de {left}.",
                        ),
                    }
                    return self._choose_nonrepeating(f"compare-tr:{left}:{right}",variants.get(relation,variants["equal"]))
                variants={
                    "greater":(
                        f"{left} is greater than {right}.",
                        f"The larger value is {left}; the other value is {right}.",
                        f"{left} exceeds {right}.",
                    ),
                    "less":(
                        f"{left} is less than {right}.",
                        f"{right} is the larger value, while {left} is smaller.",
                        f"{left} falls below {right}.",
                    ),
                    "equal":(
                        f"{left} and {right} are equal.",
                        f"Both values are the same: {left}.",
                        f"The comparison is equal; each side is {left}.",
                    ),
                }
                return self._choose_nonrepeating(f"compare-en:{left}:{right}",variants.get(relation,variants["equal"]))

            if kind=="SORT_VALUES":
                ordered=item.get("sorted") or []
                rendered=", ".join(str(x) for x in ordered)
                if tr:
                    direction="büyükten küçüğe" if item.get("descending") else "küçükten büyüğe"
                    variants=(
                        f"{direction.capitalize()} sıralama: {rendered}.",
                        f"Değerleri {direction} dizince sonuç {rendered}.",
                        f"Sıralanmış liste ({direction}): {rendered}.",
                    )
                    return self._choose_nonrepeating(f"sort-tr:{rendered}:{direction}",variants)
                direction="descending" if item.get("descending") else "ascending"
                variants=(
                    f"{direction.capitalize()} order: {rendered}.",
                    f"Sorting the values in {direction} order gives {rendered}.",
                    f"The {direction} sequence is {rendered}.",
                )
                return self._choose_nonrepeating(f"sort-en:{rendered}:{direction}",variants)

            if kind=="ANALYZE_CODE":
                diagnostics=item.get("diagnostics") or []
                if diagnostics:
                    first=diagnostics[0]
                    return (f"Kod analizi bir hata buldu: {first}" if tr else f"Code analysis found an error: {first}")
                symbols=[x.get("name","") for x in item.get("symbols") or [] if x.get("name")]
                calls=[" -> ".join(x) for x in item.get("calls") or []]
                if tr:
                    return f"Kod analizi tamamlandı. Semboller: {', '.join(symbols) or 'yok'}. Çağrılar: {', '.join(calls) or 'yok'}."
                return f"Code analysis complete. Symbols: {', '.join(symbols) or 'none'}. Calls: {', '.join(calls) or 'none'}."

            if kind=="STATE_PLAN":
                actions=item.get("actions") or []
                return (f"Plan: {' -> '.join(actions)}." if tr else f"Plan: {' -> '.join(actions)}.")

            if kind=="UI_PLAN":
                actions=item.get("actions") or []
                rendered=[str(x.get("op"))+(f"({x.get('target_id')})" if x.get("target_id") else "") for x in actions]
                return (f"Eylemler: {' -> '.join(rendered)}." if tr else f"Actions: {' -> '.join(rendered)}.")

        if response.memory_facts or response.memory_rules:
            return "Bilgi kaydedildi." if tr else "Knowledge recorded."

        return "Sonuç yok." if tr else "No result."
