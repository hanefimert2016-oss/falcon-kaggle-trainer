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
        # Avoid short-cycle repetition. For large response banks this keeps
        # the last eight phrasings out of the candidate set; for small banks it
        # uses every alternative before allowing a repeat.
        recent=set(history[-min(8,max(1,len(variants)-1)):])
        choices=[x for x in variants if x not in recent] or list(variants)
        answer=random.SystemRandom().choice(choices)
        history.append(answer)
        self._recent[key]=history[-16:]
        return answer

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
                if tr:
                    variants=(
                        "Ben FLM'im. İstekleri anlayıp Semantic Core üzerinden akıl yürüten bir yapay zekâ asistanıyım.",
                        "Adım FLM. Tek bir dil arayüzü modeliyle çalışan, asıl muhakemeyi Semantic Core'da yapan bir yapay zekâ asistanıyım.",
                        "Ben FLM adlı yapay zekâ asistanıyım; dili arayüz modeli işler, planlama ve doğrulama gibi işleri ise çekirdeğim yürütür.",
                        "FLM'im; bir yapay zekâ asistanı olarak soruları anlar, çekirdek araçlarla sonuç üretir ve bunu doğal biçimde aktarırım.",
                        "Bana FLM diyebilirsin. Bir yapay zekâ asistanıyım; dil tarafını tek bir Transformer işler, hesaplama, mantık ve doğrulama Semantic Core'da yürür.",
                        "Ben FLM adlı bir yapay zekâ asistanıyım. Soruyu önce anlam yapısına çevirir, ardından çekirdekteki araçlarla doğrulanmış bir sonuç üretirim.",
                        "FLM adlı bir AI asistanıyım. Ezberlenmiş tek bir cevap vermek yerine Core sonucunu kullanarak cevabı yeniden kurarım.",
                        "Kısaca FLM adlı bir yapay zekâ asistanıyım: dil için bir arayüz modelim, yapılandırılmış muhakeme için de eğitim gerektirmeyen bir çekirdeğim var.",
                        "Benim adım FLM; bir yapay zekâ asistanıyım. İsteğini yorumlar, gereken işlemleri Semantic Core'da çalıştırır ve sonucu doğal dille ifade ederim.",
                        "FLM'im. Tek bir eğitilebilir arayüz ile semantik hafıza, planlama ve doğrulamayı bir arada kullanan bir yapay zekâ asistanıyım.",
                        "Ben FLM adlı asistanım; ne istediğini arayüzden anlar, sonucu çekirdekte hesaplayıp farklı ama tutarlı biçimlerde anlatabilirim.",
                        "Adım FLM; bir yapay zekâ asistanıyım. Cevaplarımın anlamını Core belirler, cümleleri ise bağlama uygun biçimde yeniden oluştururum.",
                    )
                else:
                    variants=(
                        "I'm FLM, an AI assistant that uses a Semantic Core for reasoning and a single trainable interface model for language.",
                        "My name is FLM. I'm an AI assistant whose reasoning is handled by a Semantic Core while one interface model handles language.",
                        "I'm FLM: an AI assistant designed to understand requests, reason through a Semantic Core, and return useful answers.",
                        "I'm FLM, an AI assistant. I use one language interface model, while planning, verification, and other reasoning live in my Core.",
                        "You can call me FLM. I'm an AI assistant; one Transformer handles language, while structured logic and verification run in the Semantic Core.",
                        "I'm FLM, an AI assistant. I translate requests into semantic structure, execute the relevant Core operations, and then phrase the result naturally.",
                        "I'm an AI assistant named FLM. Rather than replaying one memorized sentence, I render answers from verified Core meaning.",
                        "In short, I'm FLM, an AI assistant with one trainable language interface paired with a training-free structured reasoning Core.",
                        "My name is FLM. I'm an AI assistant that interprets your request, lets the Semantic Core do the structured work, and explains the verified result.",
                        "I'm FLM, an AI assistant combining a language interface with semantic memory, planning, tools, and verification in the Core.",
                        "I'm the FLM assistant. The Core fixes the meaning of the answer while the interface can express it in different faithful ways.",
                        "I'm FLM, an AI assistant. My answers are generated from Core-verified meaning rather than copied from a fixed response string.",
                    )
                return self._choose_nonrepeating("tr" if tr else "en",variants)

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
                subject=str(facts[0].get("subject") or "Bu varlık")
                values=[str(x.get("object") or "") for x in facts if x.get("object")]
                values=list(dict.fromkeys(values))
                if tr:
                    variants=(
                        f"{subject} hakkında doğrulanan bilgiler: {', '.join(values)}.",
                        f"{subject}, {', '.join(values)} özellikleriyle tanımlanıyor.",
                        f"Özetle {subject} için geçerli bilgiler {', '.join(values)}.",
                    )
                    return self._choose_nonrepeating("synth-tr:"+subject,variants)
                variants=(
                    f"The verified facts about {subject} are: {', '.join(values)}.",
                    f"{subject} is described by these verified facts: {', '.join(values)}.",
                    f"In short, the Core associates {subject} with {', '.join(values)}.",
                )
                return self._choose_nonrepeating("synth-en:"+subject,variants)

            if kind=="ARITHMETIC":
                return (f"Sonuç: {item.get('value')}." if tr else f"Result: {item.get('value')}.")

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
