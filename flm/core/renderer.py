from __future__ import annotations

import random

from .runtime import CoreResponse


def _is_turkish(text: str) -> bool:
    low=str(text or "").casefold()
    return any(x in low for x in (
        " mı"," mi"," mu"," mü"," bir ","hesapla","kodu","kodunu","analiz","sıcakkanlı",
        "kaydet","dosya","nedir","kaç","işlemi","memeli","bilgisayar","hayvan","araç",
    )) or any(ch in low for ch in "çğıöşü")


class DeterministicRenderer:
    """Training-free renderer for verified FLM Core results.

    Semantic content stays fixed while selected conversational outputs may vary
    in wording. Variation never changes the verified Core facts.
    """

    def __init__(self):
        self._last_identity: dict[str,str] = {}

    def _choose_nonrepeating(self, key:str, variants:tuple[str,...])->str:
        last=self._last_identity.get(key)
        choices=[x for x in variants if x!=last] or list(variants)
        answer=random.SystemRandom().choice(choices)
        self._last_identity[key]=answer
        return answer

    def render(self, prompt: str, response: CoreResponse) -> str:
        tr=_is_turkish(prompt)

        if response.results:
            item=response.results[0]
            return ("Evet." if item.answer else "Hayır.") if tr else ("Yes." if item.answer else "No.")

        if response.operation_results:
            item=response.operation_results[0]
            kind=item.get("kind")
            if not item.get("ok",False):
                msg=item.get("message") or item.get("error") or "unknown error"
                return (f"İşlem başarısız: {msg}" if tr else f"Operation failed: {msg}")

            if kind=="IDENTITY":
                if tr:
                    variants=(
                        "Ben FLM'im. İstekleri anlayıp Semantic Core üzerinden akıl yürüten bir yapay zekâ asistanıyım.",
                        "Adım FLM. Tek bir dil arayüzü modeliyle çalışan, asıl muhakemeyi Semantic Core'da yapan bir yapay zekâ asistanıyım.",
                        "Ben FLM adlı yapay zekâ asistanıyım; dili arayüz modeli işler, planlama ve doğrulama gibi işleri ise çekirdeğim yürütür.",
                        "FLM'im. Amacım soruları anlamak, çekirdek araçlarla sonuç üretmek ve bunu sana doğal biçimde aktarmak.",
                    )
                else:
                    variants=(
                        "I'm FLM, an AI assistant that uses a Semantic Core for reasoning and a single trainable interface model for language.",
                        "My name is FLM. I'm an AI assistant whose reasoning is handled by a Semantic Core while one interface model handles language.",
                        "I'm FLM: an AI assistant designed to understand requests, reason through a Semantic Core, and return useful answers.",
                        "I'm FLM. I use one language interface model, while planning, verification, and other reasoning live in my Core.",
                    )
                return self._choose_nonrepeating("tr" if tr else "en",variants)

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

            if kind=="VERIFY":
                return ("Tüm kontroller geçti." if tr else "All checks passed.")

        if response.memory_facts or response.memory_rules:
            return "Bilgi kaydedildi." if tr else "Knowledge recorded."

        return "Sonuç yok." if tr else "No result."
