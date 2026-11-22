"""Translation for the manga-translator pipeline.

Free, no-key translators (bubble text is short, so these are fine):
  1. MyMemory (api.mymemory.translated.net) — free, no key, rate-limited.
  2. Google Translate via deep-translator (unofficial web endpoint) — fallback.
No API keys needed. If both fail, the original text is kept and marked.
"""
import requests
from urllib.parse import quote

MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def _via_mymemory(text, src, dest):
    r = requests.get(
        MYMEMORY_URL,
        params={"q": text, "langpair": f"{src}|{dest}"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    out = (data.get("responseData") or {}).get("translatedText", "")
    if not out or out.strip().upper() == "QUERY LENGTH LIMIT EXCEEDED":
        raise ValueError("MyMemory: empty or limit response")
    return out.strip()


def _via_google(text, src, dest):
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=src, target=dest).translate(text).strip()


def translate(text, src="ja", dest="en"):
    """Translate one bubble of text. Returns (translated, engine_used)."""
    text = (text or "").strip()
    if not text:
        return "", "none"
    if src == dest:
        return text, "none"
    for engine, fn in (("mymemory", _via_mymemory), ("google", _via_google)):
        try:
            out = fn(text, src, dest)
            if out:
                return out, engine
        except Exception:
            continue
    return text, "failed"
