"""Japanese -> Arabic manga translation via a pivot chain.

Step 1: ja -> en with the TRAINED model (agent_translate).
Step 2: en -> ar with Helsinki opus-mt-en-ar (model-en-ar/).
Models are loaded ONE at a time and unloaded between steps: the VM has
little free RAM, and two Marian models at once risk OOM.

Usage:
  python3 agent_translate_ar.py "こんにちは！"
"""
import argparse
import gc
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
EN_AR = os.path.join(HERE, "model-en-ar")
BASE_EN_AR = "Helsinki-NLP/opus-mt-en-ar"

_model = None
_tok = None


def _load_en_ar():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    path = EN_AR if os.path.isdir(EN_AR) else BASE_EN_AR
    _tok = MarianTokenizer.from_pretrained(path)
    _model = MarianMTModel.from_pretrained(path)
    _model.eval()
    return _model, _tok


def unload_en_ar():
    global _model, _tok
    _model = None
    _tok = None
    gc.collect()


def _clean_ar(text):
    """Drop a duplicated leading '!'/'؟' — the en->ar model sometimes
    mirrors the sentence-final punctuation to the front of the string
    ('! مرحباً!'), which renders as a stray mark on the wrong side."""
    text = (text or "").strip()
    if len(text) > 1 and text[0] in "!؟" and text[-1] == text[0]:
        text = text[1:].lstrip()
    return text


def _en_to_ar(en_texts):
    """Translate a list of English strings to Arabic. Returns list of str."""
    model, tok = _load_en_ar()
    outs = []
    for en in en_texts:
        if not en:
            outs.append("")
            continue
        ids = tok([en], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            gen = model.generate(**ids, max_length=64)
        outs.append(_clean_ar(tok.decode(gen[0], skip_special_tokens=True)))
    return outs


def translate_ja_ar(text):
    """One bubble ja -> ar. Returns (arabic, engine)."""
    import agent_translate as ja_en
    en, _ = ja_en.translate(text, src="ja", dest="en")
    if not en:
        return "", "none"
    ja_en.unload()  # free RAM before loading the second model
    ar = _en_to_ar([en])[0]
    return ar, "chain(trained-ja-en + en-ar)"


def translate_many(texts):
    """Batch: all ja->en first, unload, then all en->ar.
    Returns a list of (arabic, engine) in the same order."""
    import agent_translate as ja_en
    ens = []
    for t in texts:
        t = (t or "").strip()
        ens.append(ja_en.translate(t, src="ja", dest="en")[0] if t else "")
    ja_en.unload()
    ars = _en_to_ar(ens)
    return [(a, "chain" if a else "none") for a in ars]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Translate ja -> ar (pivot chain).")
    ap.add_argument("text", nargs="?", default=None)
    args = ap.parse_args()
    if args.text:
        ar, eng = translate_ja_ar(args.text)
        print(f"{ar}  [{eng}]")
    else:
        ap.print_help()
