"""The trained manga-translation agent.

Loads the fine-tuned model (model-manga-ja-en/) and translates Japanese
bubble text to English. Falls back to the base Helsinki model if the
fine-tuned one is missing.
Usage:
  python3 agent_translate.py "こんにちは！"
  python3 agent_translate.py --file bubbles.json   (reads [{"text": ...}])
"""
import argparse
import json
import os

import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = os.path.join(HERE, "model-manga-ja-en")
BASE = "Helsinki-NLP/opus-mt-ja-en"

_model = None
_tok = None


def _load():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    path = TUNED if os.path.isdir(TUNED) else BASE
    _tok = MarianTokenizer.from_pretrained(path)
    _model = MarianMTModel.from_pretrained(path)
    _model.eval()
    return _model, _tok


def translate(text, src="ja", dest="en"):
    """Translate one bubble. Returns (translated, engine)."""
    text = (text or "").strip()
    if not text:
        return "", "none"
    if src != "ja" or dest != "en":
        # The trained agent is ja->en only; other pairs go to the API module.
        from translate import translate as api_translate
        return api_translate(text, src=src, dest=dest)
    model, tok = _load()
    ids = tok([text], return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        out = model.generate(**ids, max_length=64)
    engine = "trained" if os.path.isdir(TUNED) else "base"
    return tok.decode(out[0], skip_special_tokens=True).strip(), engine


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Translate with the trained agent.")
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            t, eng = translate(it.get("text", ""))
            print(f"{it.get('text', '')!r} -> {t!r} ({eng})")
    elif args.text:
        t, eng = translate(args.text)
        print(f"{t}  [{eng}]")
    else:
        ap.print_help()
