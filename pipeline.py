"""Manga translator pipeline — detect bubbles, OCR, translate, typeset.

Usage:
  python3 pipeline.py page.png [--src ja] [--dest en] [--out page-en.png]
Output: translated page + page-en.json (bubbles with original/translated text).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect import detect_bubbles, ocr_bubble  # noqa: E402
# Primary translator: the TRAINED manga agent (fine-tuned ja->en model).
# API translators (translate.py) remain only as fallback for other pairs.
from agent_translate import translate  # noqa: E402
from typeset import whiten_and_typeset  # noqa: E402


def run(image_path, src="ja", dest="en", out_path=None):
    out_path = out_path or os.path.splitext(image_path)[0] + f"-{dest}.png"
    json_path = os.path.splitext(out_path)[0] + ".json"

    bubbles = detect_bubbles(image_path)
    results = []
    translations = []
    for i, b in enumerate(bubbles):
        ocr = ocr_bubble(image_path, b["bbox"])
        original = (ocr.get("text") or "").strip()
        trans, engine = translate(original, src=src, dest=dest) if original else ("", "none")
        translations.append(trans)
        results.append({
            "bubble": i + 1,
            "bbox": [int(v) for v in b["bbox"]],
            "original": original,
            "ocr_conf": round(float(ocr.get("conf", 0)), 1),
            "translated": trans,
            "engine": engine,
        })
        print(f"[{i+1}] {original!r} -> {trans!r} ({engine})")

    whiten_and_typeset(image_path, bubbles, translations, out_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path} + {json_path} ({len(bubbles)} bubbles)")
    return out_path, json_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Translate a manga page.")
    ap.add_argument("image", help="input manga page (png/jpg)")
    ap.add_argument("--src", default="ja", help="source language (default: ja)")
    ap.add_argument("--dest", default="en", help="target language (default: en)")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()
    run(args.image, src=args.src, dest=args.dest, out_path=args.out)
