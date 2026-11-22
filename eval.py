"""Evaluate base vs fine-tuned model on held-out manga-style pairs.

Usage: python3 eval.py [--n 200]
Prints sacreBLEU for both models + side-by-side test sentences.
"""
import argparse
import json
import os

import sacrebleu
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "Helsinki-NLP/opus-mt-ja-en"
TUNED = os.path.join(HERE, "model-manga-ja-en")
DATA = os.path.join(HERE, "data", "val.jsonl")

TESTS = ["こんにちは！", "元気ですか？", "今日はいい天気だ。",
         "行こう！", "あの箱を開けてはいけない。",
         "まさか、そんなはずはない。", "早く逃げろ！"]


def load(path):
    tok = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path)
    model.eval()
    return tok, model


def batch_translate(tok, model, texts):
    outs = []
    for t in texts:
        ids = tok([t], return_tensors="pt", truncation=True, max_length=64)
        out = model.generate(**ids, max_length=64)
        outs.append(tok.decode(out[0], skip_special_tokens=True).strip())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    with open(DATA, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f][:args.n]
    srcs = [r["ja"] for r in rows]
    refs = [[r["en"] for r in rows]]

    for name, path in (("BASE", BASE), ("TUNED", TUNED)):
        tok, model = load(path)
        hyps = batch_translate(tok, model, srcs)
        bleu = sacrebleu.corpus_bleu(hyps, refs).score
        print(f"{name} BLEU (n={len(rows)}): {bleu:.2f}")
        if name == "TUNED":
            tuned_hyps = hyps
        else:
            base_hyps = hyps

    print("\n--- test sentences ---")
    tok_b, model_b = load(BASE)
    tok_t, model_t = load(TUNED)
    for s in TESTS:
        b = batch_translate(tok_b, model_b, [s])[0]
        t = batch_translate(tok_t, model_t, [s])[0]
        print(f"JA: {s}\n  base:  {b}\n  tuned: {t}")


if __name__ == "__main__":
    main()
