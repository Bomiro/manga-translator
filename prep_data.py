"""Prepare manga-style training data from OPUS-100 (en-ja).

OPUS-100 is movie-subtitle data: short conversational lines, close to manga
speech bubbles. We keep short, clean pairs and split train/validation.
"""
from datasets import load_dataset
import random
import json

random.seed(42)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def clean(ja, en):
    ja, en = (ja or "").strip(), (en or "").strip()
    if not ja or not en:
        return None
    if len(ja) > 60 or len(en.split()) > 25:
        return None
    if "http" in ja or "http" in en:
        return None
    # strip leading "- " speaker dashes common in subtitles
    ja = ja.lstrip("- ").strip()
    en = en.lstrip("- ").strip()
    return {"ja": ja, "en": en}


def main(n_train=20000, n_val=500):
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="train")
    idx = list(range(len(ds)))
    random.shuffle(idx)
    train, val, seen = [], [], set()
    for i in idx:
        t = ds[i]["translation"]
        row = clean(t.get("ja"), t.get("en"))
        if not row:
            continue
        key = (row["ja"], row["en"])
        if key in seen:
            continue
        seen.add(key)
        if len(val) < n_val:
            val.append(row)
        elif len(train) < n_train:
            train.append(row)
        else:
            break
    import os
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train={len(train)} val={len(val)}")
    print("sample:", train[0])


if __name__ == "__main__":
    main()
