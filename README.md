# Manga Translator — trained agent

Translate a manga page: detect speech bubbles → OCR the Japanese →
translate with a **fine-tuned model** → whiten bubbles → typeset English back in.

## How it works

```
page.png
  → detect.py        (OpenCV bubble detection + Tesseract OCR, jpn)
  → agent_translate.py (fine-tuned MarianMT ja->en — the trained agent)
  → typeset.py       (whiten bubble interiors, fit translated text)
  → page-en.png + page-en.json
```

## The trained agent

- **Base:** Helsinki-NLP/opus-mt-ja-en (MarianMT, 74M params)
- **Fine-tune data:** 10,000 short conversational JA–EN pairs from OPUS-100
  (movie subtitles — short dialogue, like manga speech), 1 epoch, lr 2e-5
- **Trained on:** CPU in the dev VM (no GPU), ~2.5 h
- **Model dir:** `model-manga-ja-en/` (weights + tokenizer)
- **Eval:** `python3 eval.py` — sacreBLEU on 200 held-out pairs, base vs tuned
  (numbers filled in after the first training run)

## Usage

```bash
# whole page (needs the venv: cv2 + torch live there)
./venv/bin/python pipeline.py page.png --src ja --dest en
# -> page-en.png + page-en.json (bubble, bbox, original, translated)

# just the translator agent
./venv/bin/python agent_translate.py "こんにちは！"

# just detection + OCR
./venv/bin/python detect.py page.png
```

Non ja→en pairs fall back to free API translators (`translate.py`,
MyMemory → Google, no keys).

## Files

| file | what |
|---|---|
| `detect.py` | bubble detection + OCR (built by the second agent) |
| `agent_translate.py` | the trained agent (loads `model-manga-ja-en/`) |
| `translate.py` | free-API fallback translators |
| `typeset.py` | whiten bubbles + fit translated text |
| `pipeline.py` | full page CLI |
| `train.py` / `prep_data.py` | fine-tuning + data prep |
| `eval.py` | base vs tuned BLEU + test sentences |
| `data/` | train.jsonl (20k) + val.jsonl (500) |
| `test-page.png`, `test-page-neo.png` | synthetic test pages (self-drawn) |

## Limits (honest)

- OCR is Tesseract: clean printed text reads well (conf 85–95 on tests);
  hand-lettered or stylised text, vertical text, and text touching the
  bubble outline read worse. A bubble whose text touches the outline can
  lose edge strokes.
- The fine-tune adapts a general translator toward short dialogue; gains
  over the base model are modest — the base was already trained on OPUS
  subtitle data. Real manga-specific data would improve it further.
- Only for manga you own or have the right to translate, for personal reading.

## Running on a weak PC (his old laptop)

The trained model is small (74M, ~300 MB). Translating one page needs
Python + the venv + Tesseract — no GPU. Expect ~1–2 min per page on a slow
machine (OCR + translation are CPU-bound, batch one page at a time).
Training itself needs a stronger machine or a free Colab GPU — do not
retrain on the old laptop.
