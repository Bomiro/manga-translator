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
- **Trained on:** free Colab GPU (Tesla T4), 2 min for 313 steps — CPU
  training on the dev VM is NOT viable (dies silently, out of memory).
- **Model dir:** `model-manga-ja-en/` (weights + tokenizer)
- **Eval:** `python3 eval.py` — sacreBLEU on 200 held-out pairs, base vs tuned
  - Colab T4 run (2026-11-22, 10k pairs, 1 epoch, 313 steps, train_loss 3.888,
    eval_loss 1.764): **BASE 19.54 / TUNED 19.90** — a small gain (+0.36).
    Test sentences are nearly identical between base and tuned (only
    «まさか、そんなはずはない。» differs: base "No way. That can't be true."
    vs tuned "No, it can't be."). Honest verdict: the fine-tune helped a
    little; the base model was already strong on this subtitle-style data.
  - **End-to-end (verified on this VM 2026-11-22):** `pipeline.py` on
    `test-page-neo.png` — 4/4 bubbles, OCR conf 79.6–94.2, translations
    correct, output `test-page-neo-en.png` QA'd visually.
  NOTE: tesseract is an apt package — it VANISHED when the VM was replaced
  (2026-11-22) and had to be reinstalled (`sudo apt-get install -y
  tesseract-ocr tesseract-ocr-jpn tesseract-ocr-eng`). System packages do
  not survive VM replacement.
  **Durable tesseract (2026-11-22 ~17:20): .debs extracted to ~/tesseract/
  (binary + libs + tessdata); detect.py prefers it and sets
  TESSDATA_PREFIX (must point AT the tessdata dir) + LD_LIBRARY_PATH.
  The home copy ALSO lost a dep after another replacement (libgif.so.7) —
  extracted libgif7's .deb into ~/tesseract too; ldd shows 0 missing now.**
- **Complex test page (2026-11-22 ~17:45):** `make_test_page_complex.py`
  draws 3 panels / 8 bubbles (small bubble, long sentences, tight fits).
  EN + AR pipelines: **8/8 correct**. Typesetting fits text to the
  ELLIPSE, not the rectangle (`_ellipse_line_width` in typeset.py) —
  before the fix, multi-line text crossed the outline at the top/bottom
  lines. Lesson: test-page text must fit INSIDE its bubble, or edge
  glyphs get eaten by the outline on the OCR side too.

## Arabic (pivot chain, added 2026-11-22)

No direct JA→AR training — instead a chain, his pick (option 2, better than
direct training because EN→AR is a high-resource pair with a strong model):
`agent_translate_ar.py` — ja→en with the trained model, unload it, then
en→ar with Helsinki opus-mt-en-ar (`model-en-ar/`, downloaded from HF on
first use). One model in RAM at a time (VM memory is tight).
`pipeline.py page.png --dest ar` — typesets with Noto Sans Arabic + PIL
RAQM (letters join, RTL, no reshaper needed).
Verified end-to-end 2026-11-22 on `test-page-neo.png` → `test-page-neo-ar.png`:
مرحباً! / كيف حالك؟ / دعونا نذهب! / إنه يوم لطيف (4/4, OCR 79.6–94.2).
Quirk fixed: the en→ar model mirrors a final '!' to the front
('! مرحباً!') — `_clean_ar()` drops the duplicate.

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
