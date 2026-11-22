#!/usr/bin/env python3
"""Manga speech-bubble detection + OCR (module 1 of manga-translator).

Pipeline: page image -> detect_bubbles() -> ocr_bubble() per bubble.
Bubbles are white rounded shapes with a black outline, so detection works on
the near-white regions of the page: each bubble interior is its own connected
white component, separated from the page background by its black outline.

No network calls, no API keys. OCR uses the local `tesseract` binary
(subprocess), lang `jpn` first, falling back to `jpn+eng` on low confidence.

Usage:
    ~/workspace/manga-translator/venv/bin/python detect.py page.png
        # prints JSON list of bubbles (OpenCV lives in the project venv)
"""

import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

# Tesseract lives in ~/tesseract (extracted .debs) because apt installs do
# NOT survive VM replacement. Prefer it; fall back to the system binary.
_HOME_TESS = os.path.expanduser("~/tesseract/usr/bin/tesseract")


def _tesseract_bin():
    if os.path.isfile(_HOME_TESS):
        return _HOME_TESS
    return shutil.which("tesseract")


def _tesseract_env():
    env = dict(os.environ)
    if os.path.isfile(_HOME_TESS):
        home = os.path.expanduser("~/tesseract")
        env["TESSDATA_PREFIX"] = home + "/usr/share/tesseract-ocr/5/tessdata"
        env["LD_LIBRARY_PATH"] = (home + "/usr/lib/x86_64-linux-gnu:"
                                  + env.get("LD_LIBRARY_PATH", ""))
    return env

# --- detection tuning -------------------------------------------------------
WHITE_THRESH = 240      # gray value above this counts as "white" (bubble
                        # interiors are pure 255; light-grey art stays below)
MIN_AREA_FRAC = 0.0015  # bubble must be at least this fraction of page area
MAX_AREA_FRAC = 0.25    # ... and at most this (kills the page background)
MIN_SOLIDITY = 0.75     # contour area / convex-hull area (bubbles are convex-ish)
MIN_ASPECT = 0.2        # w/h bounds; bubbles are round/oval, not thin slivers
MAX_ASPECT = 5.0
RING_DARK_FRAC = 0.35   # fraction of the ring around a bubble that must be dark
                        # (the black outline) for it to count as a real bubble
OCR_PAD = 6             # px of padding added around the bbox before OCR
OCR_MIN_CONF = 50.0     # below this, retry OCR with jpn+eng


def _ring_is_dark(gray, mask):
    """Check the bubble is surrounded by a dark outline.

    Dilate the bubble mask a few px and look at the ring just outside it:
    a real bubble has its black outline there, so most ring pixels are dark.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=3)
    ring = cv2.subtract(dilated, mask)
    ring_pixels = gray[ring > 0]
    if ring_pixels.size == 0:
        return False
    return float(np.mean(ring_pixels < 150)) >= RING_DARK_FRAC


def detect_bubbles(image_path):
    """Find speech bubbles on a manga page.

    Returns a list of dicts, sorted in manga reading order (right-to-left
    columns, top-to-bottom within a column). Each dict:
        {"bbox": (x, y, w, h),
         "contour": [(x, y), ...],          # simplified outline points
         "mask": np.ndarray (uint8, 0/255, same size as the page)}
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    img_h, img_w = gray.shape
    img_area = img_h * img_w

    # Near-white regions: bubble interiors + page background (+ panel whites).
    _, white = cv2.threshold(gray, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    # NOTE: RETR_LIST, not RETR_EXTERNAL. Bubbles sit inside black panels,
    # which are holes in the page-background white region, so in the contour
    # hierarchy each bubble is a child of a panel hole — RETR_EXTERNAL would
    # return only the page border and miss every bubble.
    contours, _ = cv2.findContours(white, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_FRAC * img_area or area > MAX_AREA_FRAC * img_area:
            continue  # tiny speck, or the whole page background
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= 1 or y <= 1 or x + w >= img_w - 1 or y + h >= img_h - 1:
            continue  # touches the page edge -> page background, not a bubble
        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue
        hull = cv2.convexHull(cnt)
        solidity = area / max(cv2.contourArea(hull), 1.0)
        if solidity < MIN_SOLIDITY:
            continue  # ragged / non-convex shape, not a bubble

        # Interior mask (full page size, 255 inside the bubble).
        mask = np.zeros((img_h, img_w), np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if not _ring_is_dark(gray, mask):
            continue  # no black outline around it -> not a bubble

        # Simplified outline, for the caller (and JSON output).
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        contour = [(int(p[0][0]), int(p[0][1])) for p in approx]
        bubbles.append({"bbox": (x, y, w, h), "contour": contour, "mask": mask})

    return _reading_order(bubbles)


def _reading_order(bubbles):
    """Sort bubbles in manga reading order: right-to-left columns, then
    top-to-bottom inside each column. Columns are clustered by bubble-center
    x, with a band of half the median bubble width."""
    if not bubbles:
        return bubbles
    widths = sorted(b["bbox"][2] for b in bubbles)
    band = 0.5 * widths[len(widths) // 2]

    # Rightmost first.
    items = sorted(bubbles,
                   key=lambda b: -(b["bbox"][0] + b["bbox"][2] / 2.0))
    columns = []  # list of (column_center_x, [bubbles])
    for b in items:
        cx = b["bbox"][0] + b["bbox"][2] / 2.0
        placed = False
        for col_cx, col in columns:
            if abs(cx - col_cx) < band:
                col.append(b)
                placed = True
                break
        if not placed:
            columns.append((cx, [b]))
    ordered = []
    for _, col in columns:  # columns already right-to-left
        ordered.extend(sorted(col, key=lambda b: b["bbox"][1]))
    return ordered


# --- OCR --------------------------------------------------------------------
def _tesseract_tsv(png_bytes, lang):
    """Run tesseract on PNG bytes, return (text, mean_word_conf)."""
    proc = subprocess.run(
        [_tesseract_bin(), "stdin", "stdout", "--psm", "6", "-l", lang, "tsv"],
        input=png_bytes, capture_output=True, timeout=60,
        env=_tesseract_env())
    text_parts, confs = [], []
    for line in proc.stdout.decode("utf-8", errors="ignore").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        try:
            conf = float(fields[10])
        except ValueError:
            continue
        word = fields[11].strip()
        if word:
            text_parts.append(word)
            if conf >= 0:
                confs.append(conf)
    # Japanese has no word spaces; English does.
    sep = "" if lang.startswith("jpn") else " "
    text = sep.join(text_parts).strip()
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    return text, mean_conf


def ocr_bubble(image, bbox):
    """OCR the text inside one bubble.

    `image` is a grayscale/BGR numpy array or a path. `bbox` is (x, y, w, h).
    Returns {"text": str, "conf": float}. Tries `jpn` first, falls back to
    `jpn+eng` when confidence is low or nothing was read. Empty text when
    tesseract is missing or the bubble holds no readable text.
    """
    if _tesseract_bin() is None:
        return {"text": "", "conf": 0.0}
    if isinstance(image, str):
        gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        gray = image if image.ndim == 2 else cv2.cvtColor(image,
                                                          cv2.COLOR_BGR2GRAY)
    if gray is None:
        return {"text": "", "conf": 0.0}
    x, y, w, h = (int(v) for v in bbox)
    x0, y0 = max(x - OCR_PAD, 0), max(y - OCR_PAD, 0)
    x1, y1 = min(x + w + OCR_PAD, gray.shape[1]), min(y + h + OCR_PAD,
                                                      gray.shape[0])
    crop = gray[y0:y1, x0:x1].copy()  # copy: whitening must not mutate
    if crop.size == 0:                # the caller's image
        return {"text": "", "conf": 0.0}
    # Whiten everything outside the bubble interior: the largest white
    # component at the crop centre. The bbox is rectangular, so its corners
    # hold black panel (and the bubble outline) — tesseract reads those as
    # junk characters. The interior stays white around the text strokes.
    _, bw = cv2.threshold(crop, WHITE_THRESH, 255, cv2.THRESH_BINARY)
    _, labels, _, _ = cv2.connectedComponentsWithStats(bw, 8)
    label = 0
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    for r in range(0, 15):  # centre pixel may sit on a glyph stroke; widen
        win = labels[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        nz = win[win > 0]
        if nz.size:
            vals, counts = np.unique(nz, return_counts=True)
            label = int(vals[np.argmax(counts)])  # interior dominates
            break
    if label:
        # Fill the interior component's outline: that filled shape is the
        # bubble itself (text strokes included). Everything outside it —
        # black panel corners, outline — gets whitened. (Do NOT whiten by
        # label alone: glyph strokes share label 0 with the panel, so a
        # naive "labels != label" would erase the text itself.)
        comp = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
        crop[filled == 0] = 255
    # Upscale + binarize: tesseract reads big clean glyphs far better.
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, png = cv2.imencode(".png", crop)
    text, conf = _tesseract_tsv(png.tobytes(), "jpn")
    if not text or conf < OCR_MIN_CONF:
        text2, conf2 = _tesseract_tsv(png.tobytes(), "jpn+eng")
        if text2 and (not text or conf2 > conf):
            text, conf = text2, conf2
    return {"text": text.strip(), "conf": round(float(conf), 1)}


def main(argv):
    if len(argv) != 2:
        print("usage: python3 detect.py page.png", file=sys.stderr)
        return 2
    page = argv[1]
    bubbles = detect_bubbles(page)
    gray = cv2.imread(page, cv2.IMREAD_GRAYSCALE)
    out = []
    for b in bubbles:
        ocr = ocr_bubble(gray, b["bbox"])
        if not ocr["text"]:
            print(f"skip bubble {b['bbox']}: OCR empty", file=sys.stderr)
            continue
        out.append({"bbox": list(b["bbox"]),
                    "contour": [list(p) for p in b["contour"]],
                    "text": ocr["text"], "conf": ocr["conf"]})
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
