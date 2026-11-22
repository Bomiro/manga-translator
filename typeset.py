"""Whiten speech bubbles and typeset the translation back in.

Steps per bubble:
  1. Paint the bubble interior white (mask eroded by a couple of px so the
     black outline survives).
  2. Fit the translated text: largest font size that fits the bubble box
     (word-wrapped, with padding), drawn centred in black.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _fit_font(draw, text, box_w, box_h, font_path, max_size=48, min_size=10):
    """Largest font size where the wrapped text fits inside (box_w, box_h)."""
    for size in range(max_size, min_size - 1, -2):
        font = ImageFont.truetype(font_path, size)
        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if draw.textlength(trial, font=font) <= box_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        line_h = int(size * 1.25)
        if len(lines) * line_h <= box_h:
            return font, lines, line_h
    font = ImageFont.truetype(font_path, min_size)
    return font, [text], int(min_size * 1.25)


def whiten_and_typeset(image_path, bubbles, translations, out_path,
                       font_path=DEFAULT_FONT):
    """bubbles: list of dicts with bbox (x,y,w,h) and optional 'mask' (np array).
    translations: list of strings, same order. Returns out_path."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for b, text in zip(bubbles, translations):
        x, y, w, h = [int(v) for v in b["bbox"]]
        if not text:
            continue
        # 1. Whiten interior (eroded mask keeps the outline).
        mask = b.get("mask")
        if mask is not None:
            kernel = np.ones((5, 5), np.uint8)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            white = np.ones((H, W, 3), dtype=np.uint8) * 255
            arr = np.array(img)
            arr[eroded > 0] = 255
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
        else:
            draw.ellipse([x + 2, y + 2, x + w - 2, y + h - 2], fill="white")
        # 2. Fit + draw translated text, centred.
        pad = max(6, int(min(w, h) * 0.08))
        font, lines, line_h = _fit_font(
            draw, text, w - 2 * pad, h - 2 * pad, font_path
        )
        total_h = len(lines) * line_h
        cy = y + (h - total_h) // 2
        for i, line in enumerate(lines):
            lw = draw.textlength(line, font=font)
            draw.text((x + (w - lw) / 2, cy + i * line_h),
                      line, font=font, fill="black")
    img.save(out_path)
    return out_path
