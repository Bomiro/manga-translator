import os
#!/usr/bin/env python3
"""Draw MY synthetic test page for the detector (no real manga).

Page 1000x1400: two black panels, four white ellipse bubbles with black
outlines (one with a tail), Japanese text in Noto Serif CJK. Expected
reading order (right-to-left columns, top-to-bottom): B1, B2, B3, B4.
Also a tiny white speck inside a panel (noise the detector must ignore).
"""
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-page.png")

W, H = 1000, 1400
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT, 40)

# Two black panels.
d.rectangle([60, 60, 940, 640], fill="black")
d.rectangle([60, 720, 940, 1340], fill="black")

# (bbox, text, tail?) — B1/B2 right column, B3/B4 left column.
BUBBLES = [
    ((620, 120, 900, 300), "こんにちは", False),   # B1 right-top
    ((620, 780, 900, 960), "ありがとう！", False),  # B2 right-bottom
    ((120, 200, 420, 380), "元気ですか。", True),   # B3 left-top, with tail
    ((120, 860, 420, 1040), "すごいね。", False),  # B4 left-bottom
]

for (x0, y0, x1, y1), text, tail in BUBBLES:
    if tail:  # tail first so the ellipse covers its top edge
        d.polygon([(180, 350), (130, 440), (250, 360)],
                  fill="white", outline="black", width=5)
    d.ellipse([x0, y0, x1, y1], fill="white", outline="black", width=6)
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    d.text((cx - tw // 2 - tb[0], cy - th // 2 - tb[1]),
           text, font=font, fill="black")

# Noise: tiny white speck inside panel 1 (must be ignored by the detector).
d.rectangle([500, 500, 508, 508], fill="white")

img.save(OUT)
print("wrote", OUT)
