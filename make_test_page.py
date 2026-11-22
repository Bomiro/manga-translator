import os
"""Draw a synthetic manga test page (no copyrighted material).

Two panels, four speech bubbles with Japanese text, using Noto Serif CJK JP.
Used to test the pipeline end-to-end.
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 1300
CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

# Panels
d.rectangle([40, 40, 860, 620], outline="black", width=5)
d.rectangle([40, 680, 860, 1260], outline="black", width=5)

# Simple "character" silhouettes (circles + bodies, just shapes)
d.ellipse([150, 350, 300, 500], fill="#dddddd", outline="black", width=3)
d.rectangle([170, 500, 280, 600], fill="#dddddd", outline="black", width=3)
d.ellipse([600, 950, 750, 1100], fill="#dddddd", outline="black", width=3)
d.rectangle([620, 1100, 730, 1240], fill="#dddddd", outline="black", width=3)


def bubble(cx, cy, rw, rh, text, size=44):
    d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill="white",
              outline="black", width=4)
    font = ImageFont.truetype(CJK, size, index=1)
    tw = d.textlength(text, font=font)
    d.text((cx - tw / 2, cy - size / 2), text, font=font, fill="black")


bubble(620, 200, 190, 90, "こんにちは！")
bubble(250, 480, 170, 80, "元気ですか？")
bubble(300, 850, 200, 90, "今日はいい天気だ。")
bubble(650, 1150, 180, 85, "行こう！")

img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-page-neo.png"))
print("saved test-page-neo.png")
