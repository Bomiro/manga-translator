import os
"""Draw a HARDER synthetic manga test page (no copyrighted material).

Three panels, eight speech bubbles: varied sizes, one small bubble, one
long sentence (stresses word-wrap in typesetting), bubbles near panel
edges. Used to stress-test the pipeline end-to-end (EN + AR).
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 900, 1900
CJK = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"

img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

# Panels
d.rectangle([40, 40, 860, 640], outline="black", width=5)
d.rectangle([40, 680, 860, 1280], outline="black", width=5)
d.rectangle([40, 1320, 860, 1860], outline="black", width=5)

# Simple "character" silhouettes
d.ellipse([150, 350, 300, 500], fill="#dddddd", outline="black", width=3)
d.rectangle([170, 500, 280, 600], fill="#dddddd", outline="black", width=3)
d.ellipse([600, 950, 750, 1100], fill="#dddddd", outline="black", width=3)
d.rectangle([620, 1100, 730, 1240], fill="#dddddd", outline="black", width=3)
d.ellipse([120, 1550, 260, 1690], fill="#dddddd", outline="black", width=3)
d.rectangle([150, 1690, 240, 1840], fill="#dddddd", outline="black", width=3)


def bubble(cx, cy, rw, rh, text, size=44):
    d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill="white",
              outline="black", width=4)
    font = ImageFont.truetype(CJK, size, index=1)
    tw = d.textlength(text, font=font)
    d.text((cx - tw / 2, cy - size / 2), text, font=font, fill="black")


# Panel 1
bubble(620, 200, 190, 90, "こんにちは！")
bubble(250, 480, 170, 80, "元気ですか？")
# Panel 2 — medium-long + small bubble (font sized so the text fits
# INSIDE the bubble: text wider than the bubble touches the outline and
# OCR misreads the edge glyphs)
bubble(600, 850, 210, 90, "あの箱を開けてはいけない。", size=30)
bubble(280, 1100, 210, 85, "まさか、そんなはずはない。", size=30)
bubble(700, 1180, 115, 55, "早く逃げろ！", size=32)
# Panel 3 — long sentence (wrap test) + two normal bubbles
bubble(450, 1480, 270, 90, "もう遅いよ、早く行こう。", size=32)
bubble(250, 1720, 180, 85, "行こう！")
bubble(650, 1760, 190, 85, "今日はいい天気だ。", size=40)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "test-page-complex.png")
img.save(out)
print("saved test-page-complex.png")
