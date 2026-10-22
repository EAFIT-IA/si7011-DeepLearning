"""Fix S06-F04 (generated image), two errors of the generator:

1. The formula over head 1 read softmax(QK^T / d_k): the square root was missing. It is
   redrawn as softmax(QK^T / sqrt(d_k)), the formula of slide 11.
2. "concatenate" was drawn as three horizontal bands, which reads as stacking rows (3T x d_k).
   Heads are concatenated along the features, T x (H d_k): the block is redrawn as three
   vertical strips, labelled 1, 2, H.

Usage: python fix_s06_f04_multihead.py s06_f04_multihead_original.png s06_f04_multihead.png
"""
import io
import sys

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape
NAVY = (20, 33, 61)

# ---- 1. concatenate block: outer border x 1260-1262 / 1389-1391, y 309-310 / 570-571
x0, x1, y0, y1 = 1263, 1389, 311, 570          # inside the border (x1, y1 exclusive)
fills = [a[350, 1325].copy(), a[440, 1325].copy(), a[530, 1325].copy()]
border = a[440, 1261].copy()
w = (x1 - x0) / 3
for y in range(y0, y1):
    for x in range(x0, x1):
        near_corner = (min(x - x0, x1 - 1 - x) < 12) and (min(y - y0, y1 - 1 - y) < 12)
        px = a[y, x]
        if px.sum() > 740 or (near_corner and px[1] < 160):
            continue                              # white outside the rounded corner, or the corner border
        a[y, x] = fills[min(int((x - x0) // w), 2)]
for k in (1, 2):
    xd = int(round(x0 + k * w))
    a[y0:y1, xd - 1:xd + 1] = border


def render(tex, size, color='#14213d'):
    mpl.rcParams['mathtext.fontset'] = 'stix'
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=color, family='STIXGeneral')
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')


img = Image.fromarray(a).convert('RGBA')
for lab, k in (('1', 0), ('2', 1), ('$H$', 2)):
    t = render(lab, 17)
    cx = int(x0 + (k + 0.5) * w)
    img.alpha_composite(t, (cx - t.width // 2, y0 + 8))

# ---- 2. formula: erase x 745-990, y 68-113 with the panel background of each row, redraw
a = np.asarray(img.convert('RGB')).copy()
for y in range(68, 114):
    a[y, 745:991] = a[y, 738]
img = Image.fromarray(a).convert('RGBA')
t = render(r'$\mathrm{softmax}\!\left(QK^{\top}/\sqrt{d_k}\right)$', 25)
img.alpha_composite(t, (868 - t.width // 2, 91 - t.height // 2))
img.convert('RGB').save(dst)
print('saved', dst, t.size)
