"""Fix S06-F11 (generated image): panel 4 ("Predict") was drawn upside down. The encoder sat
below the tokens with its arrows going up into them, so it read as if the encoder produced
"me [MASK] en el banco", and the context arrows left the encoder instead of the neighbouring
tokens. Panels 1-3 and the header of panel 4 are kept; the body of panel 4 is redrawn
bottom to top, as in F07 and F08: the masked sentence goes into the encoder; inside it,
attention brings context from both sides to the [MASK] position; from that position comes
the prediction "senté ?", the only place where the loss is computed.

Usage: python fix_s06_f11_bert_masking.py s06_f11_bert_masking_original.png s06_f11_bert_masking.png
"""
import io
import sys

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape

# body of panel 4: inside the border x 1198-1657, below the header (y >= 202), above y 654
BG = a[400, 1210].copy()
a[203:652, 1201:1655] = BG

mpl.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
NAVY = '#14213d'
CHIP_F, CHIP_E = '#eaf4fd', '#8fc0ec'
MASK_F, MASK_E = '#e4e6ec', '#8a94a3'
TEAL_F, TEAL_E = '#cdf3ea', '#1a9d86'
ENC_F, ENC_E = '#bcdcfb', '#1f6fe0'
ORANGE = '#ec7d2c'
GRAY = '#6b7380'

fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1672)
ax.set_ylim(941, 0)
ax.axis('off')
fig.patch.set_alpha(0)


def chip(cx, cy, w, h, text, fc, ec, size=19, weight='normal'):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=7',
                                facecolor=fc, edgecolor=ec, lw=1.8, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, color=NAVY, weight=weight, zorder=4)


def arrow(p0, p1, color=GRAY, lw=2.0, rad=0.0, mut=15):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=mut, lw=lw, color=color,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=5))


words = ['me', '[MASK]', 'en', 'el', 'banco']
xs = [1250, 1349, 1449, 1522, 1604]
ws = [60, 108, 60, 60, 80]
y_chip, ch = 604, 54
xm = xs[1]

# 1. input row: the sentence after replacement, going up into the encoder
for x, w_, t in zip(xs, ws, words):
    if t == '[MASK]':
        chip(x, y_chip, w_, ch, t, MASK_F, MASK_E)
    else:
        chip(x, y_chip, w_, ch, t, CHIP_F, CHIP_E)
    arrow((x, y_chip - ch / 2), (x, 536))
ax.text(1428, 646, 'input after step 3', ha='center', va='center', fontsize=15, color=GRAY, style='italic')

# 2. encoder: title, and attention bringing context from both sides to the [MASK] column
ax.add_patch(FancyBboxPatch((1215, 378), 428, 158, boxstyle='round,pad=0,rounding_size=10',
                            facecolor=ENC_F, edgecolor=ENC_E, lw=2.4, zorder=2))
ax.text(1429, 404, 'encoder', ha='center', va='center', fontsize=25, color=NAVY, weight='bold', zorder=4)
ax.text(1429, 434, '(no mask: reads both sides)', ha='center', va='center', fontsize=17, color=NAVY, zorder=4)
yb = 515
for x in xs:
    if x != xm:
        ax.plot([x], [yb], 'o', color=ORANGE, ms=5, zorder=5)
ax.plot([xm], [yb - 2], 's', color=MASK_E, ms=7, zorder=5)
arrow((xs[0], yb), (xm - 8, yb - 6), color=ORANGE, lw=2.6, rad=-0.5, mut=16)
for x in xs[2:]:
    arrow((x, yb), (xm + 8, yb - 6), color=ORANGE, lw=2.0, rad=0.22, mut=14)
ax.text(1282, 461, 'left context', ha='center', va='center', fontsize=15, color=ORANGE, zorder=6)
ax.text(1560, 461, 'right context', ha='center', va='center', fontsize=15, color=ORANGE, zorder=6)

# 3. output at the [MASK] position only, and the loss
arrow((xm, 378), (xm, 330), color=TEAL_E, lw=2.6, mut=17)
chip(xm, 302, 118, 54, 'senté ?', TEAL_F, TEAL_E, size=20)
ax.text(1500, 302, 'prediction at the\n[MASK] position', ha='center', va='center', fontsize=15,
        color=GRAY, linespacing=0.95)
ax.text(1429, 232, 'loss only on the chosen token', ha='center', va='center', fontsize=18, color=NAVY)
arrow((xm, 250), (xm, 273), color=NAVY, lw=2.2, mut=15)

buf = io.BytesIO()
fig.savefig(buf, dpi=100, transparent=True)
plt.close(fig)
buf.seek(0)
over = Image.open(buf).convert('RGBA')
img = Image.fromarray(a).convert('RGBA')
img.alpha_composite(over)
img.convert('RGB').save(dst)
print('saved', dst)
