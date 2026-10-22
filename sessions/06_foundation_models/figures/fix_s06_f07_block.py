"""Fix S06-F07 (generated image), three errors of the generator:

1. The two update formulas came out garbled (stray tilde and cedilla over LN and X). They are
   redrawn: X <- X + MHA(LN(X)) and X <- X + MLP(LN(X)).
2. The post-norm inset put attention and a LayerNorm on the residual path and fed the MLP
   from a side line starting at the bottom. That is not the post-norm block. The inset is
   redrawn: MHA and MLP in side branches, each followed by "+" on the band and then LN
   on the band itself, which is what "LN on the residual path" means.
3. The input was labelled with a lowercase x; the deck writes the token matrix as X.

Usage: python fix_s06_f07_block.py s06_f07_block_original.png s06_f07_block.png
"""
import io
import sys

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape

mpl.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})
INK = '#0d0f6e'          # the navy of the generated labels
BAND = '#2f8ef0'
ORANGE_F, ORANGE_E = '#ffd8a8', '#f07d1a'
TEAL_F, TEAL_E = '#c9f2e6', '#1aa385'
GRAY_F, GRAY_E = '#e3e6ec', '#6b7380'


def render(tex, size, color=INK, weight='normal'):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size, color=color, weight=weight)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')


# ---- 1. formulas: fill the text area with the box colour, redraw centred in the box
boxes = [  # (text bbox to erase x0, x1, y0, y1), (box centre x, y)
    ((925, 1350, 228, 290), (1137, 256)),   # MLP box: x 907-1368, y 213-297
    ((912, 1265, 584, 630), (1089, 606)),   # MHA box: x 903-1275, y 566-645
]
for (x0, x1, y0, y1), _ in boxes:
    for y in range(y0, y1):
        a[y, x0:x1] = a[y, x0 - 6]
# ---- 3. the lowercase x under the band
a[848:892, 412:482] = 255
img = Image.fromarray(a).convert('RGBA')
for tex, (_, (cx, cy)) in zip([r'$X \leftarrow X + \mathrm{MLP}(\mathrm{LN}(X))$',
                                r'$X \leftarrow X + \mathrm{MHA}(\mathrm{LN}(X))$'], boxes):
    t = render(tex, 27)
    img.alpha_composite(t, (cx - t.width // 2, cy - t.height // 2))
t = render(r'$\boldsymbol{X}$', 34)
img.alpha_composite(t, (446 - t.width // 2, 868 - t.height // 2))

# ---- 2. post-norm inset, panel x 1285-1594, y 367-894 (redrawn whole)
px0, py0, pw, ph = 1285, 367, 310, 528
fig = plt.figure(figsize=(pw / 100, ph / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, pw)
ax.set_ylim(ph, 0)
ax.axis('off')
fig.patch.set_alpha(0)
ax.add_patch(FancyBboxPatch((2, 2), pw - 4, ph - 4, boxstyle='round,pad=0,rounding_size=14',
                            facecolor='#f5f6f9', edgecolor='#cfd4dc', lw=1.5))
ax.text(pw / 2, 26, 'post-norm (original)', ha='center', va='center', fontsize=15, color=INK, weight='bold')
bx = 92                                     # band x
y_in, y_out = 452, 62
y_add1, y_ln1, y_add2, y_ln2 = 338, 288, 176, 124
y_br1, y_br2 = 418, 250                     # where each branch leaves the band
seg = [(y_in, y_add1 + 14), (y_add1 - 14, y_ln1 + 17), (y_ln1 - 17, y_add2 + 14),
       (y_add2 - 14, y_ln2 + 17)]
for ya, yb in seg:
    ax.plot([bx, bx], [ya, yb], color=BAND, lw=5, solid_capstyle='butt', zorder=1)
ax.add_patch(FancyArrowPatch((bx, y_ln2 - 17), (bx, y_out), arrowstyle='-|>', mutation_scale=18,
                             color=BAND, lw=5, shrinkA=0, shrinkB=0, zorder=1))


def add(y):
    ax.add_patch(Circle((bx, y), 13, facecolor='white', edgecolor=INK, lw=1.6, zorder=3))
    ax.text(bx, y + 1, '+', ha='center', va='center', fontsize=17, color=INK, zorder=4)


def box(cx, cy, w, h, text, fc, ec, size=12):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=5',
                                facecolor=fc, edgecolor=ec, lw=1.6, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, color=INK, zorder=4, linespacing=0.95)


def branch(y_from, y_to, cy, text, fc, ec):
    xr = 222
    ax.plot([bx, xr], [y_from, y_from], color=GRAY_E, lw=1.6, zorder=2)
    ax.plot([xr, xr], [y_from, cy + 22], color=GRAY_E, lw=1.6, zorder=2)
    box(xr, cy, 110, 42, text, fc, ec)
    ax.plot([xr, xr], [cy - 22, y_to], color=GRAY_E, lw=1.6, zorder=2)
    ax.add_patch(FancyArrowPatch((xr, y_to), (bx + 14, y_to), arrowstyle='-|>', mutation_scale=13,
                                 color=GRAY_E, lw=1.6, shrinkA=0, shrinkB=0, zorder=2))
    ax.add_patch(Circle((bx, y_from), 3.2, color=GRAY_E, zorder=2))


branch(y_br1, y_add1, 382, 'multi-head\nattention', ORANGE_F, ORANGE_E)
branch(y_br2, y_add2, 213, 'MLP', TEAL_F, TEAL_E)
add(y_add1)
add(y_add2)
box(bx, y_ln1, 64, 30, 'LN', GRAY_F, GRAY_E, size=14)
box(bx, y_ln2, 64, 30, 'LN', GRAY_F, GRAY_E, size=14)
ax.text(bx, 478, r'$\boldsymbol{X}$', ha='center', va='center', fontsize=22, color=INK)
ax.text(pw / 2, 508, 'LN on the residual path', ha='center', va='center', fontsize=13.5, color=INK)
buf = io.BytesIO()
fig.savefig(buf, dpi=100, transparent=True)
plt.close(fig)
buf.seek(0)
inset = Image.open(buf).convert('RGBA')
assert inset.size == (pw, ph), inset.size
# clear the old inset (white page background around it) and place the new one
base = np.asarray(img).copy()
base[py0 - 2:py0 + ph + 2, px0 - 2:px0 + pw + 2, :3] = 255
img = Image.fromarray(base)
img.alpha_composite(inset, (px0, py0))
img.convert('RGB').save(dst)
print('saved', dst)
