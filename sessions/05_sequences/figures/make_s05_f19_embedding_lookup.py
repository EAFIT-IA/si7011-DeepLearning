"""S05-F19 · An embedding is a row lookup — reproducible 16:9 matplotlib figure (drawn, no data).

Run: python make_s05_f19_embedding_lookup.py
Produces s05_f19_embedding_lookup.png (1672x941) in this folder.
Top: one-hot(w) times E selects row w of E. Bottom: a sentence of ids gathers rows of the same E into a
(T, d_e) matrix, and only those rows receive gradient in that step. Values of E are random (seed 5).
"""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import matplotlib as mpl

W, H = 16.72, 9.41
NAVY = '#14213d'
BLUE = '#1f6fb4'
BLUE_L = '#dceefd'
TEAL = '#167d78'
GRAY = '#7b8795'
GRAY_L = '#f1f3f6'
ORANGE = '#d9631e'
ORANGE_L = '#fbe3d3'
GREEN = '#229d65'
PINK = '#da2e90'
INK = '#253447'

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'mathtext.fontset': 'dejavusans',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
fig, ax = plt.subplots(figsize=(W, H), dpi=100)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

VOCAB = ['[PAD]', '[UNK]', 'the', 'good', 'not', 'fun', 'very', 'but']
D = 4
rng = np.random.default_rng(5)
E = np.round(rng.normal(0, 0.6, (len(VOCAB), D)), 1)
E[0] = 0.0                                              # padding_idx: the [PAD] row stays at zero


def label(x, y, s, size=15, color=NAVY, ha='center', va='center', weight='normal', **kw):
    ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=size, fontweight=weight, zorder=10, **kw)


def arrow(a, b, color=INK, lw=1.5, ms=13, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2, zorder=7))


def matriz(x0, y_top, vals, cw=0.62, ch=0.42, fills=None, size=12.5, ec='#9aa5b4'):
    """Draws a matrix with its top-left corner at (x0, y_top); returns (x0, y_top, width, height)."""
    n, m = vals.shape
    for i in range(n):
        for j in range(m):
            fc = fills[i] if fills is not None and fills[i] else 'white'
            ax.add_patch(Rectangle((x0 + j * cw, y_top - (i + 1) * ch), cw, ch, facecolor=fc, edgecolor=ec, lw=1))
            v = vals[i, j]
            label(x0 + (j + 0.5) * cw, y_top - (i + 0.5) * ch, f'{v:+.1f}' if v else '0', size=size, color=NAVY)
    return x0, y_top, m * cw, n * ch


# ------------------------------------------------------------------ top: one-hot times E
label(0.45, 8.95, '1 · A one-hot vector times $E$ picks one row', size=19, weight='bold', ha='left', color=BLUE)
w = VOCAB.index('good')
onehot = np.zeros((1, len(VOCAB))); onehot[0, w] = 1
cw = 0.5
ox, oy = 0.7, 7.55
for j, v in enumerate(onehot[0]):
    fc = ORANGE_L if v else 'white'
    ax.add_patch(Rectangle((ox + j * cw, oy - 0.42), cw, 0.42, facecolor=fc, edgecolor='#9aa5b4', lw=1))
    label(ox + (j + 0.5) * cw, oy - 0.21, f'{int(v)}', size=13, color=ORANGE if v else GRAY,
          weight='bold' if v else 'normal')
    label(ox + (j + 0.5) * cw, oy + 0.32, VOCAB[j], size=10.5, color=GRAY, rotation=55, ha='left', va='bottom')
label(ox + 4 * cw, oy - 0.85, 'one-hot("good"), length $|V|$', size=13, color=GRAY)
label(ox + 8 * cw + 0.38, oy - 0.21, '×', size=26)

ex, ey = 6.35, 8.55
fills = [ORANGE_L if i == w else None for i in range(len(VOCAB))]
_, _, ew, eh = matriz(ex, ey, E, fills=fills, ch=0.38)
for i, t in enumerate(VOCAB):
    label(ex - 0.12, ey - (i + 0.5) * 0.38, t, size=12, color=ORANGE if i == w else GRAY, ha='right',
          weight='bold' if i == w else 'normal')
label(ex + ew / 2, ey - eh - 0.32, '$E \\in \\mathbb{R}^{|V| \\times d_e}$, learned', size=14)
label(ex + ew + 0.45, oy - 0.21, '=', size=26)
rx = ex + ew + 0.9
matriz(rx, oy, E[w:w + 1], fills=[ORANGE_L])
label(rx + D * 0.62 / 2, oy - 0.85, '$\\mathbf{e}_{good} = E[\\,3\\,]$, length $d_e$', size=14, color=ORANGE)
label(rx, 6.45, 'The product never runs: nn.Embedding reads\nrow $w$ directly, without $|V|$ multiplications.',
      size=13.5, ha='left', va='top', color=INK, linespacing=1.4)
label(rx, 5.55, 'Notebook 5.4: $|V| = 20\\,002$, $d_e = 64$: 1.28 M values,\nalmost all the parameters of the bag of embeddings.',
      size=13.5, ha='left', va='top', color=GRAY, linespacing=1.4)

ax.plot([0.45, W - 0.45], [4.75, 4.75], color='#d5dbe3', lw=1.2)

# ------------------------------------------------------------------ bottom: a sentence gathers rows
label(0.45, 4.35, '2 · A sentence of ids gathers rows of the same $E$', size=19, weight='bold', ha='left', color=TEAL)
frase = ['not', 'good', 'but', 'very', 'fun']
cols = [BLUE, ORANGE, GREEN, PINK, TEAL]
light = ['#dceefd', ORANGE_L, '#d6f1e3', '#f8dcef', '#daf3ef']
ids = [VOCAB.index(t) for t in frase]
tx = 0.75
for k, (t, i, c) in enumerate(zip(frase, ids, cols)):
    y = 3.55 - k * 0.6
    ax.add_patch(FancyBboxPatch((tx, y - 0.22), 1.25, 0.44, boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor=light[k], edgecolor=c, lw=1.4))
    label(tx + 0.62, y, t, size=14, color=c, weight='bold')
    label(tx + 1.75, y, f'→ {i}', size=14, color=INK)
label(tx + 1.0, 0.55, 'tokens → ids', size=13, color=GRAY)

ex2, ey2 = 6.35, 3.8
fills2 = [light[ids.index(i)] if i in ids else None for i in range(len(VOCAB))]
_, _, ew2, eh2 = matriz(ex2, ey2, E, fills=fills2, ch=0.38, size=12)
for i, t in enumerate(VOCAB):
    c = cols[ids.index(i)] if i in ids else GRAY
    label(ex2 - 0.12, ey2 - (i + 0.5) * 0.38, f'{i}  {t}', size=11.5, color=c, ha='right',
          weight='bold' if i in ids else 'normal')
label(ex2 + ew2 / 2, ey2 - eh2 - 0.3, '$E$  (the same matrix)', size=13)
for k, i in enumerate(ids):
    arrow((tx + 2.35, 3.55 - k * 0.6), (ex2 - 1.35, ey2 - (i + 0.5) * 0.38), color=cols[k], lw=1.2, ms=10)

mx, my = 10.75, 3.55 + 0.21
matriz(mx, my, E[ids], fills=light, ch=0.6 - 0.0, size=13)
for k, t in enumerate(frase):
    label(mx - 0.15, my - (k + 0.5) * 0.6, t, size=13, color=cols[k], ha='right', weight='bold')
    arrow((ex2 + ew2 + 0.08, ey2 - (ids[k] + 0.5) * 0.38), (mx - 1.15, my - (k + 0.5) * 0.6), color=cols[k],
          lw=1.0, ms=9, ls=(0, (3, 2)))
label(mx + D * 0.62 / 2, my - 5 * 0.6 - 0.35, '$E[\\mathrm{ids}]$: shape $(T, d_e) = (5, 4)$', size=14)
label(13.6, 3.6, 'In a batch: $(B, T)$ ids\n→ $(B, T, d_e)$, the input\nof the bag, the CNN\nand the LSTM.',
      size=13.5, ha='left', va='top', color=INK, linespacing=1.4)
label(13.6, 1.95, 'Backward: only these 5 rows\nreceive gradient in this step.\nRare words: rare updates.',
      size=13.5, ha='left', va='top', color=TEAL, linespacing=1.4)

fig.savefig(Path(__file__).with_name('s05_f19_embedding_lookup.png'), dpi=100)
print('wrote s05_f19_embedding_lookup.png')
