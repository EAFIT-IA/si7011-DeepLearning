# Generates s03_f17_equivariance.png: conv(shift(x)) = shift(conv(x)), drawn as a commuting square, plus the difference.
# Usage: python make_s03_f17_equivariance.py out.png
# Same image and vertical-edge kernel as S03-F03 (skimage camera, 128×128, cross-correlation, zero padding 1).
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import correlate
from skimage import data, transform

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
DIVERGE = LinearSegmentedColormap.from_list('bo', [BLUE, 'white', ORANGE])

x = transform.resize(data.camera().astype(float) / 255.0, (128, 128), anti_aliasing=True)
k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
D = 24                                                         # shift: 24 pixels to the right, zeros enter on the left


def shift(a):
    out = np.zeros_like(a); out[:, D:] = a[:, :-D]; return out


conv = lambda a: correlate(a, k, mode='constant', cval=0.0)   # padding 1 with zeros, as nn.Conv2d(padding=1)
y, y_s = conv(x), conv(shift(x))
diff = np.abs(y_s - shift(y))
inner = diff[1:-1, D + 1:-1]                                   # away from every border (image and shifted-in zeros)
assert inner.max() < 1e-12, inner.max()
m = np.percentile(np.abs(y), 97)

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
size = 0.155                                                   # panel width in figure units (square panels)
hh = size * W / H
cols, rows = [0.075, 0.37], [0.585, 0.175]
panels = [((0, 0), x, 'gray', r'$x$'), ((0, 1), y, 'div', r'$\mathrm{conv}(x)$'),
          ((1, 0), shift(x), 'gray', r'$\mathrm{shift}(x)$'), ((1, 1), y_s, 'div', r'$\mathrm{conv}(\mathrm{shift}(x))$')]
for (r, c), img, kind, label in panels:
    ax = fig.add_axes([cols[c], rows[r], size, hh])
    if kind == 'gray':
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='auto')
    else:
        ax.imshow(np.clip(img, -m, m), cmap=DIVERGE, vmin=-m, vmax=m, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('#c9d3df')
    ax.axvline(70 + (D if r else 0), color=TEAL, lw=1.6, ls=(0, (4, 3)))   # the same column, before and after the shift
    fig.text(cols[c] + size / 2, rows[r] + hh + 0.012, label, ha='center', fontsize=18, color=NAVY)

arrow = dict(arrowstyle='->', mutation_scale=22, color=GRAY, lw=2)
for r in range(2):
    yc = rows[r] + hh * 0.43
    fig.add_artist(FancyArrowPatch((cols[0] + size + 0.012, yc), (cols[1] - 0.012, yc), transform=fig.transFigure, **arrow))
    fig.text((cols[0] + size + cols[1]) / 2, yc + 0.02, 'conv', ha='center', fontsize=15, color=GRAY)
for c in range(2):
    xc = cols[c] + size / 2
    fig.add_artist(FancyArrowPatch((xc, rows[0] - 0.008), (xc, rows[1] + hh + 0.045), transform=fig.transFigure, **arrow))
    fig.text(xc + 0.01, rows[0] - 0.05, f'shift {D} px →', fontsize=14, color=GRAY, va='center')

# right: the difference
dl = 0.665
ax = fig.add_axes([dl, rows[1], size, hh])
ax.imshow(diff, cmap='Greys', vmin=0, vmax=m, aspect='auto'); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_color('#c9d3df')
fig.text(dl + size / 2, rows[1] + hh + 0.012, r'$|\mathrm{conv}(\mathrm{shift}(x)) - \mathrm{shift}(\mathrm{conv}(x))|$',
         ha='center', fontsize=14.5, color=NAVY)
fig.text(dl, 0.85, 'Shift then convolve,\nor convolve then shift:\nthe same feature map.', fontsize=17, color=NAVY,
         weight='bold', va='top')
fig.text(dl, 0.70, f'Difference: exactly 0, except in\nthe columns next to the borders,\nwhere padding and the {D} new\nzero columns enter.',
         fontsize=13.5, color=GRAY, va='top')
fig.text(dl, 0.56, 'dashed line: the same image column,\nbefore and after the shift', fontsize=12, color=TEAL, va='top')

fig.patches.append(FancyBboxPatch((0.0325, 0.025), 0.935, 0.09, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.07, 'A pattern learned in one place is detected everywhere: the layer moves with the image.',
         ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
fig.savefig(sys.argv[1], dpi=DPI, facecolor='white')
