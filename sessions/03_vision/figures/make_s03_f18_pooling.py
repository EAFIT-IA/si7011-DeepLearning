# Generates s03_f18_pooling.png: 2×2 max and average pooling (stride 2) on a 4×4 feature map,
# and on the same map shifted one pixel to the right. All outputs are computed here.
# Usage: python make_s03_f18_pooling.py out.png
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
WIN = ['#e8f1fa', '#e9f5ee', '#fdf0e6', '#f1eef8']             # one tint per 2×2 window

# each window has one strong response in an even column, so a 1-pixel shift keeps it inside its window
A = np.array([[1, 0, 7, 1],
              [9, 1, 0, 0],
              [0, 0, 1, 0],
              [6, 1, 8, 2]], float)
A_shift = np.zeros_like(A); A_shift[:, 1:] = A[:, :-1]         # one pixel to the right, zero enters on the left
pool = lambda a, f: np.array([[f(a[2 * i:2 * i + 2, 2 * j:2 * j + 2]) for j in range(2)] for i in range(2)])
PEAKS = {9, 7, 6, 8}


def grid(ax, a, cell, x0, y0, windows, fmt='{:g}', fs=20):
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            fc = WIN[(i // 2) * 2 + (j // 2)] if windows else WIN[i * 2 + j]
            ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell, fc=fc, ec='#b5c0cf', lw=1))
            peak = windows and a[i, j] in PEAKS
            ax.text(x0 + (j + .5) * cell, y0 + (i + .5) * cell, fmt.format(a[i, j]), ha='center', va='center',
                    fontsize=fs, color=ORANGE if peak else NAVY, weight='bold' if peak else 'normal')
    if windows:
        for t in (0, 2):
            for u in (0, 2):
                ax.add_patch(Rectangle((x0 + u * cell, y0 + t * cell), 2 * cell, 2 * cell, fc='none', ec=NAVY, lw=2.2))


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16.72); ax.set_ylim(9.41, 0); ax.axis('off')
ci, co = 0.62, 0.78
outs = {}
for h, (a, name) in enumerate([(A, 'feature map'), (A_shift, 'the same map, shifted 1 px right')]):
    cx = 4.18 + 8.36 * h                                       # center of this half
    ax.add_patch(FancyBboxPatch((cx - 3.9, 0.55), 7.8, 6.95, boxstyle='round,pad=0,rounding_size=0.15',
                                fc='#f4f6f9', ec='#c9d3df', lw=1.2, zorder=-5))
    ax.text(cx, 0.95, name, ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
    grid(ax, a, ci, cx - 2 * ci, 1.35, windows=True)
    for k, (f, label, fmt) in enumerate([(np.max, 'max pooling', '{:g}'), (np.mean, 'average pooling', '{:.2f}')]):
        ox = cx - 2.0 + 2.4 * k                                # left edge of the 2×2 output
        out = pool(a, f); outs[(h, k)] = out
        ax.add_artist(FancyArrowPatch((cx + (k - 0.5) * 0.9, 1.35 + 4 * ci + 0.1), (ox + co, 4.55),
                                      arrowstyle='->', mutation_scale=20, color=GRAY, lw=1.8))
        ax.text(ox + co, 4.85, label, ha='center', va='center', fontsize=15, color=TEAL, weight='bold')
        grid(ax, out, co, ox, 5.1, windows=False, fmt=fmt, fs=18)

same_max = np.array_equal(outs[(0, 0)], outs[(1, 0)])
changed_avg = int((outs[(0, 1)] != outs[(1, 1)]).sum())
assert same_max and changed_avg == 3
ax.text(8.36, 7.85, f'max pooling: identical after the shift  ·  average pooling: {changed_avg} of 4 values change',
        ha='center', va='center', fontsize=15, color=ORANGE)

fig.patches.append(FancyBboxPatch((0.0325, 0.025), 0.935, 0.09, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.07, 'Pooling halves the resolution. A response that moves inside its window does not move the max.',
         ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
fig.savefig(sys.argv[1], dpi=DPI, facecolor='white')
print('max', outs[(0, 0)].tolist(), '→', outs[(1, 0)].tolist(), '· avg', outs[(0, 1)].tolist(), '→', outs[(1, 1)].tolist())
