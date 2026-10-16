# Generates s03_f16_padding_stride.png: the same 6×6 input and 3×3 kernel with (p, s) = (0, 1), (1, 1), (1, 2).
# Usage: python make_s03_f16_padding_stride.py out.png
# Each panel draws the first two kernel positions (solid, dashed) and the output cells they produce;
# output sizes come from H_out = floor((H + 2p - K) / s) + 1, the formula on the slide.
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
N, K = 6, 3
CASES = [(0, 1, 'no padding'), (1, 1, 'padding 1: size kept'), (1, 2, 'padding 1, stride 2: size halved')]

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
pw, gap, x0 = 0.3, 0.0175, 0.0325
for c, (p, s, title) in enumerate(CASES):
    out = (N + 2 * p - K) // s + 1
    left = x0 + c * (pw + gap)
    fig.patches.append(FancyBboxPatch((left, 0.19), pw, 0.73, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc='#f4f6f9', ec='#c9d3df', lw=1.2, zorder=-5))
    fig.text(left + pw / 2, 0.885, title, ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
    fig.text(left + pw / 2, 0.835, rf'$K = 3,\ \ p = {p},\ \ s = {s}$', ha='center', va='center', fontsize=17, color=GRAY)

    # one data unit = one pixel; input (with padding) on top, output below
    ax = fig.add_axes([left + 0.01, 0.235, pw - 0.02, 0.565])
    ax.set_xlim(-0.5, 9.5); ax.set_ylim(-1.2, 17.2); ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
    side = N + 2 * p
    ox = (10 - side) / 2 - 0.5
    for i in range(side):
        for j in range(side):
            pad = not (p <= i < p + N and p <= j < p + N)
            ax.add_patch(Rectangle((ox + j, i), 1, 1, fc='white' if pad else '#dfe7f1', ec='#b5c0cf', lw=0.8,
                                   ls=(0, (2, 2)) if pad else '-'))
            if pad:
                ax.text(ox + j + 0.5, i + 0.5, '0', ha='center', va='center', fontsize=9, color=GRAY)
    ax.text(ox + side + 0.2, side / 2, f'{N}×{N}' + (f'\n+ border {p}' if p else ''), va='center', fontsize=13, color=NAVY)
    # first two kernel positions: solid orange, then shifted by s (dashed blue)
    ax.add_patch(Rectangle((ox, 0), K, K, fc=ORANGE, alpha=0.18, ec=ORANGE, lw=3))
    ax.add_patch(Rectangle((ox + s, 0), K, K, fc='none', ec=BLUE, lw=3, ls=(0, (4, 2))))
    if s > 1:
        ax.annotate('', xy=(ox + s + 0.1, -0.3), xytext=(ox + 0.1, -0.3),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.8))
        ax.text(ox + s / 2 + 0.1, -0.5, f'{s} px', ha='center', va='bottom', fontsize=11, color=BLUE)

    # output grid
    oy = side + 2.2
    ax.add_artist(FancyArrowPatch((4.5, side + 0.3), (4.5, oy - 0.3), arrowstyle='->', mutation_scale=18, color=GRAY, lw=1.6))
    oo = (10 - out) / 2 - 0.5
    for i in range(out):
        for j in range(out):
            fc = ORANGE if (i, j) == (0, 0) else (BLUE if (i, j) == (0, 1) else '#e9f5ee')
            ax.add_patch(Rectangle((oo + j, oy + i), 1, 1, fc=fc, alpha=0.85 if (i == 0 and j < 2) else 1,
                                   ec='#9fb9ad', lw=0.8))
    ax.text(oo + out + 0.2, oy + out / 2, f'{out}×{out}', va='center', fontsize=15, color=TEAL, weight='bold')
    fig.text(left + pw / 2, 0.215,
             rf'$H_{{out}} = \lfloor ({N} + 2\cdot{p} - 3)\,/\,{s} \rfloor + 1 = {out}$',
             ha='center', va='center', fontsize=16, color=NAVY)

fig.patches.append(FancyBboxPatch((0.0325, 0.045), 0.935, 0.105, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.0975, 'Padding lets the kernel sit on the edge pixels. Stride decides how far it moves: '
         'stride 2 halves the map.', ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
fig.savefig(sys.argv[1], dpi=DPI, facecolor='white')
