# Generates s03_f20_one_by_one_conv.png: a 1×1 convolution as the same dense layer W (C_out × C_in) applied to the
# channel vector at every position. Oblique projection: the front face is the H × W grid, depth is the channel axis.
# Usage: python make_s03_f20_one_by_one_conv.py out.png
# Redraw of the generated version, whose highlighted "position" was a column of pixels on the front face.
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
N = 8                                   # H = W = 8
C_IN, C_OUT = 64, 16
I, J = 2, 4                             # the highlighted position (row, column)
OTHERS = [(0, 0), (7, 7)]
DX, DY = 0.62, -0.42                    # oblique direction of the channel axis, per depth unit
CELL = 0.36


def slab(ax, x0, y0, depth, face, edge, label_c, c_val):
    """Front face: N × N grid with top-left corner (x0, y0); depth units along (DX, DY)."""
    s = N * CELL
    ox, oy = DX * depth, DY * depth
    top = [(x0, y0), (x0 + s, y0), (x0 + s + ox, y0 + oy), (x0 + ox, y0 + oy)]
    side = [(x0 + s, y0), (x0 + s, y0 + s), (x0 + s + ox, y0 + s + oy), (x0 + s + ox, y0 + oy)]
    ax.add_patch(Polygon(top, fc=face, ec=edge, lw=1.4, alpha=0.55, zorder=1))
    ax.add_patch(Polygon(side, fc=face, ec=edge, lw=1.4, alpha=0.75, zorder=1))
    for k in range(1, 6):               # channel slices, drawn as lines on the top and side faces
        t = depth * k / 6
        ax.plot([x0 + DX * t, x0 + s + DX * t], [y0 + DY * t, y0 + DY * t], color=edge, lw=0.6, alpha=0.6, zorder=1)
        ax.plot([x0 + s + DX * t, x0 + s + DX * t], [y0 + DY * t, y0 + s + DY * t], color=edge, lw=0.6, alpha=0.6, zorder=1)
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((x0 + j * CELL, y0 + i * CELL), CELL, CELL, fc=face, ec=edge, lw=0.7, alpha=0.5, zorder=2))
    ax.annotate('', xy=(x0 + ox - 0.2, y0 + oy - 0.1), xytext=(x0 - 0.2, y0 - 0.1),       # along the left depth edge
                arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.6), zorder=6)
    ax.text(x0 + ox / 2 - 0.35, y0 + oy / 2 - 0.35, label_c, fontsize=17, color=NAVY, ha='right', va='center')
    return s


def rod(ax, x0, y0, depth, i, j, color, alpha=1.0, dashed=False):
    """A 1 × 1 position going through every channel: front square, top and side faces along the depth."""
    cx, cy = x0 + j * CELL, y0 + i * CELL
    ox, oy = DX * depth, DY * depth
    kw = dict(fc=color, ec=NAVY if not dashed else GRAY, lw=1.1, alpha=alpha, zorder=4, ls='--' if dashed else '-')
    ax.add_patch(Polygon([(cx, cy), (cx + CELL, cy), (cx + CELL + ox, cy + oy), (cx + ox, cy + oy)], **kw))
    ax.add_patch(Polygon([(cx + CELL, cy), (cx + CELL, cy + CELL), (cx + CELL + ox, cy + CELL + oy), (cx + CELL + ox, cy + oy)],
                         **{**kw, 'alpha': alpha * 0.8}))
    ax.add_patch(Rectangle((cx, cy), CELL, CELL, **kw))
    return cx + CELL / 2, cy + CELL / 2


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 16.72); ax.set_ylim(9.41, 0); ax.set_aspect('equal'); ax.axis('off')

# input slab (deeper: 64 channels) and output slab (16 channels)
xi, yi, di = 2.3, 2.55, 3.2
xo, yo, do = 11.4, 2.55, 0.8
s = slab(ax, xi, yi, di, '#dbe8f6', BLUE, r'$C_\mathrm{in} = 64$', C_IN)
slab(ax, xo, yo, do, '#dcf0ea', TEAL, r'$C_\mathrm{out} = 16$', C_OUT)
for (oi, oj) in OTHERS:
    rod(ax, xi, yi, di, oi, oj, '#c9d3df', alpha=0.9, dashed=True)
    rod(ax, xo, yo, do, oi, oj, '#c9d3df', alpha=0.9, dashed=True)
pin = rod(ax, xi, yi, di, I, J, ORANGE)
pout = rod(ax, xo, yo, do, I, J, ORANGE)

ax.text(xi + s / 2, yi + s + 0.4, r'$H \times W = 8 \times 8$', ha='center', fontsize=14, color=GRAY)
ax.text(xi + s / 2, yi + s + 0.95, r'$a_l:\ 64 \times 8 \times 8$', ha='center', fontsize=20, color=NAVY)
ax.text(xo + s / 2, yo + s + 0.4, r'$H \times W = 8 \times 8$', ha='center', fontsize=14, color=GRAY)
ax.text(xo + s / 2, yo + s + 0.95, r'$a_{l+1}:\ 16 \times 8 \times 8$', ha='center', fontsize=20, color=NAVY)
ax.text(xi - 0.15, yi + (I + 0.5) * CELL, r'$x_{\colon,i,j}$' + '\n64 numbers', ha='right', va='center', fontsize=15, color=ORANGE)
ax.text(xo + s + DX * do + 0.25, yo + (I + 0.5) * CELL + 0.55, r'$y_{\colon,i,j}$' + '\n16 numbers', ha='left', va='center',
        fontsize=15, color=ORANGE)

# the matrix
mx, my, mw, mh = 7.75, 3.05, 2.7, 1.5
ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle='round,pad=0,rounding_size=0.15', fc='#f1f2f6', ec=NAVY, lw=1.5, zorder=3))
ax.text(mx + mw / 2, my + 0.55, r'$W \in \mathbb{R}^{16 \times 64}$', ha='center', va='center', fontsize=21, color=NAVY, zorder=4)
ax.text(mx + mw / 2, my + 1.1, r'$+\ b \in \mathbb{R}^{16}$', ha='center', va='center', fontsize=17, color=NAVY, zorder=4)
ax.text(mx + mw / 2, my - 0.55, r'$y_{\colon,i,j} = W\,x_{\colon,i,j} + b$', ha='center', va='center', fontsize=22, color=NAVY)
far_in = (pin[0] + DX * di, pin[1] + DY * di)
ax.add_artist(FancyArrowPatch(far_in, (mx - 0.05, my + mh / 2 - 0.1), arrowstyle='->', mutation_scale=22, color=ORANGE, lw=2.2, zorder=5))
ax.add_artist(FancyArrowPatch((mx + mw + 0.05, my + mh / 2 - 0.1), (pout[0] - 0.12, pout[1] + 0.05), arrowstyle='->',
                              mutation_scale=22, color=ORANGE, lw=2.2, zorder=5))
for (oi, oj), rad in zip(OTHERS, (-0.25, 0.25)):     # the same W for the other positions
    a = (xi + (oj + 0.5) * CELL + DX * di, yi + (oi + 0.5) * CELL + DY * di)
    b = (xo + (oj + 0.5) * CELL, yo + (oi + 0.5) * CELL)
    ax.add_artist(FancyArrowPatch(a, (mx + mw * 0.3, my + (0 if oi == 0 else mh)), arrowstyle='->', mutation_scale=16,
                                  color=GRAY, lw=1.3, ls='--', connectionstyle=f'arc3,rad={rad}', zorder=2))
    ax.add_artist(FancyArrowPatch((mx + mw * 0.7, my + (0 if oi == 0 else mh)), b, arrowstyle='->', mutation_scale=16,
                                  color=GRAY, lw=1.3, ls='--', connectionstyle=f'arc3,rad={rad}', zorder=2))
ax.text(mx + mw / 2, my + mh + 1.15, r'the same $W$ at all $8 \times 8 = 64$ positions', ha='center', fontsize=15, color=GRAY)

# parameters and banner
ax.add_patch(FancyBboxPatch((4.9, 7.0), 6.9, 0.7, boxstyle='round,pad=0,rounding_size=0.1', fc='#f4f6f9', ec='#c9d3df', lw=1.2))
ax.text(8.35, 7.35, r'parameters: $16 \cdot 64 + 16 = 1040$, whatever $H$ and $W$', ha='center', va='center', fontsize=17, color=NAVY)
fig.patches.append(FancyBboxPatch((0.0325, 0.025), 0.935, 0.09, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.07, 'A 1 × 1 convolution is a small dense layer applied to every pixel: it mixes channels, not positions.',
         ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
assert C_OUT * C_IN + C_OUT == 1040
fig.savefig(sys.argv[1], dpi=DPI, facecolor='white')
