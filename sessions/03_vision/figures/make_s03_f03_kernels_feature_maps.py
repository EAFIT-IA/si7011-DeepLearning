# Generates s03_f03_kernels_feature_maps.png: one image, four hand-made 3×3 kernels and their feature maps.
# Usage: python make_s03_f03_kernels_feature_maps.py out.png
# Image: skimage.data.camera() (no known copyright restrictions). Same cross-correlation as nn.Conv2d (no flip).
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import correlate
from skimage import data, transform

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100

DIVERGE = LinearSegmentedColormap.from_list('bo', [BLUE, 'white', ORANGE])
x = transform.resize(data.camera().astype(float) / 255.0, (128, 128), anti_aliasing=True)

KERNELS = [
    ('vertical edges',   np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float), 'signed'),
    ('horizontal edges', np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float), 'signed'),
    ('blur',             np.full((3, 3), 1 / 9), 'plain'),
    ('sharpen',          np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], float), 'plain'),
]


def fmt(v):
    if abs(v - 1 / 9) < 1e-9:
        return r'$\frac{1}{9}$'
    return f'${int(v)}$' if v == int(v) else f'${v:.2f}$'


fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

# ---- input image (left)
fig.patches.append(FancyBboxPatch((0.025, 0.2), 0.215, 0.7, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#f4f6f9', ec='#c9d3df', lw=1.2, zorder=-5))
ax = fig.add_axes([0.1325 - 0.0825, 0.245, 0.165, 0.165 * W / H])
ax.imshow(x, cmap='gray', vmin=0, vmax=1); ax.axis('off')
fig.text(0.1325, 0.84, r'input $x$', ha='center', fontsize=22, color=NAVY)
fig.text(0.1325, 0.57, r'$1 \times 128 \times 128$', ha='center', fontsize=17, color=GRAY)

# ---- four kernels and their maps
x0, cw, gap = 0.265, 0.172, 0.0105
for j, (name, k, kind) in enumerate(KERNELS):
    cx = x0 + j * (cw + gap)
    fill = ['#e8f1fa', '#e8f1fa', '#e9f5ee', '#fdf0e6'][j]
    fig.patches.append(FancyBboxPatch((cx, 0.2), cw, 0.7, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc=fill, ec='#c9d3df', lw=1.2, zorder=-5))
    fig.text(cx + cw / 2, 0.865, name, ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
    # kernel as a 3×3 table of numbers
    ka = fig.add_axes([cx + cw / 2 - 0.05, 0.665, 0.1, 0.1 * W / H * 0.95])
    ka.set_xlim(0, 3); ka.set_ylim(0, 3); ka.invert_yaxis(); ka.set_aspect('equal')
    ka.set_xticks([]); ka.set_yticks([])
    for s in ka.spines.values():
        s.set_color(NAVY); s.set_linewidth(1.2)
    for i in range(3):
        for jj in range(3):
            v = k[i, jj]
            col = ORANGE if v > 0 and kind == 'signed' else (BLUE if v < 0 else NAVY)
            ka.add_patch(plt.Rectangle((jj, i), 1, 1, fc='white', ec='#c9d3df', lw=0.8))
            ka.text(jj + 0.5, i + 0.5, fmt(v), ha='center', va='center', fontsize=14, color=col)
    fig.text(cx + cw / 2, 0.635, r'$\downarrow\ \ y = k \star x$', ha='center', va='center', fontsize=16, color=GRAY)
    # feature map
    y = correlate(x, k, mode='nearest')
    ma = fig.add_axes([cx + 0.008, 0.245, cw - 0.016, (cw - 0.016) * W / H])
    if kind == 'signed':
        m = np.percentile(np.abs(y), 97)
        ma.imshow(np.clip(y, -m, m), cmap=DIVERGE, vmin=-m, vmax=m)
    else:
        ma.imshow(np.clip(y, 0, 1), cmap='gray', vmin=0, vmax=1)
    ma.axis('off')

fig.text(x0 + cw + gap / 2, 0.172, 'orange: positive response · blue: negative', ha='center', fontsize=12.5,
         color=GRAY)
fig.patches.append(FancyBboxPatch((0.025, 0.045), 0.95, 0.105, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.0975, 'Same image, same sliding operation: the kernel decides which pattern lights up. '
         'In a CNN the kernels are learned.', ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
fig.savefig(sys.argv[1], dpi=DPI, facecolor='white')
