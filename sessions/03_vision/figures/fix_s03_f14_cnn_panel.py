# Redraws the CNN panel of s03_f14_architecture_summary.png on the grid, with exact receptive fields.
# The generated image drew a 4×3 grid labelled "3 × 3" and a 6×6 region labelled "5 × 5".
# Usage: python fix_s03_f14_cnn_panel.py s03_f14_architecture_summary_generated.png s03_f14_architecture_summary.png
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

NAVY, TEAL, ORANGE = '#14213d', '#167d78', '#d9631e'
GRID_FILL, GRID_EDGE = '#e1e3e8', '#3f4653'
LIGHT, DARK = '#9fe3d8', '#2bb3a0'
L1_FILL, L1_EDGE = '#bcdcf7', '#2a64b8'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif'})

# region of the CNN panel to replace (image pixels): inside the panel, between header and caption box
X0, Y0, X1, Y1 = 576, 100, 1094, 672
W, H, DPI = X1 - X0, Y1 - Y0, 100

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
fig.patch.set_facecolor('white')

C = 33                                   # cell size (px)
GX, GY = 86, 258                        # top-left of the 8×8 input grid
cx = lambda j: GX + (j + 0.5) * C        # centre of input column j
cy = lambda i: GY + (i + 0.5) * C

# input grid 8×8
for i in range(8):
    for j in range(8):
        ax.add_patch(Rectangle((GX + j * C, GY + i * C), C, C, fc=GRID_FILL, ec=GRID_EDGE, lw=1.0, zorder=1))
# layer-2 receptive field: 5×5 (rows/cols 1..5), layer-1 unit: 3×3 (rows/cols 2..4), centred on (3, 3)
ax.add_patch(Rectangle((GX + 1 * C, GY + 1 * C), 5 * C, 5 * C, fc=LIGHT, ec=TEAL, lw=1.6, alpha=0.75, zorder=2))
for i in range(1, 6):
    for j in range(1, 6):
        ax.add_patch(Rectangle((GX + j * C, GY + i * C), C, C, fill=False, ec=GRID_EDGE, lw=0.8, zorder=3))
ax.add_patch(Rectangle((GX + 2 * C, GY + 2 * C), 3 * C, 3 * C, fc=DARK, ec=TEAL, lw=2.0, zorder=4))
for i in range(2, 5):
    for j in range(2, 5):
        ax.add_patch(Rectangle((GX + j * C, GY + i * C), C, C, fill=False, ec='#1b5f56', lw=0.9, zorder=5))

# layer-1 3×3 window, centred above input column 3
LX, LY = cx(3) - 1.5 * C, 110
for i in range(3):
    for j in range(3):
        fc = DARK if (i, j) == (1, 1) else L1_FILL
        ax.add_patch(Rectangle((LX + j * C, LY + i * C), C, C, fc=fc, ec=L1_EDGE, lw=1.2, zorder=4))

# layer-2 unit
UX, UY, R = cx(3), 48, 15
ax.add_patch(Circle((UX, UY), R, fc='#f6a96b', ec=ORANGE, lw=1.6, zorder=6))
# fan onto the top edge of the 3×3 window, as in the generated panel
for j in range(4):
    ax.plot([UX, LX + j * C], [UY + R, LY], color=TEAL, lw=1.4, zorder=3)

# one layer-1 unit (centre cell) → its 3×3 input window
for (xa, xb) in [(LX + C, GX + 2 * C), (LX + 2 * C, GX + 5 * C)]:
    ax.plot([xa, xb], [LY + 2 * C, GY + 2 * C], color='#1b5f56', lw=1.6, zorder=6)
# the whole layer-1 window → the 5×5 input window
for (xa, xb) in [(LX, GX + 1 * C), (LX + 3 * C, GX + 6 * C)]:
    ax.plot([xa, xb], [LY + 3 * C, GY + 1 * C], color=TEAL, lw=1.4, ls=(0, (4, 3)), zorder=6)

ax.text(UX - R - 12, UY + 2, 'layer 2 unit', ha='right', va='center', fontsize=16.5, color=NAVY)
ax.text(LX + 3 * C + 18, LY + 1.5 * C, '3 × 3\nin layer 1', ha='left', va='center', fontsize=15.5, color=NAVY,
        linespacing=1.3)
RX = GX + 8 * C + 12
ax.text(RX, GY + 1.6 * C, '5 × 5: seen by\nthe layer 2 unit', ha='left', va='center', fontsize=12.5,
        color=TEAL, linespacing=1.25)
ax.text(RX, GY + 4.4 * C, '3 × 3: seen by\none layer 1 unit', ha='left', va='center', fontsize=12.5,
        color='#1b5f56', linespacing=1.25)
ax.text(GX + 4 * C, GY + 8 * C + 26, '8 × 8 input (64 cells)', ha='center', va='center', fontsize=17.5, color=NAVY)

fig.canvas.draw()
panel = Image.frombuffer('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert('RGB')
img = Image.open(sys.argv[1]).convert('RGB')
img.paste(panel, (X0, Y0))
img.save(sys.argv[2])
