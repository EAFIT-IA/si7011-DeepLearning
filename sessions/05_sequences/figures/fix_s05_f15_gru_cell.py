"""Patch the generated GRU cell (S05-F15): rewire the reset gate.

The generator drew the reset product backwards: the circle "reset" received h_{t-1} and the bracket
[h_{t-1}, x_t], and its result went INTO r_t; the candidate bracket [r_t ⊙ h_{t-1}, x_t] never received it.
This script redraws:
  1. r_t → the reset circle (the arrow now points from r_t into the circle);
  2. h_{t-1} → the bracket [h_{t-1}, x_t] by a branch that bypasses the circle;
  3. the reset circle → the candidate bracket [r_t ⊙ h_{t-1}, x_t], with gaps where it crosses
     the two arrows that feed r_t and u_t (redrawn as straight verticals).

    python fix_s05_f15_gru_cell.py   (reads s05_f15_gru_cell_generated.png, writes s05_f15_gru_cell.png)
"""
from PIL import Image, ImageDraw

SRC, OUT = 's05_f15_gru_cell_generated.png', 's05_f15_gru_cell.png'
BG = (220, 241, 254)          # cell fill
INK = (70, 82, 108)           # arrow color of the generated figure
LW = 3

im = Image.open(SRC).convert('RGB')
d = ImageDraw.Draw(im)


def erase(x0, y0, x1, y1):
    d.rectangle([x0, y0, x1, y1], fill=BG)


def line(pts):
    d.line(pts, fill=INK, width=LW, joint='curve')


def head(x, y, direction, L=14, Wd=8):
    """Filled arrowhead with its tip at (x, y)."""
    if direction == 'right':
        pts = [(x, y), (x - L, y - Wd), (x - L, y + Wd)]
    elif direction == 'left':
        pts = [(x, y), (x + L, y - Wd), (x + L, y + Wd)]
    elif direction == 'down':
        pts = [(x, y), (x - Wd, y - L), (x + Wd, y - L)]
    else:
        pts = [(x, y), (x - Wd, y + L), (x + Wd, y + L)]
    d.polygon(pts, fill=INK)


CX, CY, R = 391, 368, 27      # reset circle
RT_TOP, RT_X = 411, 512       # top edge and center of r_t
BR_L, BR_Y = 537, 640         # left edge of [h_{t-1}, x_t] and the height of the old feed
CAND_L, CAND_Y = 958, 564     # left edge and center of [r_t ⊙ h_{t-1}, x_t]

# remove the old wiring: circle → r_t elbow, bracket → circle vertical and its horizontal
erase(CX + R + 1, CY - 8, RT_X + 14, RT_TOP - 1)
erase(CX - 13, CY + R + 1, CX + 13, BR_Y + 9)
erase(CX - 13, BR_Y - 9, BR_L - 2, BR_Y + 9)

# 1. r_t → reset circle
line([(RT_X, RT_TOP - 1), (RT_X, CY), (CX + R + 14, CY)])
head(CX + R + 1, CY, 'left')

# 2. h_{t-1} → [h_{t-1}, x_t], branching off the vertical above the circle
YB, XB = 290, 350
line([(CX, YB), (XB, YB), (XB, BR_Y), (BR_L - 14, BR_Y)])
head(BR_L - 1, BR_Y, 'right')
d.ellipse([CX - 5, YB - 5, CX + 5, YB + 5], fill=INK)

# 3a. the feeds [h_{t-1}, x_t] → r_t and → u_t, redrawn as straight verticals to make room for 3b
erase(500, 557, 600, 616)
erase(685, 557, 780, 616)
for x in (560, 728):
    line([(x, 617), (x, 571)])
    head(x, 556, 'up')

# 3b. reset circle → candidate bracket, with gaps where it crosses the two feeds
Y3, X3 = 590, 935
gaps = [560, 728]
xs = [CX] + [g + s for g in gaps for s in (-9, 9)] + [X3]
line([(CX, CY + R + 1), (CX, Y3)])
for a, b in zip(xs[::2], xs[1::2]):
    line([(a, Y3), (b, Y3)])
line([(X3, Y3), (X3, CAND_Y), (CAND_L - 14, CAND_Y)])
head(CAND_L - 1, CAND_Y, 'right')

im.save(OUT)
print('saved', OUT)
