"""Fix S05-F10 (generated image).

1. Remove the "T = 8" label above the step axis: in the deck T is the window length
   (slide 9), and the window row says "window of T steps" around 5 steps.
2. Reverse the RNN ribbon. Read as "how much each step weighs in y-hat", the ribbon must be
   thin at step 1 and thick at step 8: the gradient dL/dx_t is large near T and fades toward
   step 1 (same shape as F03 and animation A01). The note becomes "recent steps dominate".

Usage: python fix_s05_f10_summary.py s05_f10_summary_original.png s05_f10_summary.png
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape

# 1 · "T = 8", on the white background above the first panel
a[36:74, 1336:1418] = 254

# 2 · RNN row (second panel): boxes measured on the image, 8 boxes, ribbon centred at y = 368
BOXES = [(383, 448), (505, 569), (626, 690), (746, 810), (866, 930), (986, 1050), (1107, 1171), (1228, 1292)]
YC, Y0, Y1 = 368, 342, 394                     # ribbon centre; band to clean between the boxes
ORANGE, GRAY = (250, 134, 62), (95, 108, 128)      # ORANGE sampled from the original ribbon
gaps = [(BOXES[i][1] + 2, BOXES[i + 1][0] - 2) for i in range(7)]
for x0, x1 in gaps:                            # erase the old ribbon and the arrows under it
    a[Y0:Y1, x0:x1 + 1] = 254

im = Image.fromarray(a)
d = ImageDraw.Draw(im)
xa, xb = BOXES[0][1], BOXES[-1][0]             # ribbon runs from step 1 to step 8
t_min, t_max = 2.0, 36.0                       # thickness in px at step 1 and at step 8
def half(x):
    return (t_min + (t_max - t_min) * (x - xa) / (xb - xa)) / 2
for x0, x1 in gaps:                            # tapered ribbon, drawn only between the boxes
    d.polygon([(x0, YC - half(x0)), (x1, YC - half(x1)), (x1, YC + half(x1)), (x0, YC + half(x0))],
              fill=ORANGE)
for x0, x1 in gaps:                            # the recurrence arrows, on top of the ribbon
    d.line([(x0, YC), (x1 - 9, YC)], fill=GRAY, width=2)
    d.polygon([(x1 - 11, YC - 6), (x1 - 11, YC + 6), (x1, YC)], fill=GRAY)

# note: "fades at every step" -> "recent steps dominate"
note_bg = tuple(int(c) for c in a[360, 1460])
d.rectangle([1456, 340, 1634, 378], fill=note_bg)
font = ImageFont.truetype('/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf', 21)
text = 'recent steps dominate'
w = d.textlength(text, font=font)
d.text((1545 - w / 2, 359), text, font=font, fill=(20, 33, 61), anchor='lm')

im.save(dst)
print('saved', dst)
