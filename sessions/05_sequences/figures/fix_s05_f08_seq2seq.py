"""Fix S05-F08 (generated image): each decoder step after the first received two inputs,
the character chip below it and a curved arrow from the chip to its left. With teacher
forcing, step tau receives only y_{tau-1}, which is already the chip below it. Remove the
curves and redraw one straight arrow per step.

Usage: python fix_s05_f08_seq2seq.py s05_f08_seq2seq_original.png s05_f08_seq2seq.png
"""
import sys
import numpy as np
from PIL import Image, ImageDraw

src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
a = np.asarray(im).copy()
assert a.shape[:2] == (941, 1672), a.shape

BOX_BOTTOM, CHIP_TOP = 455, 515                 # decoder boxes end / input chips start
XS = [977, 1056, 1136, 1216, 1295, 1375, 1454, 1532, 1607]   # steps 2..10 (step 1 has no curve)
color = tuple(int(c) for c in a[480, 891])      # arrow color of the untouched first step
bg = a[480, 940].copy()                         # panel fill between the rows

a[BOX_BOTTOM - 2:CHIP_TOP, 935:1645] = bg       # erase straight arrows, curves and their heads (steps 2..10)
im = Image.fromarray(a)
d = ImageDraw.Draw(im)
for x in XS:
    d.line([(x, CHIP_TOP - 1), (x, BOX_BOTTOM + 12)], fill=color, width=2)
    d.polygon([(x - 6, BOX_BOTTOM + 13), (x + 6, BOX_BOTTOM + 13), (x, BOX_BOTTOM + 1)], fill=color)
im.save(dst)
print('saved', dst)
