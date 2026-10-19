"""Fix S05-F10 (generated image): remove the "T = 8" label above the step axis.
In the deck T is the window length (slide 9), and the window row says "window of T steps"
around 5 steps, so "T = 8" contradicted it. The step numbers 1-8 stay as the axis.

Usage: python fix_s05_f10_summary.py s05_f10_summary_original.png s05_f10_summary.png
"""
import sys
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape
a[36:74, 1336:1418] = 254          # "T = 8", on the white background above the first panel
Image.fromarray(a).save(dst)
print('saved', dst)
