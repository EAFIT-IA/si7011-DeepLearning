"""Fix S05-F04 (generated image): remove the upward h_t arrow that leaves from the
c_t memory line. In an LSTM, h_t = o_t ⊙ tanh(c_t); drawing an output straight out of
c_t suggests h_t = c_t. The correct h_t output (right edge, through the read gate) stays.

Usage: python fix_s05_f04_lstm_cell.py s05_f04_lstm_cell_original.png s05_f04_lstm_cell.png
"""
import sys
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape

# vertical arrow x≈1440, y 30–223 (it ends where the orange line starts, y≈224):
# copy the same rows from 40 px to the left, which hold only background and the cell border
a[24:225, 1430:1452] = a[24:225, 1390:1412]
# the "h_t" label to the right of the arrowhead: plain white background there
a[10:85, 1452:1530] = 255
Image.fromarray(a).save(dst)
print('saved', dst)
