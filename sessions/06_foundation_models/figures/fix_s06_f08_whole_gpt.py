"""Fix S06-F08 (generated image): the token ids of "el" (8) and "banco" (2210) were drawn
going into the position embedding box. Every token id goes to the token embedding E; the
position embedding is indexed by the position t, not by the token. The two arrows are
redrawn into the token embedding box (one bus line, crossing the "41" arrow with a gap), and the position box gets a note, matching the
"lookup: row w of E" note of the token box.

Usage: python fix_s06_f08_whole_gpt.py s06_f08_whole_gpt_original.png s06_f08_whole_gpt.png
"""
import io
import sys

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert('RGB')).copy()
assert a.shape[:2] == (941, 1672), a.shape
ink = '#%02x%02x%02x' % tuple(a[790, 523])          # colour of the existing arrows

# erase the two vertical arrows under the position box (chips top y=812, box bottom y=765)
for x0, x1 in ((940, 967), (1087, 1113)):
    a[767:811, x0:x1] = 255

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1672)
ax.set_ylim(941, 0)
ax.axis('off')
fig.patch.set_alpha(0)
# a small bus: both ids go up to y=790, left to x=760 (with a gap where the "41" arrow
# crosses), then one arrow up into the token embedding box
yb, xb = 790, 760
kw = dict(color=ink, lw=2.6, solid_capstyle='butt')
ax.plot([953, 953], [811, yb], **kw)
ax.plot([1100, 1100], [811, yb], **kw)
ax.plot([1101.3, 819], [yb, yb], **kw)
ax.plot([803, xb - 1.3], [yb, yb], **kw)
ax.add_patch(FancyArrowPatch((xb, yb + 1.3), (xb, 767), arrowstyle='-|>', mutation_scale=16, lw=2.6,
                             color=ink, shrinkA=0, shrinkB=0))
ax.text(1043, 745, r'lookup: row $t$, one per position', ha='center', va='center', fontsize=15.5,
        color='#4b3f7a')
buf = io.BytesIO()
fig.savefig(buf, dpi=100, transparent=True)
plt.close(fig)
buf.seek(0)
over = Image.open(buf).convert('RGBA')
assert over.size == (1672, 941), over.size
img = Image.fromarray(a).convert('RGBA')
img.alpha_composite(over)
img.convert('RGB').save(dst)
print('saved', dst)
