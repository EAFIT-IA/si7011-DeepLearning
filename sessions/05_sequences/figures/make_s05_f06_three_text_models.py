"""S05-F06 · What each text model sees — reproducible 16:9 matplotlib figure.

Run: python make_s05_f06_three_text_models.py
Produces s05_f06_three_text_models.png (1672x941) in this folder.
"""
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import matplotlib as mpl

W, H = 16.72, 9.41
NAVY = '#14213d'
BLUE = '#1f6fb4'
BLUE_L = '#dceefd'
TEAL = '#167d78'
TEAL_L = '#daf3ef'
GRAY = '#7b8795'
GRAY_L = '#f1f3f6'
ORANGE = '#d9631e'
RED = '#dc3b64'
GREEN = '#229d65'
PINK = '#da2e90'
INK = '#253447'
COLORS = [BLUE, ORANGE, GREEN, PINK]
WORDS = ['not', 'good', ',', 'but', 'very', 'fun']
X = [3.18 + i * 1.12 for i in range(6)]

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 14,
    'mathtext.fontset': 'dejavusans',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})
fig, ax = plt.subplots(figsize=(16.72, 9.41), dpi=100)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def rounded(x, y, w, h, fc, ec, radius=.13, lw=1.3, z=1):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad=0.025,rounding_size={radius}',
        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z)
    ax.add_patch(p)
    return p


def label(x, y, s, size=16, color=NAVY, ha='center', va='center', weight='normal', z=10, **kwargs):
    ax.text(x, y, s, ha=ha, va=va, color=color, fontsize=size, fontweight=weight, zorder=z, **kwargs)


def arrow(start, end, color=INK, lw=1.4, ms=11, z=7):
    a = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=ms,
                        linewidth=lw, color=color, shrinkA=1.5, shrinkB=1.5, zorder=z)
    ax.add_patch(a)
    return a


def panel(y, title, fc, stroke):
    rounded(.16, y, 16.40, 2.76, 'white', stroke, radius=.15, lw=1.2)
    # header/tag region within the shared rounded panel
    rounded(.16, y, 2.34, 2.76, fc, fc, radius=.15, lw=0)
    label(.42, y+1.40, title, size=23, ha='left', weight='bold')


def word_row(y):
    for x, word in zip(X, WORDS):
        rounded(x-.40, y, .80, .43, '#f0f8ff', '#75b9ed', radius=.04, lw=1.1)
        label(x, y+.21, word, size=16.5)


def embeds(y, row, show_labels=True):
    # Small bars above word chips. Label placed to the RIGHT, never on the vertical arrow.
    for i, x in enumerate(X):
        rect = Rectangle((x-.085, y), .17, .56, linewidth=1.15,
                         edgecolor=BLUE, facecolor='#e1f1ff', zorder=5)
        ax.add_patch(rect)
        for j in (1, 2, 3):
            ax.plot([x-.085, x+.085], [y + j*.14]*2, color='#73b9eb', lw=.65, zorder=6)
        if show_labels:
            label(x+.19, y+.30, r'$e_w$', size=13.5, ha='left')
        arrow((x, y-.09), (x, y-.005), color=GRAY, lw=1.2, ms=8)


def right_note(y, text):
    rounded(13.64, y+.73, 2.75, 1.02, GRAY_L, GRAY_L, radius=.09, lw=0)
    label(15.02, y+1.24, text, size=14.0)

# top: bag of embeddings
y=6.43
panel(y, 'bag of\nembeddings', '#e1f0fd', '#7bbefd')
word_row(y+.30)
embeds(y+.88, 'bag', show_labels=True)
# fan to mean: the circle sits above the embedding tops, so the six arrows open as a fan
yc = y + 1.86
for x in X:
    arrow((x, y + 1.47), (10.10, yc), color=GRAY, lw=1.05, ms=8, z=3)
ax.add_patch(Circle((10.54, yc), radius=.45, edgecolor=TEAL,
                    facecolor=TEAL_L, lw=1.5, zorder=5))
label(10.54, yc, 'mean', size=16)
arrow((11.00, yc), (11.43, yc), color=GRAY, ms=11)
rounded(11.45, yc - .30, 1.00, .59, TEAL_L, TEAL, radius=.07)
label(11.95, yc, 'linear', size=15)
arrow((12.48, yc), (12.89, yc), color=GRAY)
label(13.10, yc, r'$\hat{y}$', size=22)
right_note(y,'same output for any\norder of the words')

# middle: 1-D CNN
y=3.34
panel(y, '1-D CNN', '#e5f8f3', '#67cbb9')
word_row(y+.25)
embeds(y+.80, 'cnn', show_labels=True)
# EXACTLY FOUR brackets over words 1-3, 2-4, 3-5, 4-6; no other brackets.
# Draw staggered heights to keep the overlapping spans distinguishable.
for k, c in enumerate(COLORS):
    i = k
    x1, x3 = X[i]-.40, X[i+2]+.40
    by = y + 1.52 + .065*k
    # open downward bracket touching the 3 covered embedding columns
    ax.plot([x1,x1,x3,x3],[by-.13,by,by,by-.13], lw=2.0, color=c, zorder=8)
    # three fine stems emphasize exactly which 3 embeddings are in this window
    for j in range(i,i+3):
        ax.plot([X[j],X[j]],[y+1.38,by-.13], lw=.7, color=c, alpha=.6, zorder=7)
    filter_x = (X[i]+X[i+2])/2
    fy = y+2.18
    rounded(filter_x-.28, fy-.19, .56,.39, '#d7eafb', BLUE, radius=.065, lw=1.2)
    # matching color arrow from bracket to filter: blue/orange/green/magenta
    arrow((filter_x,by+.04),(filter_x,fy-.20),color=c,lw=2.0,ms=11,z=9)
    for dx,height in [(-.12,.085),(-.04,.19),(.04,.27),(.12,.12)]:
        ax.plot([filter_x+dx]*2,[fy-height/2,fy+height/2], color=NAVY, lw=1.8, zorder=11)
    # output from each filter to shared max pool
    arrow((filter_x+.29,fy+.02),(10.44,y+1.72 + .07*k),color=GRAY,lw=.95,ms=7,z=2)
rounded(10.48,y+1.29,1.52,.88,TEAL_L,TEAL,radius=.09)
label(11.24,y+1.73,'max\nover time',size=15)
arrow((12.03,y+1.73),(12.86,y+1.73),color=GRAY)
label(13.10,y+1.73,r'$\hat{y}$',size=22)
right_note(y,'sees order within\n3 words')

# bottom: LSTM
y=.24
panel(y, 'LSTM', '#ffe7ed', '#ff85a6')
word_row(y+.24)
embeds(y+.81, 'lstm', show_labels=True)
for i,x in enumerate(X):
    rounded(x-.36,y+1.77,.72,.48,BLUE_L,BLUE,radius=.075,lw=1.5)
    arrow((x,y+1.39),(x,y+1.75),color=GRAY,lw=1.2,ms=10)
    if i<5:
        arrow((x+.38,y+2.01),(X[i+1]-.38,y+2.01),color=INK,lw=1.4,ms=11)
arrow((X[-1]+.37,y+2.01),(9.95,y+2.01),color=ORANGE,lw=1.9)
label(10.20,y+2.01,r'$h_T$',size=21,color=ORANGE,weight='bold')
arrow((10.50,y+2.01),(10.84,y+2.01),color=ORANGE,lw=1.9)
rounded(10.87,y+1.71,1.01,.59,TEAL_L,TEAL,radius=.07)
label(11.37,y+2.01,'linear',size=15)
arrow((11.90,y+2.01),(12.67,y+2.01),color=GRAY)
label(12.92,y+2.01,r'$\hat{y}$',size=22)
right_note(y,'sees the whole order,\nstep by step')

out = Path(__file__).with_name('s05_f06_three_text_models.png')
fig.savefig(out, dpi=100, bbox_inches=None, pad_inches=0)
plt.close(fig)
print(out)
