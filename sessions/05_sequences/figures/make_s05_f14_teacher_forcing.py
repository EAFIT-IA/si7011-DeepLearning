"""S05-F14 · Teacher forcing: what the decoder receives in training and at inference.

Drawn, no data. Same style as F06 (make_s05_f06_three_text_models.py). Run from this folder:
    python make_s05_f14_teacher_forcing.py
"""
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY, BLUE, TEAL, ORANGE, RED, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#c0392b', '#8a94a3'
W, H, DPI = 16.72, 9.41, 100
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, W); ax.set_ylim(0, H)


def box(x, y, w, h, fc, ec, lw=1.4, r=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.01,rounding_size={r}', fc=fc, ec=ec, lw=lw))


def arrow(x0, y0, x1, y1, c=GRAY, lw=1.4, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=13))


def chip(x, y, t, ec=BLUE, fc='#eef5fd', tc=NAVY, w=0.8, h=0.55, fs=19):
    box(x - w / 2, y - h / 2, w, h, fc, ec, r=0.07)
    ax.text(x, y, t, ha='center', va='center', fontsize=fs if len(t) == 1 else 13, color=tc,
            family='DejaVu Sans Mono' if len(t) == 1 else 'DejaVu Sans')


TRUE = ['[START]', '2', '0', '2', '6', '-']        # y_0 … y_5 of 2026-10-19
TARGET = ['2', '0', '2', '6', '-', '1']
bands = [
    dict(y0=4.95, title='training', sub='teacher forcing', fc='#e3f4f2', ec='#7ccdc2',
         inputs=TRUE, outputs=['2', '0', '1', '6', '-', '1'],
         note='the input is always the true\nprevious character: one error\nstays at its step\n\nloss: each output against\nthe true one, all steps\nin one pass'),
    dict(y0=0.45, title='inference', sub='its own predictions', fc='#fde7ea', ec='#f29aa8',
         inputs=['[START]', '2', '0', '1', '9', '-'], outputs=['2', '0', '1', '9', '-', '0'],
         note='the input is its own previous\nprediction: one error feeds\nthe next step\n\nexposure bias'),
]
x_cells = [3.55 + k * 1.62 for k in range(6)]
for b in bands:
    y0 = b['y0']; hb = 4.0
    box(0.25, y0, W - 0.5, hb, 'white', b['ec'], lw=1.3, r=0.15)
    box(0.25, y0, 2.5, hb, b['fc'], 'none', r=0.15)
    ax.text(0.5, y0 + hb / 2 + 0.25, b['title'], fontsize=25, weight='bold', color=NAVY, va='center')
    ax.text(0.5, y0 + hb / 2 - 0.35, b['sub'], fontsize=15, color=NAVY, va='center')
    yc, yin, yout = y0 + 1.95, y0 + 0.62, y0 + 3.35
    for k, x in enumerate(x_cells):
        box(x - 0.55, yc - 0.36, 1.1, 0.72, '#dcecfb', BLUE)
        ax.text(x, yc, f'$\\mathbf{{s}}_{k + 1}$', ha='center', va='center', fontsize=17, color=NAVY)
        if k:
            arrow(x_cells[k - 1] + 0.55, yc, x - 0.55, yc, c=NAVY)
        wrong_in = b['inputs'][k] != TRUE[k]
        chip(x, yin, b['inputs'][k], ec=RED if wrong_in else BLUE, fc='#fbe5e2' if wrong_in else '#eef5fd',
             tc=RED if wrong_in else NAVY, w=1.15 if k == 0 else 0.8)
        arrow(x, yin + 0.28, x, yc - 0.36)
        wrong_out = b['outputs'][k] != TARGET[k]
        chip(x, yout, b['outputs'][k], ec=RED if wrong_out else TEAL, fc='#fbe5e2' if wrong_out else '#e3f4f2',
             tc=RED if wrong_out else NAVY)
        arrow(x, yc + 0.36, x, yout - 0.28)
        if b['title'] == 'inference' and k < 5:
            c = RED if b['outputs'][k] != TARGET[k] else GRAY
            xm = (x + x_cells[k + 1]) / 2
            ax.plot([x + 0.4, xm, xm], [yout, yout, yin], color=c, lw=1.4, ls='--')
            arrow(xm, yin, x_cells[k + 1] - 0.41, yin, c=c, lw=1.4)
    ax.text(x_cells[0] - 1.05, yin, 'input', ha='right', va='center', fontsize=14, color=GRAY)
    ax.text(x_cells[0] - 1.05, yout, 'output', ha='right', va='center', fontsize=14, color=GRAY)
    if b['title'] == 'training':
        for k, x in enumerate(x_cells):
            ax.text(x + 0.47, yout, f'vs {TARGET[k]}', ha='left', va='center', fontsize=11, color=GRAY,
                    family='DejaVu Sans Mono')
    box(13.45, y0 + 0.45, 2.9, hb - 0.9, '#f1f3f6', 'none', r=0.12)
    ax.text(13.45 + 1.45, y0 + hb / 2, b['note'], ha='center', va='center', fontsize=13.5, color=NAVY, linespacing=1.35)

fig.savefig('s05_f14_teacher_forcing.png', dpi=DPI)
print('saved s05_f14_teacher_forcing.png')
