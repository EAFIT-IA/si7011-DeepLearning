"""S05-F16, F17, F18 · Part 3 figures, drawn (no data). Same style as F06 and F14.

    F16  truncated BPTT: the state goes forward across chunks, the gradient stops at each detach()
    F17  stacked and bidirectional, and where the window ends when forecasting
    F18  recurrent dropout: the same mask at every step vs a new mask at every step

Run from this folder:  python make_s05_f16_f18_parte3.py
"""
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY, BLUE, TEAL, ORANGE, RED, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#c0392b', '#8a94a3'
W, H, DPI = 16.72, 9.41, 100
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def lienzo():
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    return fig, ax


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.1, ls='-'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.01,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(ax, x0, y0, x1, y1, c=GRAY, lw=1.4, ls='-', ms=13):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=ms))


def nota(ax, x, y, w, h, texto, fs=14):
    box(ax, x, y, w, h, '#f1f3f6', 'none', r=0.12)
    ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center', fontsize=fs, color=NAVY, linespacing=1.35)


def celda(ax, x, y, etiqueta='', w=0.9, h=0.62, fc='#dcecfb', ec=BLUE, fs=15):
    box(ax, x - w / 2, y - h / 2, w, h, fc, ec)
    if etiqueta:
        ax.text(x, y, etiqueta, ha='center', va='center', fontsize=fs, color=NAVY)


# ---------------------------------------------------------------- F16 · truncated BPTT
def f16():
    fig, ax = lienzo()
    k, chunks = 4, 3
    n = k * chunks
    x0, dx = 1.25, 1.12
    xs = [x0 + i * dx for i in range(n)]
    yc, yin, yl = 5.0, 3.55, 6.55
    for c in range(chunks):
        a, b = xs[c * k] - 0.62, xs[c * k + k - 1] + 0.62
        box(ax, a, 2.95, b - a, 4.35, '#f5f9fe', '#c9d9ec', r=0.15)
        ax.text((a + b) / 2, 7.55, f'chunk {c + 1}', ha='center', fontsize=17, color=NAVY, weight='bold')
    for i, x in enumerate(xs):
        celda(ax, x, yc)
        ax.text(x, yin, f'$\\mathbf{{x}}_{{{i + 1}}}$', ha='center', va='center', fontsize=15, color=NAVY)
        arrow(ax, x, yin + 0.25, x, yc - 0.31)
        if i:
            arrow(ax, xs[i - 1] + 0.45, yc, x - 0.45, yc, c=NAVY, lw=1.8)
    # loss at the end of each chunk and the gradient flowing back inside the chunk only
    for c in range(chunks):
        last = xs[c * k + k - 1]
        first = xs[c * k]
        box(ax, last - 0.33, yl - 0.25, 0.66, 0.5, '#e3f4f2', TEAL, r=0.08)
        ax.text(last, yl, '$L$', ha='center', va='center', fontsize=16, color=NAVY)
        arrow(ax, last, yc + 0.31, last, yl - 0.25)
        ys = yc + 0.72
        ax.annotate('', xy=(first - 0.2, ys), xytext=(last + 0.05, ys),
                    arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=3.2, mutation_scale=18, shrinkA=0, shrinkB=0))
        if c:
            xb = (xs[c * k - 1] + xs[c * k]) / 2
            ax.plot([xb, xb], [3.0, 7.25], color=RED, lw=2, ls='--')
            ax.text(xb, 2.65, 'h.detach()', ha='center', fontsize=14, color=RED, family='DejaVu Sans Mono')
            ax.plot([xb], [ys], 'X', color=RED, ms=14)
    ax.plot([], [], color=NAVY, lw=1.8, label='forward: the state $\\mathbf{h}$ crosses every cut')
    ax.plot([], [], color=ORANGE, lw=3.2, label='backward: the gradient stops at the cut')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.04, 0.97), fontsize=15, ncol=2)
    nota(ax, 0.6, 0.45, W - 1.2, 1.55,
         'the gradient reaches at most $k = 4$ steps back  ·  memory for $k$ steps of activations, not for the whole sequence\n'
         'a dependency longer than $k$ steps can still be carried by $\\mathbf{h}$, but it is never trained', fs=15)
    fig.savefig('s05_f16_truncated_bptt.png', dpi=DPI); plt.close(fig)
    print('saved s05_f16_truncated_bptt.png')


# ---------------------------------------------------------------- F17 · stacked and bidirectional
def f17():
    fig, ax = lienzo()
    # left panel: stacked
    box(ax, 0.25, 0.45, 7.75, 8.5, 'white', '#9cc5ea', r=0.15)
    ax.text(0.6, 8.35, 'stacked:  num_layers=2', fontsize=20, weight='bold', color=NAVY, va='center')
    xs = [1.95 + i * 1.3 for i in range(5)]
    y1, y2, yin = 4.1, 6.25, 2.2
    for i, x in enumerate(xs):
        ax.text(x, yin, f'$\\mathbf{{x}}_{{{i + 1}}}$', ha='center', va='center', fontsize=16, color=NAVY)
        celda(ax, x, y1)
        celda(ax, x, y2, fc='#c8def6')
        arrow(ax, x, yin + 0.28, x, y1 - 0.31)
        arrow(ax, x, y1 + 0.31, x, y2 - 0.31, c=BLUE, lw=1.8)
        if i:
            arrow(ax, xs[i - 1] + 0.45, y1, x - 0.45, y1, c=NAVY)
            arrow(ax, xs[i - 1] + 0.45, y2, x - 0.45, y2, c=NAVY)
    ax.text(0.45, y1, 'layer 1', fontsize=14, color=GRAY, va='center')
    ax.text(0.45, y2, 'layer 2', fontsize=14, color=GRAY, va='center')
    box(ax, xs[-1] - 0.35, 7.25, 0.7, 0.55, '#e3f4f2', TEAL, r=0.08)
    ax.text(xs[-1], 7.52, '$\\hat y$', ha='center', va='center', fontsize=17, color=NAVY)
    arrow(ax, xs[-1], y2 + 0.31, xs[-1], 7.25)
    nota(ax, 0.6, 0.75, 7.05, 0.95, '$\\mathbf{h}_t$ of layer 1 is the input $\\mathbf{x}_t$ of layer 2', fs=15)

    # right panel: bidirectional over a forecasting window
    box(ax, 8.35, 0.45, 8.12, 8.5, 'white', '#9cc5ea', r=0.15)
    ax.text(8.7, 8.35, 'bidirectional:  bidirectional=True', fontsize=20, weight='bold', color=NAVY, va='center')
    xs = [9.3 + i * 1.12 for i in range(5)]
    yf, yb, yin = 6.0, 2.95, 4.47
    for i, x in enumerate(xs):
        ax.text(x, yin, f'$\\mathbf{{x}}_{{{i + 1}}}$', ha='center', va='center', fontsize=16, color=NAVY)
        celda(ax, x, yf, w=0.8)
        celda(ax, x, yb, w=0.8, fc='#e3f4f2', ec=TEAL)
        arrow(ax, x, yin + 0.28, x, yf - 0.31)
        arrow(ax, x, yin - 0.28, x, yb + 0.31)
        if i:
            arrow(ax, xs[i - 1] + 0.4, yf, x - 0.4, yf, c=NAVY)
            arrow(ax, x - 0.4, yb, xs[i - 1] + 0.4, yb, c=TEAL)
    ax.text(xs[0] - 0.55, yf, '→', fontsize=20, color=NAVY, ha='right', va='center')
    ax.text(xs[0] - 0.55, yb, '←', fontsize=20, color=TEAL, ha='right', va='center')
    ax.text(xs[2], 6.85, 'state at $t$:  $[\\,\\overrightarrow{\\mathbf{h}}_t\\,;\\,\\overleftarrow{\\mathbf{h}}_t\\,]$',
            ha='center', fontsize=16, color=NAVY)
    # the end of the window
    xe = xs[-1] + 0.68
    ax.plot([xe, xe], [1.85, 7.4], color=RED, lw=2, ls='--')
    ax.text(xe, 7.6, 'end of the window = now', ha='center', fontsize=13.5, color=RED)
    ax.annotate('', xy=(xs[0] - 0.3, 1.95), xytext=(xs[-1] + 0.3, 1.95),
                arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.2))
    ax.text(xs[2], 1.7, 'the window: may be read in both directions', ha='center', va='top', fontsize=13.5, color=NAVY)
    for j in range(2):
        x = xe + 0.7 + j * 0.8
        box(ax, x - 0.3, yin - 0.31, 0.6, 0.62, '#f1f3f6', GRAY, ls='--', r=0.06)
        ax.text(x, yin, '?', ha='center', va='center', fontsize=16, color=GRAY)
    ax.text(xe + 1.05, yin - 0.55, 'the future:\nnever an input', ha='center', va='top', fontsize=13.5, color=RED)
    fig.savefig('s05_f17_stacked_bidirectional.png', dpi=DPI); plt.close(fig)
    print('saved s05_f17_stacked_bidirectional.png')


# ---------------------------------------------------------------- F18 · recurrent dropout masks
def f18():
    fig, ax = lienzo()
    n_t, n_u, p = 6, 6, 0.2
    rng = np.random.default_rng(4)
    fija = np.array([1, 0, 1, 1, 0, 1])               # same mask at every step: units 2 and 5 off
    nuevas = (rng.random((n_t, n_u)) > p).astype(int)
    nuevas[-1] = 1                                    # h_6 does not enter another step: no mask
    nuevas[:, 2] = [1, 1, 0, 1, 1, 1]                 # the highlighted unit is dropped once, at step 3
    U = 2                                             # highlighted unit (third from the top)
    misma = fija[None].repeat(n_t, 0); misma[-1] = 1
    filas = [('same mask at every step  (Gal & Ghahramani, 2016)', misma, 5.05, TEAL,
              'a unit is on for the whole sequence\nor off: what the units that are on\nremember survives'),
             ('a new mask at every step', nuevas, 0.55, RED,
              f'with $p = {p}$, a unit crosses $k$ steps\nwith probability $(1-p)^k$:\n'
              f'{(1 - p) ** (n_t - 1):.2f} for these {n_t - 1} steps,\n{(1 - p) ** 24:.3f} for one day of hourly data')]
    xs = [3.4 + i * 1.45 for i in range(n_t)]
    s = 0.36
    for titulo, M, y0, c, texto in filas:
        hb = 3.95
        box(ax, 0.25, y0, W - 0.5, hb, 'white', '#9cc5ea', r=0.15)
        ax.text(0.55, y0 + hb - 0.45, titulo, fontsize=19, weight='bold', color=NAVY, va='center')
        ytop = y0 + hb - 1.05
        ys = [ytop - u * (s + 0.06) for u in range(n_u)]
        for t, x in enumerate(xs):
            for u, yy in enumerate(ys):
                on = M[t, u] == 1
                box(ax, x - s / 2, yy - s / 2, s, s, '#dcecfb' if on else '#eceff3', BLUE if on else GRAY, lw=1.1, r=0.04)
                if not on:
                    ax.text(x, yy, '×', ha='center', va='center', fontsize=14, color=GRAY)
            ax.text(x, ys[-1] - 0.45, f'$\\mathbf{{h}}_{{{t + 1}}}$', ha='center', va='center', fontsize=14, color=NAVY)
            if t:
                arrow(ax, xs[t - 1] + s / 2 + 0.08, ys[-1] - 0.45, x - s / 2 - 0.08, ys[-1] - 0.45, c=GRAY, lw=1.1, ms=10)
        # the highlighted unit's path through time
        yy = ys[U]
        for t in range(n_t - 1):
            ok = M[t, U] == 1 and M[t + 1, U] == 1
            ax.plot([xs[t] + s / 2, xs[t + 1] - s / 2], [yy, yy], color=ORANGE if ok else RED,
                    lw=3 if ok else 1.6, ls='-' if ok else ':')
        ax.text(xs[0] - s / 2 - 0.12, yy, 'unit 3', ha='right', va='center', fontsize=13, color=ORANGE)
        nota(ax, 12.45, y0 + 0.45, 3.65, hb - 0.9, texto, fs=13.5)
    fig.text(0.5, 0.012, 'the mask acts on $\\mathbf{h}_t$ where it enters the next step ($W_{hh}$); the last state enters no further step',
             ha='center', fontsize=13, color=GRAY)
    fig.savefig('s05_f18_recurrent_dropout.png', dpi=DPI); plt.close(fig)
    print('saved s05_f18_recurrent_dropout.png')


if __name__ == '__main__':
    f16(); f17(); f18()
