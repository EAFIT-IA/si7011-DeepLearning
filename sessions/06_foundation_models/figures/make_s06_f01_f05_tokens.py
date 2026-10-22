"""S06-F01, F02, F03, F05 · The token-column series of Part 1 — reproducible 16:9 matplotlib figures.

Run: python make_s06_f01_f05_tokens.py
Produces in this folder (1672x941 each):
  s06_f01_same_word_two_meanings.png
  s06_f02_attention_no_weights.png
  s06_f03_query_key_value.png
  s06_f05_adding_position.png

The four figures share one running example ("me senté en el banco"), one drawing of a token
(a column of 4 cells) and one palette, so each figure differs from the previous one only in
the step it adds.

Numbers: the scores of F02 and F03 are illustrative (chosen so that the softmax weights,
rounded to two decimals, sum to 1). Every weight, weighted average and output column is
computed here from those numbers and from the token vectors below, not drawn by hand.
F05 runs real parameter-free self-attention on the vectors, with and without position.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

OUT = Path(__file__).resolve().parent
W, H = 16.72, 9.41

NAVY = '#14213d'
BLUE = '#1f6fb4'
TEAL = '#167d78'
ORANGE = '#d9631e'
GRAY = '#8a94a3'
GRAY_L = '#eef1f5'
RED = '#c0392b'
PURPLE = '#6a5d8f'

BLUES = LinearSegmentedColormap.from_list('b', ['#e3eefa', '#7fb2e5', '#1f6fb4', '#123f6b'])
ORANGES = LinearSegmentedColormap.from_list('o', ['#fdebdc', '#f3a46b', '#d9631e', '#8f3a0c'])
PURPLES = LinearSegmentedColormap.from_list('p', ['#efecf5', '#b8afd0', '#7e70a6', '#463b66'])

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 18,
    'mathtext.fontset': 'dejavusans',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# Token vectors: 4 values in [0, 1], one per cell, drawn top to bottom.
E = {
    'me':       np.array([0.10, 0.30, 0.50, 0.70]),
    'senté':    np.array([0.85, 0.20, 0.65, 0.15]),
    'en':       np.array([0.30, 0.60, 0.25, 0.55]),
    'el':       np.array([0.25, 0.50, 0.35, 0.65]),
    'banco':    np.array([0.15, 0.45, 0.70, 0.95]),
    'deposité': np.array([0.70, 0.95, 0.10, 0.40]),
    'dinero':   np.array([0.55, 0.80, 0.15, 0.25]),
}
SENT = ['me', 'senté', 'en', 'el', 'banco']


def canvas():
    fig, ax = plt.subplots(figsize=(W, H), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def column(ax, x, y, vec, w=0.55, h=1.9, cmap=BLUES, alpha=1.0, edge=NAVY, lw=2.2, split=None):
    """A token vector as a column of len(vec) cells; (x, y) is the bottom-left corner.
    split: a second vector drawn in the right half of each cell (used for x + p)."""
    n = len(vec)
    ch = h / n
    for i, v in enumerate(vec):
        cy = y + h - (i + 1) * ch
        if split is None:
            ax.add_patch(Rectangle((x, cy), w, ch, facecolor=cmap(float(np.clip(v, 0, 1))),
                                   edgecolor='none', alpha=alpha))
        else:
            ax.add_patch(Rectangle((x, cy), w / 2, ch, facecolor=BLUES(float(v)), edgecolor='none'))
            ax.add_patch(Rectangle((x + w / 2, cy), w / 2, ch, facecolor=PURPLES(float(split[i])),
                                   edgecolor='none'))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                facecolor='none', edgecolor=edge, lw=lw, alpha=max(alpha, 0.35)))


def word(ax, x, y, text, size=20, color=NAVY, style='normal', weight='normal'):
    ax.text(x, y, text, ha='center', va='top', fontsize=size, color=color, style=style, weight=weight)


def arc(ax, x0, x1, y, lw, rad=-0.45, color=ORANGE, mut=24):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), connectionstyle=f'arc3,rad={rad}',
                                 arrowstyle='-|>', mutation_scale=mut, lw=lw, color=color,
                                 shrinkA=2, shrinkB=2))


def arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.2, mut=22):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=mut,
                                 lw=lw, color=color, shrinkA=0, shrinkB=0))


def strip(ax, text, edge=GRAY, y=0.25, h=0.78):
    ax.add_patch(FancyBboxPatch((0.35, y), W - 0.7, h, boxstyle='round,pad=0,rounding_size=0.18',
                                facecolor=GRAY_L, edgecolor=edge, lw=2.2))
    ax.text(W / 2, y + h / 2, text, ha='center', va='center', fontsize=22, color=NAVY, weight='bold')


def step_tag(ax, x, y, n, label):
    ax.add_patch(Circle((x, y), 0.26, facecolor=NAVY, edgecolor='none'))
    ax.text(x, y, str(n), ha='center', va='center', fontsize=17, color='white', weight='bold')
    ax.text(x + 0.42, y, label, ha='left', va='center', fontsize=22, color=NAVY, weight='bold')


def softmax(s):
    e = np.exp(np.asarray(s) - np.max(s))
    return e / e.sum()


def save(fig, name):
    fig.savefig(OUT / name, dpi=100)
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------------------- F01
def f01():
    fig, ax = canvas()
    xs = [0.9 + i * 1.75 for i in range(5)]
    rows = [
        (5.75, ['me', 'senté', 'en', 'el', 'banco'], [0.05, 0.41, 0.10, 0.10, 0.34], 'bench', (1, 2)),
        (1.75, ['deposité', 'dinero', 'en', 'el', 'banco'], [0.38, 0.12, 0.08, 0.08, 0.34], 'bank', (0, 1)),
    ]
    cw, chh = 0.62, 2.0
    outs = []
    for y, words, wts, label, (thick, thin) in rows:
        for x, wd in zip(xs, words):
            column(ax, x, y, E[wd], w=cw, h=chh)
            word(ax, x + cw / 2, y - 0.12, wd, size=22)
        top = y + chh + 0.08
        bx = xs[4] + cw / 2
        arc(ax, xs[thick] + cw / 2, bx - 0.06, top, lw=5.0, rad=-0.2)
        arc(ax, xs[thin] + cw / 2, bx + 0.06, top, lw=1.8, rad=-0.16)
        out = sum(w_ * E[wd] for w_, wd in zip(wts, words))   # y = sum_j A_j x_j
        outs.append(out)
        arrow(ax, xs[4] + cw + 0.95, y + chh / 2, 12.1, y + chh / 2)
        ax.text((xs[4] + cw + 0.95 + 12.1) / 2, y + chh / 2 + 0.18, 'self-attention', ha='center',
                va='bottom', fontsize=20, color=NAVY)
        column(ax, 12.45, y, out, w=cw, h=chh)
        word(ax, 12.45 + cw / 2, y - 0.12, 'banco', size=22)
        ax.text(13.35, y + chh / 2, label, ha='left', va='center', fontsize=28, color=TEAL, weight='bold')
    # bracket joining the two input "banco" columns
    bx = xs[4] + cw + 0.18
    ax.plot([bx, bx + 0.25, bx + 0.25, bx], [5.75 + chh / 2, 5.75 + chh / 2, 1.75 + chh / 2, 1.75 + chh / 2],
            ls=(0, (4, 3)), color=GRAY, lw=2)
    ax.text(bx + 0.4, (5.75 + 1.75 + chh) / 2, r'same embedding $\mathbf{e}_{\mathrm{banco}}$',
            ha='left', va='center', fontsize=19, color=GRAY)
    strip(ax, 'the input vector is the same; the output depends on the other tokens')
    save(fig, 's06_f01_same_word_two_meanings.png')
    return outs


# --------------------------------------------------------------- F02 and F03 (shared layout)
PANELS = [0.45, 5.75, 11.05]            # left edge of each panel
PW = 5.1                                 # panel width
TOK_Y, TOK_W, TOK_H = 1.75, 0.46, 1.45   # token row inside each panel


def tok_x(panel, j):
    return PANELS[panel] + 0.35 + j * 0.92


def token_row(ax, panel, alphas=None, outline_last=True, vecs=None, cmap=BLUES, words=True, prefix=None):
    for j, wd in enumerate(SENT):
        v = E[wd] if vecs is None else vecs[j]
        a = 1.0 if alphas is None else alphas[j]
        edge = NAVY
        column(ax, tok_x(panel, j), TOK_Y, v, w=TOK_W, h=TOK_H, alpha=a, edge=edge,
               lw=3.2 if (outline_last and j == 4) else 1.6, cmap=cmap)
        if words:
            label = wd if prefix is None else rf'${prefix}_{j + 1}$ {wd}'
            word(ax, tok_x(panel, j) + TOK_W / 2, TOK_Y - 0.1, label, size=17 if prefix is None else 15,
                 weight='bold' if j == 4 else 'normal')


def separators(ax):
    for x in (PANELS[1] - 0.2, PANELS[2] - 0.2):
        ax.plot([x, x], [1.25, 8.9], color='#d5dae1', lw=1.5)


def scores_panel(ax, scores, label):
    p = 0
    step_tag(ax, PANELS[p] + 0.35, 8.55, 1, 'compare')
    token_row(ax, p)
    bx = tok_x(p, 4) + TOK_W / 2
    ytop = TOK_Y + TOK_H + 0.08
    for j in range(5):
        cx = tok_x(p, j) + TOK_W / 2
        if j != 4:
            arc(ax, bx, cx, ytop, lw=1.3, rad=0.35, color=GRAY, mut=14)
        ax.add_patch(FancyBboxPatch((cx - 0.36, 5.15), 0.72, 0.5, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor='white', edgecolor=GRAY, lw=1.5))
        ax.text(cx, 5.4, f'{scores[j]:.1f}'.replace('-', '−'), ha='center', va='center', fontsize=18, color=NAVY)
    ax.text(PANELS[p] + PW / 2, 6.25, label, ha='center', va='bottom', fontsize=21, color=NAVY)


def weights_panel(ax, w, label):
    p = 1
    step_tag(ax, PANELS[p] + 0.35, 8.55, 2, 'normalize')
    token_row(ax, p)
    base = TOK_Y + TOK_H + 0.25
    scale = 2.6 / max(w)
    for j in range(5):
        cx = tok_x(p, j) + TOK_W / 2
        ax.add_patch(Rectangle((cx - 0.26, base), 0.52, w[j] * scale, facecolor=ORANGE, edgecolor='none'))
        ax.text(cx, base + w[j] * scale + 0.08, f'{w[j]:.2f}', ha='center', va='bottom', fontsize=18, color=NAVY)
    ax.text(PANELS[p] + PW / 2, 7.15, label, ha='center', va='bottom', fontsize=21, color=NAVY)
    ax.text(PANELS[p] + PW / 2, 6.75, 'the weights sum to 1', ha='center', va='bottom', fontsize=17, color=GRAY)


def average_panel(ax, w, vecs, cmap, out_label, out_cmap, prefix=None):
    p = 2
    step_tag(ax, PANELS[p] + 0.35, 8.55, 3, 'average')
    alphas = [0.15 + 0.85 * wi / max(w) for wi in w]
    token_row(ax, p, alphas=alphas, vecs=vecs, cmap=cmap, prefix=prefix)
    for j in range(5):
        cx = tok_x(p, j) + TOK_W / 2
        ax.text(cx, TOK_Y + TOK_H + 0.12, f'×{w[j]:.2f}', ha='center', va='bottom', fontsize=16, color=NAVY)
    y = sum(wi * v for wi, v in zip(w, vecs))
    ox = PANELS[p] + PW / 2 - 0.35
    oy = 5.0
    for j in range(5):
        cx = tok_x(p, j) + TOK_W / 2
        ax.plot([cx, ox + 0.35], [TOK_Y + TOK_H + 0.55, oy - 0.1], color='#c3cad3', lw=1.3)
    ax.text(ox + 0.35, oy - 0.42, '+', ha='center', va='center', fontsize=22, color=NAVY, weight='bold')
    column(ax, ox, oy, y, w=0.7, h=1.7, cmap=out_cmap, lw=3.2)
    ax.text(ox + 0.95, oy + 0.85, out_label, ha='left', va='center', fontsize=26, color=NAVY)
    return y


def f02():
    scores = [0.0, 0.8, 0.6, 0.6, 2.9]
    w = softmax(scores)
    shown = np.round(w, 2)
    assert abs(shown.sum() - 1) < 1e-9, shown
    fig, ax = canvas()
    separators(ax)
    scores_panel(ax, scores, r'score $\omega_{5j} = \mathbf{x}_5^{\top}\mathbf{x}_j$')
    weights_panel(ax, shown, r'$A_{5j} = \mathrm{softmax}_j(\omega_{5j})$')
    average_panel(ax, shown, [E[t] for t in SENT], BLUES, r'$\mathbf{y}_5$', BLUES)
    ax.text(PANELS[2] + PW / 2, 7.55, r'$\mathbf{y}_5 = \sum_j A_{5j}\,\mathbf{x}_j$', ha='center',
            va='bottom', fontsize=21, color=NAVY)
    strip(ax, r'no parameters: every token is most similar to itself, so $\mathbf{y}_5$ is almost $\mathbf{x}_5$',
          edge=RED)
    save(fig, 's06_f02_attention_no_weights.png')


def f03():
    scores = [-1.0, 1.2, -0.2, -0.2, 1.0]
    w = softmax(scores)
    shown = np.round(w, 2)
    assert abs(shown.sum() - 1) < 1e-9, shown
    rng = np.random.default_rng(6)
    WK, WV = rng.uniform(0, 1, (4, 4)), rng.uniform(0, 1, (4, 4))
    WQ = rng.uniform(0, 1, (4, 4))

    def proj(Wm, x):
        v = Wm @ x
        return (v - v.min()) / (v.max() - v.min() + 1e-9) * 0.85 + 0.1

    K = [proj(WK, E[t]) for t in SENT]
    V = [proj(WV, E[t]) for t in SENT]
    q5 = proj(WQ, E['banco'])
    fig, ax = canvas()
    separators(ax)
    # panel 1: tokens, their keys, and the query of "banco"
    p = 0
    step_tag(ax, PANELS[p] + 0.35, 8.55, 1, 'compare')
    token_row(ax, p)
    ky = TOK_Y + TOK_H + 0.35
    for j in range(5):
        cx = tok_x(p, j) + TOK_W / 2
        column(ax, tok_x(p, j) + 0.06, ky, K[j], w=0.34, h=1.0, cmap=ORANGES, lw=1.4)
        ax.text(cx, ky + 1.06, rf'$\mathbf{{k}}_{j + 1}$', ha='center', va='bottom', fontsize=16, color=ORANGE)
        ax.add_patch(FancyBboxPatch((cx - 0.36, 5.15), 0.72, 0.5, boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor='white', edgecolor=GRAY, lw=1.5))
        ax.text(cx, 5.4, f'{scores[j]:.1f}'.replace('-', '−'), ha='center', va='center', fontsize=18, color=NAVY)
    qx = tok_x(p, 4) + TOK_W + 0.15
    column(ax, qx, ky, q5, w=0.34, h=1.0, cmap=ORANGES, lw=3.0)
    ax.text(qx + 0.17, ky + 1.06, r'$\mathbf{q}_5$', ha='center', va='bottom', fontsize=17, color=ORANGE, weight='bold')
    ax.text(PANELS[p] + PW / 2, 6.25, r'score $\mathbf{q}_5^{\top}\mathbf{k}_j\,/\,\sqrt{d_k}$', ha='center',
            va='bottom', fontsize=21, color=NAVY)
    ax.text(PANELS[p] + PW / 2, 7.25, r'$\mathbf{q}=W_Q\mathbf{x}$    $\mathbf{k}=W_K\mathbf{x}$    $\mathbf{v}=W_V\mathbf{x}$',
            ha='center', va='bottom', fontsize=19, color=TEAL)
    ax.text(PANELS[p] + PW / 2, 6.88, 'learned projections', ha='center', va='bottom', fontsize=16, color=TEAL)
    weights_panel(ax, shown, r'$A_{5j} = \mathrm{softmax}_j(\mathbf{q}_5^{\top}\mathbf{k}_j/\sqrt{d_k})$')
    average_panel(ax, shown, V, ORANGES, r'$\mathbf{y}_5$', ORANGES, prefix=r'\mathbf{v}')
    ax.text(PANELS[2] + PW / 2, 7.55, r'$\mathbf{y}_5 = \sum_j A_{5j}\,\mathbf{v}_j$', ha='center',
            va='bottom', fontsize=21, color=NAVY)
    strip(ax, 'learned projections: "banco" now looks mostly at "senté"', edge=TEAL)
    save(fig, 's06_f03_query_key_value.png')


# --------------------------------------------------------------------------- F05
def attend(X):
    """Parameter-free self-attention, rows are tokens: Y = softmax(X X^T) X."""
    S = X @ X.T
    A = np.exp(S - S.max(1, keepdims=True))
    A /= A.sum(1, keepdims=True)
    return A @ X


def f05():
    fig, ax = canvas()
    X = np.stack([E[t] for t in SENT])
    pos = np.arange(5)[:, None]
    freq = np.array([1.0, 0.5, 0.25, 0.125])[None, :]
    P = 0.5 + 0.45 * np.sin(pos * freq * 1.3 + np.array([0, 1.1, 2.2, 3.3])[None, :])   # one vector per position
    xs = [2.35 + i * 1.3 for i in range(5)]
    cw, chh = 0.6, 1.5
    ys = {'x': 1.95, 'p': 4.35, 's': 6.75}
    for i, t in enumerate(SENT):
        column(ax, xs[i], ys['x'], X[i], w=cw, h=chh)
        word(ax, xs[i] + cw / 2, ys['x'] - 0.1, t, size=19)
        column(ax, xs[i], ys['p'], P[i], w=cw, h=chh, cmap=PURPLES)
        column(ax, xs[i], ys['s'], X[i], w=cw, h=chh, split=P[i])
        ax.text(xs[i] + cw / 2, ys['x'] + chh + 0.45, '+', ha='center', va='center', fontsize=30, color=NAVY)
        ax.text(xs[i] + cw / 2, ys['p'] + chh + 0.45, '=', ha='center', va='center', fontsize=30, color=NAVY)
    lx = xs[0] - 0.35
    ax.text(lx, ys['x'] + chh / 2, r'$\mathbf{x}_i$', ha='right', va='center', fontsize=26, color=BLUE)
    ax.text(lx, ys['x'] + chh / 2 - 0.45, 'token', ha='right', va='center', fontsize=15, color=BLUE)
    ax.text(lx, ys['p'] + chh / 2, r'$\mathbf{p}_i$', ha='right', va='center', fontsize=26, color=PURPLE)
    ax.text(lx, ys['p'] + chh / 2 - 0.45, 'position', ha='right', va='center', fontsize=15, color=PURPLE)
    ax.text(lx, ys['s'] + chh / 2, r'$\mathbf{x}_i + \mathbf{p}_i$', ha='right', va='center', fontsize=24, color=NAVY)
    ax.text(lx, ys['s'] + chh / 2 - 0.45, 'input to attention', ha='right', va='center', fontsize=15, color=NAVY)

    # right: the output of "banco" when the sentence is in two orders
    ax.plot([9.0, 9.0], [1.4, 8.9], color='#d5dae1', lw=1.5)
    order2 = [4, 3, 2, 1, 0]
    Y1, Y2 = attend(X), attend(X[order2])
    assert np.allclose(Y2, Y1[order2])                # without position: same outputs, shuffled
    Z1, Z2 = attend(X + P), attend(X[order2] + P)     # position i is added to whatever token sits at i
    assert not np.allclose(Z2, Z1[order2])
    lo, hi = 0.6, 1.5                                 # shared colour scale for the two outputs with position

    def block(ytop, title, out1, out2, sign, caption, ccolor):
        ax.text(12.6, ytop, title, ha='center', va='center', fontsize=22, color=NAVY, weight='bold')
        for k, (txt, out) in enumerate([('me senté en el banco', out1), ('banco el en senté me', out2)]):
            y = ytop - 1.0 - k * 1.25
            ax.text(9.35, y, txt, ha='left', va='center', fontsize=17, color=NAVY, style='italic')
            arrow(ax, 12.75, y, 13.55, y, lw=1.8, mut=16)
            column(ax, 13.75, y - 0.5, out, w=0.42, h=1.0, lw=1.8)
            ax.text(14.35, y, r'output of "banco"', ha='left', va='center', fontsize=14, color=GRAY)
        ax.text(13.96, ytop - 1.62, sign, ha='center', va='center', fontsize=26, color=ccolor, weight='bold')
        ax.text(12.6, ytop - 3.05, caption, ha='center', va='center', fontsize=16, color=ccolor)

    block(8.45, r'without $\mathbf{p}_i$', Y1[4], Y2[0], '=', 'order is invisible to attention', RED)
    block(4.85, r'with $\mathbf{p}_i$', (Z1[4] - lo) / (hi - lo), (Z2[0] - lo) / (hi - lo), '≠',
          'the output depends on where "banco" is', TEAL)
    strip(ax, r'self-attention sees a set of tokens; adding $\mathbf{p}_i$ turns it into a sequence')
    save(fig, 's06_f05_adding_position.png')


if __name__ == '__main__':
    f01()
    f02()
    f03()
    f05()
