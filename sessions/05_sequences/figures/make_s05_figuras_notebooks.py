"""S05 figures drawn from what the notebooks save, so the slide and the notebook always agree.

Run from this folder after running the notebooks:
    python make_s05_figuras_notebooks.py ../notebooks .

| Figure | Reads (saved by)                                       |
|--------|--------------------------------------------------------|
| F02    | particion_jena.json        (notebook 5.1)              |
| F03    | gradiente_tiempo_rnn.npz   (notebook 5.2)              |
| F07    | vecinos_embedding.json     (notebook 5.4)              |
| F09    | atencion_fechas.npz        (notebook 5.5, section 6)   |
| F11    | gradiente_tiempo_lstm_gru.npz (notebook 5.3, section 4) |
| F12    | resultados_jena_51.json    (notebook 5.1, section 6)   |
| F13    | resultados_seq2seq.json    (notebook 5.5)              |

A figure whose input file is missing is skipped with a notice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else '../notebooks')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else '.')

NAVY, BLUE, TEAL, ORANGE, RED, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#c0392b', '#8a94a3'
W, H, DPI = 16.72, 9.41, 100                       # 1672 × 941 px, as the generator figures
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'axes.edgecolor': GRAY,
                     'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def need(name):
    p = SRC / name
    if not p.exists():
        print(f'skip: {name} not found in {SRC} (run the notebook that saves it)')
        return None
    return p


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print('saved', OUT / name)


# ---------------------------------------------------------------- F07 · neighbors in the embedding
def f07():
    p = need('vecinos_embedding.json')
    if p is None:
        return
    d = json.load(open(p))
    # list of {"palabra", "puntaje" (w·e_w), "vecinos": [[word, cos], ...5], "signo": {word: +1/-1}}
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    POS, NEG = (BLUE, '#e3eefa'), (RED, '#fbe5e2')
    x_q, x0, dx, cw, ch = 0.55, 3.35, 2.6, 2.35, 0.82
    ax.text(x0 + 2 * dx + cw / 2, H - 0.62, 'five nearest neighbors in the learned embedding $E$  (cosine similarity)',
            ha='center', fontsize=17, color=GRAY)
    n = len(d)
    for i, row in enumerate(d):
        y = H - 1.55 - i * (H - 2.3) / n
        col = POS if row.get('puntaje', 0) > 0 else NEG
        ax.text(x_q, y, row['palabra'], fontsize=26, weight='bold', color=col[0], va='center')
        for j, (w, c) in enumerate(row['vecinos']):
            cc = POS if row['signo'].get(w, 0) > 0 else NEG
            x = x0 + j * dx
            ax.add_patch(FancyBboxPatch((x, y - ch / 2), cw, ch, boxstyle='round,pad=0.02,rounding_size=0.12',
                                        fc=cc[1], ec=cc[0], lw=1.4))
            ax.text(x + cw / 2, y + 0.12, w, ha='center', va='center', fontsize=17 if len(w) < 12 else 14, color=NAVY)
            ax.text(x + cw / 2, y - 0.22, f'{c:.2f}', ha='center', va='center', fontsize=13, color=GRAY)
    ax.text(W / 2, 0.35, 'color = sign of the word in the output layer, $\\mathbf{w}^\\top \\mathbf{e}_w$:   '
            'blue pushes toward "positive", red toward "negative"  ·  bag of embeddings, notebook 5.4',
            ha='center', fontsize=14, color=GRAY)
    save(fig, 's05_f07_embedding_neighbors.png')


# ---------------------------------------------------------------- F09 · attention as an alignment
def f09():
    p = need('atencion_fechas.npz')
    if p is None:
        return
    z = np.load(p, allow_pickle=False)
    ejemplos, salidas = [str(e) for e in z['ejemplos']], [str(s) for s in z['salidas']]
    mapas = [z[f'alpha_{k}'] for k in range(len(ejemplos))]
    widths = [len(e.split()) for e in ejemplos]
    fig, axes = plt.subplots(1, 3, figsize=(W, H), gridspec_kw={'width_ratios': widths, 'wspace': 0.12})
    fig.subplots_adjust(left=0.05, right=0.93, top=0.95, bottom=0.27)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('att', ['#ffffff', '#fbd9bf', ORANGE, '#8a3410'])
    for ax, m, e, s in zip(axes, mapas, ejemplos, salidas):
        words = e.split()
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(words)), words, rotation=62, ha='right', rotation_mode='anchor', fontsize=13)
        ax.set_yticks(range(len(s)), list(s), fontsize=16, family='DejaVu Sans Mono')
        ax.set_xticks(np.arange(-.5, len(words)), minor=True)
        ax.set_yticks(np.arange(-.5, len(s)), minor=True)
        ax.grid(which='minor', color='white', lw=1.2)
        ax.tick_params(which='minor', length=0)
        ax.tick_params(which='major', length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel('output character  $\\tau$', fontsize=15)
    cax = fig.add_axes([0.945, 0.27, 0.012, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('$\\alpha_{\\tau t}$', fontsize=18, color=NAVY)
    cb.outline.set_visible(False)
    fig.text(0.485, 0.03, 'input word  $t$  ·  each row sums to 1  ·  seq2seq with additive attention, $d = 24$ (notebook 5.5)',
             ha='center', fontsize=14, color=GRAY)
    save(fig, 's05_f09_attention_alignment.png')


# ---------------------------------------------------------------- F02 · Jena windows and split
def f02():
    p = need('particion_jena.json')
    if p is None:
        return
    d = json.load(open(p))
    # keys: temp (hourly °C), cortes [i_val, i_test], T, H_fut, ventanas (three window starts before the cut)
    temp = np.asarray(d['temp']); t = np.arange(len(temp))
    i_val, i_test = d['cortes']
    T, Hf = d['T'], d['H_fut']
    fig, (ax, axz) = plt.subplots(2, 1, figsize=(W, H), gridspec_kw={'height_ratios': [1, 1.45], 'hspace': 0.32})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.08)
    for (a0, b0, c, lab) in [(0, i_val, BLUE, 'train · 50 %'), (i_val, i_test, TEAL, 'validation · 25 %'),
                             (i_test, len(t), ORANGE, 'test · 25 %')]:
        ax.axvspan(a0, b0, color=c, alpha=0.10, lw=0)
        ax.text((a0 + b0) / 2, 41, lab, ha='center', va='bottom', color=c, fontsize=16)
    ax.plot(t, temp, color=GRAY, lw=0.3)
    ax.set_ylim(-26, 40); ax.set_ylabel('°C'); ax.set_xlim(0, len(t))
    ax.set_xticks([0, i_val, i_test, len(t)], ['2009', '', '', '2017'])
    for s0 in (i_val, i_test):
        ax.axvline(s0, color=NAVY, lw=1)
    x0, x1 = i_val - 3 * (T + Hf) - 12, i_val + T + 60
    ax.axvspan(x0, x1, ymin=0, ymax=1, fill=False, ec=NAVY, lw=1.2, ls=':')

    zz = slice(x0, x1)
    axz.plot(t[zz] - i_val, temp[zz], color=GRAY, lw=1.2)
    axz.axvline(0, color=NAVY, lw=1.5, ls='--')
    lo = temp[zz].min()
    axz.text(-6, temp[zz].max() + 0.6, 'train', ha='right', fontsize=15, color=BLUE)
    axz.text(6, temp[zz].max() + 0.6, 'validation', ha='left', fontsize=15, color=TEAL)
    lanes = [lo - 3 - 2.6 * k for k in range(4)]
    for k, s0 in enumerate(d['ventanas']):
        y = lanes[k]
        axz.plot([s0 - i_val, s0 + T - i_val], [y, y], color=BLUE, lw=7, solid_capstyle='butt')
        axz.plot([s0 + T - i_val, s0 + T - 1 + Hf - i_val], [y, y], color=ORANGE, lw=1.4, ls=':')
        axz.plot([s0 + T - 1 + Hf - i_val], [y], 'o', color=ORANGE, ms=11)
    bad = -T // 2; y = lanes[3]
    axz.plot([bad, bad + T], [y, y], color=RED, lw=7, solid_capstyle='butt')
    axz.plot([bad + T, bad + T - 1 + Hf], [y, y], color=RED, lw=1.4, ls=':')
    axz.plot([bad + T - 1 + Hf], [y], 'X', color=RED, ms=13)
    axz.text(bad - 10, y, 'crosses the cut: not allowed', color=RED, va='center', ha='right', fontsize=14)
    axz.plot([], [], color=BLUE, lw=7, label=f'window: T = {T} h of the 14 variables')
    axz.plot([], [], 'o', color=ORANGE, ms=11, label=f'target: temperature {Hf} h after the window')
    axz.legend(frameon=False, loc='upper left', fontsize=14)
    axz.set_xlim(x0 - i_val, x1 - i_val); axz.set_ylim(lanes[3] - 2.2, temp[zz].max() + 3.5)
    axz.set_xlabel('hours from the train | validation cut'); axz.set_ylabel('°C')
    save(fig, 's05_f02_windows_split.png')


# ---------------------------------------------------------------- F03 · gradient through time
def f03():
    p = need('gradiente_tiempo_rnn.npz')
    if p is None:
        return
    z = np.load(p)
    # expected arrays: g (mean ||dL/dx_t|| over a validation batch, t = 1..T), optional g_lstm, g_gru
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.11)
    T = len(z['g'])
    t = np.arange(1, T + 1)
    pos = lambda v: np.where(v > 0, v / z['g'][-1], np.nan)          # relative to the last hour; zeros (float32) not drawn
    ax.semilogy(t, pos(z['g']), color=BLUE, lw=3.5, label='RNN, trained')
    if 'g_inicial' in z.files:
        ax.semilogy(t, np.where(z['g_inicial'] > 0, z['g_inicial'] / z['g_inicial'][-1], np.nan),
                    color=BLUE, lw=1.8, ls='--', alpha=0.7, label='RNN, random weights')
    for key, c, lab in (('g_lstm', ORANGE, 'LSTM'), ('g_gru', TEAL, 'GRU')):
        if key in z.files:
            ax.semilogy(t, pos(z[key]) if False else np.where(z[key] > 0, z[key] / z[key][-1], np.nan), color=c, lw=3, label=lab)
    ax.set_xlabel('hour $t$ of the window  (the prediction is made at $t = T$)')
    ax.set_ylabel('$\\Vert \\partial L / \\partial \\mathbf{x}_t \\Vert$, relative to the last hour')
    ax.set_xlim(1, T); ax.grid(alpha=0.3, which='major')
    ceros = np.where(z['g'] == 0)[0]
    if len(ceros):                                                   # hours whose gradient is exactly 0 in float32
        ax.axvspan(0.5, ceros[-1] + 1.5, color=GRAY, alpha=0.15, lw=0)
        ax.text((ceros[-1] + 2) / 2, 0.5, f'first {len(ceros)} hours:\nexactly 0\nin float32', transform=ax.get_xaxis_transform(),
                ha='center', va='center', color=NAVY, fontsize=14)
    top = ax.secondary_xaxis('top', functions=(lambda x: T - x, lambda x: T - x))
    top.set_xlabel('hours before the prediction', color=GRAY)
    ax.legend(frameon=False, loc='lower right')
    save(fig, 's05_f03_rnn_gradient_time.png')


# ---------------------------------------------------------------- F11 · gradient through time, RNN vs LSTM vs GRU
def f11():
    p = need('gradiente_tiempo_lstm_gru.npz')
    if p is None:
        return
    z = np.load(p)
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.11)
    T = len(z['g_rnn']); t = np.arange(1, T + 1)
    for key, c, lab in (('g_rnn', BLUE, 'RNN'), ('g_lstm', ORANGE, 'LSTM'), ('g_gru', TEAL, 'GRU')):
        g = z[key]
        ax.semilogy(t, np.where(g > 0, g / g[-1], np.nan), color=c, lw=3.5, label=lab)
    ceros = np.where(z['g_rnn'] == 0)[0]
    if len(ceros):
        ax.axvspan(0.5, ceros[-1] + 1.5, color=GRAY, alpha=0.15, lw=0)
        ax.text((ceros[-1] + 2) / 2, 0.5, f'RNN: first {len(ceros)} hours\nexactly 0', transform=ax.get_xaxis_transform(),
                ha='center', va='center', color=NAVY, fontsize=14)
    ax.set_xlabel('hour $t$ of the window  (the prediction is made at $t = T$)')
    ax.set_ylabel('$\\Vert \\partial L / \\partial \\mathbf{x}_t \\Vert$, relative to the last hour')
    ax.set_xlim(1, T); ax.grid(alpha=0.3)
    top = ax.secondary_xaxis('top', functions=(lambda x: T - x, lambda x: T - x))
    top.set_xlabel('hours before the prediction', color=GRAY)
    ax.legend(frameon=False, loc='lower right', fontsize=17)
    save(fig, 's05_f11_gradient_rnn_lstm_gru.png')


# ---------------------------------------------------------------- F12 · three ways to leak the future
def f12():
    p = need('resultados_jena_51.json')
    if p is None:
        return
    r = json.load(open(p))
    if 'test' not in r.get('mlp_barajado', {}):
        print('skip F12: re-run notebook 5.1 (resultados_jena_51.json lacks mlp_barajado.test)')
        return
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    cols = [0.35, 5.92, 11.49]; cw = 4.9
    titles = ['1 · shuffle, then split', '2 · normalize with all of it', '3 · cross a cut']
    notes = ['neighboring windows share 119 of 120 hours:\none lands in train, its twin in validation',
             '$\\mu$ and $\\sigma$ of the future years enter\nthe inputs the model trains on',
             'the input lives in train,\nthe target in validation']
    for x0, t, n in zip(cols, titles, notes):
        ax.add_patch(FancyBboxPatch((x0, 3.05), cw, 6.0, boxstyle='round,pad=0.02,rounding_size=0.15',
                                    fc='white', ec='#c9d3e0', lw=1.3))
        ax.text(x0 + 0.25, 8.62, t, fontsize=19, weight='bold', color=NAVY, va='center')
        ax.text(x0 + cw / 2, 3.62, n, fontsize=14, color=NAVY, ha='center', va='center', linespacing=1.3)

    # panel 1: overlapping windows, randomly assigned
    x0 = cols[0]; rng = np.random.default_rng(3)
    lab = ['train', 'val', 'train', 'train', 'val', 'train', 'val']
    for k, l in enumerate(lab):
        y = 7.75 - k * 0.47
        a = x0 + 0.45 + k * 0.22
        c = BLUE if l == 'train' else TEAL
        ax.plot([a, a + 3.0], [y, y], color=c, lw=9, solid_capstyle='butt')
        ax.text(a + 3.12, y, l, color=c, fontsize=13, va='center')
    ax.annotate('', xy=(x0 + 0.45, 4.35), xytext=(x0 + cw - 0.3, 4.35),
                arrowprops=dict(arrowstyle='<-', color=GRAY, lw=1.3))
    ax.text(x0 + cw - 0.3, 4.55, 'time', color=GRAY, fontsize=12, ha='right')

    # panel 2: schematic series with a trend; mean of train vs mean of all
    x0 = cols[1]; xs = np.linspace(0, 1, 300)
    ys = 0.55 * xs + 0.12 * np.sin(2 * np.pi * 6 * xs) + 0.03 * np.random.default_rng(0).standard_normal(300)
    X = x0 + 0.4 + xs * (cw - 0.8); Y = 5.0 + ys * 3.2
    cut = x0 + 0.4 + 0.5 * (cw - 0.8)
    ax.axvspan(0, 0, color='w')
    ax.fill_between([x0 + 0.4, cut], 4.55, 7.95, color=BLUE, alpha=0.08, lw=0)
    ax.fill_between([cut, x0 + cw - 0.4], 4.55, 7.95, color=ORANGE, alpha=0.08, lw=0)
    ax.text((x0 + 0.4 + cut) / 2, 7.75, 'train', color=BLUE, fontsize=13, ha='center')
    ax.text((cut + x0 + cw - 0.4) / 2, 7.75, 'validation · test', color=ORANGE, fontsize=13, ha='center')
    ax.plot(X, Y, color=GRAY, lw=1.6)
    m_tr, m_all = Y[xs <= 0.5].mean(), Y.mean()
    ax.plot([x0 + 0.4, cut], [m_tr, m_tr], color=BLUE, lw=2.6, ls='--')
    ax.text(x0 + 0.45, 5.0 - 0.05, '$\\mu$ of train: correct', color=BLUE, fontsize=13, va='top',
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.plot([x0 + 0.4, x0 + cw - 0.4], [m_all, m_all], color=RED, lw=2.6, ls=':')
    ax.text(x0 + 0.45, 7.35, '$\\mu$ of all: leaks', color=RED, fontsize=13, ha='left',
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1))
    ax.annotate('', xy=(x0 + 1.2, m_all + 0.05), xytext=(x0 + 1.2, 7.25), arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
    ax.text(x0 + cw / 2, 4.32, 'schematic series with a trend', color=GRAY, fontsize=11.5, ha='center', style='italic')

    # panel 3: the cut
    x0 = cols[2]; cut = x0 + 2.9
    ax.fill_between([x0 + 0.3, cut], 4.4, 7.95, color=BLUE, alpha=0.08, lw=0)
    ax.fill_between([cut, x0 + cw - 0.3], 4.4, 7.95, color=TEAL, alpha=0.08, lw=0)
    ax.plot([cut, cut], [4.4, 7.95], color=NAVY, lw=1.6, ls='--')
    ax.text((x0 + 0.3 + cut) / 2, 7.7, 'train', color=BLUE, fontsize=13, ha='center')
    ax.text((cut + x0 + cw - 0.3) / 2, 7.7, 'validation', color=TEAL, fontsize=13, ha='center')
    for y, a, ok in ((6.75, x0 + 0.45, True), (5.35, x0 + 1.55, False)):
        c = BLUE if ok else RED
        b = a + 1.25
        tg = b + 0.75
        ax.plot([a, b], [y, y], color=c, lw=9, solid_capstyle='butt')
        ax.plot([b, tg], [y, y], color=ORANGE if ok else RED, lw=1.4, ls=':')
        ax.plot([tg], [y], 'o' if ok else 'X', color=ORANGE if ok else RED, ms=13)
        ax.text(a, y - 0.42, 'allowed' if ok else 'not allowed', color=c, fontsize=12.5)

    # bottom: leak 1 measured
    ax.text(0.35, 2.55, 'Leak 1, measured  ·  flattened MLP on Jena, notebook 5.1  ·  MAE in °C, lower is better',
            fontsize=15, color=NAVY, weight='bold')
    lo, hi = 2.2, 2.75
    xa, xb = 4.1, W - 0.6
    sx = lambda v: xa + (v - lo) / (hi - lo) * (xb - xa)
    rows = [(1.75, 'validation  (what you report)', [(r['mlp_barajado']['val_barajada'], 'shuffled split', RED),
                                                     (r['mlp']['val'], 'temporal split', BLUE),
                                                     (r['base']['val'], 'baseline', GRAY)]),
            (0.85, 'the future  (what you get)', [(r['mlp_barajado']['test'], 'shuffled split', RED),
                                                  (r['mlp']['test'], 'temporal split', BLUE),
                                                  (r['base']['test'], 'baseline', GRAY)])]
    for y, name, pts in rows:
        ax.plot([xa, xb], [y, y], color='#dfe5ee', lw=1.2)
        ax.text(0.35, y, name, fontsize=14, color=NAVY, va='center')
        for k, (v, l, c) in enumerate(pts):
            ax.plot([sx(v)], [y], 'o', color=c, ms=13 if c != GRAY else 11, mfc=c if c != GRAY else 'white', mew=2)
            up = (k == 2) if y < 1.3 else (k != 1)
            ax.text(sx(v), y + (0.3 if up else -0.36), f'{l} {v:.2f}', color=c, fontsize=12.5, ha='center', va='center')
    for v in np.arange(2.2, 2.76, 0.1):
        ax.text(sx(v), 0.25, f'{v:.1f}', color=GRAY, fontsize=11, ha='center')
    save(fig, 's05_f12_three_leaks.png')


# ---------------------------------------------------------------- F13 · which characters a small h_T loses
def f13():
    p = need('resultados_seq2seq.json')
    if p is None:
        return
    r = json.load(open(p))
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14)
    chars = list('2026-10-19'); pos = np.arange(10)
    for a, b, c, lab in ((-0.5, 3.5, BLUE, 'year'), (4.5, 6.5, TEAL, 'month'), (7.5, 9.5, ORANGE, 'day')):
        ax.axvspan(a, b, color=c, alpha=0.07, lw=0)
        ax.text((a + b) / 2, 1.075, lab, color=c, ha='center', fontsize=17, weight='bold')
    claves = [k for k in r if k.startswith('sin atención')]
    claves.sort(key=lambda k: int(k.split('d=')[1]))
    ramp = mpl.colormaps['Blues'](np.linspace(0.95, 0.4, len(claves)))
    for k, c in zip(claves, ramp):
        d = int(k.split('d=')[1])
        ax.plot(pos, r[k]['por_posicion'], 'o-', color=c, lw=2.6, ms=8,
                label=f'without attention, $d = {d}$  ·  exact {100 * r[k]["exacto"]:.1f} %')
    k = 'con atención, d=24'
    ax.plot(pos, r[k]['por_posicion'], 's-', color=ORANGE, lw=3.5, ms=9,
            label=f'with attention, $d = 24$  ·  exact {100 * r[k]["exacto"]:.1f} %')
    ax.set_xticks(pos, chars, fontsize=22, family='DejaVu Sans Mono')
    ax.set_xlabel('output character  $\\tau$  (the example date 2026-10-19)', labelpad=10)
    ax.set_ylabel('accuracy at this character')
    ax.set_ylim(0, 1.04); ax.set_xlim(-0.5, 9.5); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.legend(frameon=True, facecolor='white', edgecolor='#dfe5ee', framealpha=0.95, loc='lower left', fontsize=13.5)
    fig.text(0.08, 0.035, 'Notebook 5.5, 6 000 test sentences, one training run per model. As $\\mathbf{h}_T$ shrinks, the day goes first'
             ' (read first, written last); at $d = 24$ the month goes too.', fontsize=13.5, color=GRAY)
    save(fig, 's05_f13_bottleneck_by_character.png')


if __name__ == '__main__':
    for f in (f02, f03, f07, f09, f11, f12, f13):
        f()
