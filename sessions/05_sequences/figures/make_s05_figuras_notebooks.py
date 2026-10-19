"""S05 figures drawn from what the notebooks save, so the slide and the notebook always agree.

Run from this folder after running the notebooks:
    python make_s05_figuras_notebooks.py ../notebooks .

| Figure | Reads (saved by)                                       |
|--------|--------------------------------------------------------|
| F02    | particion_jena.json        (notebook 5.1)              |
| F03    | gradiente_tiempo_rnn.npz   (notebook 5.2)              |
| F07    | vecinos_embedding.json     (notebook 5.4)              |
| F09    | atencion_fechas.npz        (notebook 5.5, section 6)   |

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
    # expected keys: horas (list of hourly timestamps as strings or indices), temp (list), cortes [i_val, i_test],
    # T (window, hours), H_fut (horizon, hours), ventanas (list of window start indices for the inset)
    t, temp = np.arange(len(d['temp'])), np.asarray(d['temp'])
    i_val, i_test = d['cortes']
    T, Hf = d['T'], d['H_fut']
    fig, (ax, axz) = plt.subplots(2, 1, figsize=(W, H), gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.35})
    for (a, b, c, lab) in [(0, i_val, BLUE, 'train 50 %'), (i_val, i_test, TEAL, 'validation 25 %'),
                           (i_test, len(t), ORANGE, 'test 25 %')]:
        ax.axvspan(a, b, color=c, alpha=0.12, lw=0)
        ax.text((a + b) / 2, temp.max() + 2, lab, ha='center', color=c, fontsize=15)
    ax.plot(t, temp, color=GRAY, lw=0.4)
    ax.set_ylabel('temperature (°C)'); ax.set_xlim(0, len(t)); ax.set_xticks([])
    z0 = i_val - 3 * (T + Hf)
    zz = slice(z0, i_val + (T + Hf))
    axz.plot(t[zz], temp[zz], color=GRAY, lw=1)
    axz.axvline(i_val, color=NAVY, lw=1.5, ls='--'); axz.text(i_val, temp[zz].max() + 1, 'train | validation', ha='center')
    for k, s0 in enumerate(d['ventanas']):
        y = temp[zz].min() - 2 - 1.6 * k
        axz.plot([s0, s0 + T], [y, y], color=BLUE, lw=5, solid_capstyle='butt')
        axz.plot([s0 + T + Hf], [y], 'o', color=ORANGE, ms=9)
    bad = i_val - T // 2
    y = temp[zz].min() - 2 - 1.6 * len(d['ventanas'])
    axz.plot([bad, bad + T], [y, y], color=RED, lw=5, solid_capstyle='butt')
    axz.plot([bad + T + Hf], [y], 'x', color=RED, ms=12, mew=3)
    axz.text(bad + T + Hf + 10, y, 'crosses the cut: not allowed', color=RED, va='center')
    axz.set_xlabel('hours'); axz.set_ylabel('temperature (°C)')
    save(fig, 's05_f02_windows_split.png')


# ---------------------------------------------------------------- F03 · gradient through time
def f03():
    p = need('gradiente_tiempo_rnn.npz')
    if p is None:
        return
    z = np.load(p)
    # expected arrays: g (mean ||dL/dx_t|| over a validation batch, t = 1..T), optional g_lstm, g_gru
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.12)
    T = len(z['g'])
    t = np.arange(1, T + 1)
    ax.semilogy(t, z['g'], color=BLUE, lw=3, label='RNN')
    for key, c, lab in (('g_lstm', ORANGE, 'LSTM'), ('g_gru', TEAL, 'GRU')):
        if key in z.files:
            ax.semilogy(t, z[key], color=c, lw=3, label=lab)
    ax.set_xlabel('input step $t$  (the prediction is made at $t = T$)')
    ax.set_ylabel('$\\lVert \\partial L / \\partial \\mathbf{x}_t \\rVert$  (log scale)')
    ax.set_xlim(1, T); ax.grid(alpha=0.3)
    if len(ax.get_lines()) > 1:
        ax.legend(frameon=False)
    save(fig, 's05_f03_rnn_gradient_time.png')


# ---------------------------------------------------------------- F07 · neighbors in the embedding
def f07():
    p = need('vecinos_embedding.json')
    if p is None:
        return
    d = json.load(open(p))
    # expected: list of {"palabra": str, "vecinos": [[str, cos], ...5], "signo": {word: +1/-1}}
    fig, ax = plt.subplots(figsize=(W, H)); ax.axis('off')
    n = len(d)
    for i, row in enumerate(d):
        y = 1 - (i + 0.5) / n
        ax.text(0.02, y, row['palabra'], fontsize=22, weight='bold', color=NAVY, va='center')
        for j, (w, c) in enumerate(row['vecinos']):
            col = BLUE if row['signo'].get(w, 0) > 0 else RED
            ax.text(0.22 + j * 0.155, y, f'{w}\n{c:.2f}', fontsize=17, color=col, va='center', ha='left')
    save(fig, 's05_f07_embedding_neighbors.png')


if __name__ == '__main__':
    for f in (f02, f03, f07, f09):
        f()
