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


if __name__ == '__main__':
    for f in (f02, f03, f07, f09, f11):
        f()
