"""S06 figures drawn from the saved outputs of the notebooks, so slide and notebook agree.

Run from this folder:  python make_s06_figuras_notebooks.py ../notebooks .
Produces (1672x941 each):
  s06_f06_masks_permutation.png   from notebook 6.1, sections 5-6 (atencion_mascaras.npz)
  s06_f10_bpe.png                 from notebook 6.3, sections 2-3 (bpe_vocabularios.json)
  s06_f13_gpt_samples.png         from notebook 6.3, sections 5-6 (muestras_gpt.json)
A figure whose input file is missing is skipped with a message. Optional third argument: the
list of figures to draw, e.g. "f10,f13".
"""
import json
import textwrap
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NB = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('../notebooks')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
SOLO = sys.argv[3].split(',') if len(sys.argv) > 3 else None
BLUE = '#1f6fb4'
ORANGE = '#d9631e'

NAVY = '#14213d'
TEAL = '#167d78'
RED = '#c0392b'
GRAY = '#8a94a3'
ORANGES = LinearSegmentedColormap.from_list('o', ['#ffffff', '#fbd3b5', '#f08a3e', '#d9631e', '#8f3a0c'])
DIVERG = mpl.colormaps['RdBu_r']

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'mathtext.fontset': 'dejavusans',
                     'text.color': NAVY, 'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY})


def f06():
    z = np.load(NB / 'atencion_mascaras.npz')
    tok = [str(t) for t in z['tokens']]
    pal = [str(t) for t in z['palabras']]
    orden = z['orden']
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)

    # top: one sentence, three masks
    titles = ['no mask', 'padding mask', 'padding + causal mask']
    mats = [z['A_sin'], z['A_pad'], z['A_causal']]
    vtop = max(m.max() for m in mats)                 # one colour scale for the three matrices
    for k, (A, t) in enumerate(zip(mats, titles)):
        ax = fig.add_axes([0.05 + k * 0.315, 0.46, 0.25, 0.43])
        ax.imshow(A, cmap=ORANGES, vmin=0, vmax=vtop)
        n = len(tok)
        ax.set_xticks(range(n), tok, rotation=55, ha='right', fontsize=13)
        ax.set_yticks(range(n), tok, fontsize=13)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', color='white', lw=2)
        ax.tick_params(which='minor', length=0)
        ax.set_title(t, fontsize=18, color=NAVY, pad=10, weight='bold')
        if k:
            masked = np.isclose(A, 0)
            for i, j in zip(*np.where(masked)):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#e3e6ec', zorder=2))
                ax.plot([j - 0.22, j + 0.22], [i - 0.22, i + 0.22], color=RED, lw=1.4, zorder=3)
                ax.plot([j - 0.22, j + 0.22], [i + 0.22, i - 0.22], color=RED, lw=1.4, zorder=3)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.05, 0.955, 'each row: who token i reads; each row sums to 1; masked cells get exactly 0',
             fontsize=15, color=GRAY)

    # bottom: output of "banco" in two orders, without and with position
    b1 = list(pal).index('banco')                 # position of banco in order 1
    b2 = int(np.where(orden == b1)[0][0])         # position of banco in order 2
    orden_txt = ' '.join(pal[i] for i in orden)
    vmax = np.abs(np.r_[z['Y1'][b1], z['Z1'][b1]]).max()
    for k, (titulo, A1, A2, col) in enumerate([('without position', z['Y1'], z['Y2'], RED),
                                                ('with position  x + p', z['Z1'], z['Z2'], TEAL)]):
        x0 = 0.05 + k * 0.48
        fig.text(x0, 0.335, titulo, fontsize=18, weight='bold', color=NAVY)
        for r, (frase, vec) in enumerate([(' '.join(pal), A1[b1]), (orden_txt, A2[b2])]):
            ax = fig.add_axes([x0 + 0.17, 0.22 - r * 0.075, 0.25, 0.05])
            ax.imshow(vec[None, :], cmap=DIVERG, vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('#c4cbd5')
            fig.text(x0, 0.245 - r * 0.075, f'"{frase}"', fontsize=13.5, color=NAVY, va='center', style='italic')
        dif = np.linalg.norm(A2[b2] - A1[b1])
        fig.text(x0, 0.085, f'output of "banco", order 1 vs order 2: difference {dif:.4f}', fontsize=15,
                 color=col, weight='bold')
    fig.text(0.05, 0.03, 'one attention layer, 4 heads, d = 64, random weights (notebook 6.1). '
             'Without position the two outputs are identical: the layer sees a set, not a sequence.',
             fontsize=13, color=GRAY)
    fig.savefig(OUT / 's06_f06_masks_permutation.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f06_masks_permutation.png')


def _fusionar(simbolos, par):
    out, i = [], 0
    while i < len(simbolos):
        if i < len(simbolos) - 1 and (simbolos[i], simbolos[i + 1]) == par:
            out.append(simbolos[i] + simbolos[i + 1]); i += 2
        else:
            out.append(simbolos[i]); i += 1
    return out


def _chips(ax, y, piezas, x0=0.0, alto=0.55, fs=15, color=BLUE, fill='#e4eefc', gap=0.06):
    """Draw tokens as chips on a line (data units: 1 = one character width); returns the end x."""
    x = x0
    for p in piezas:
        w = max(len(p), 1) * 0.62 + 0.3
        ax.add_patch(plt.Rectangle((x, y - alto / 2), w, alto, facecolor=fill, edgecolor=color, lw=1.2))
        ax.text(x + w / 2, y, p.replace(' ', '·'), ha='center', va='center', fontsize=fs, color=NAVY)
        x += w + gap
    return x


def f10():
    ruta = NB / 'bpe_vocabularios.json'
    if not ruta.exists():
        print('skip f10: no', ruta); return
    z = json.load(open(ruta, encoding='utf-8'))
    frase, fusiones = z['frase'], [tuple(p) for p in z['fusiones_mini']]
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0.03, 0.40, 0.94, 0.55]); ax.axis('off')
    ax.set_xlim(0, 80); ax.set_ylim(-0.6, 5.0)
    filas = [0, 10, 30, 60]
    for r, k in enumerate(filas):
        piezas = []
        for w in frase.split():
            s = list(w)
            for par in fusiones[:k]:
                s = _fusionar(s, par)
            piezas += s
        y = 4.2 - r * 1.25
        ax.text(0, y, f'{k} merges', ha='left', va='center', fontsize=16, color=GRAY)
        fin = _chips(ax, y, piezas, x0=7.5, fs=13)
        ax.text(fin + 0.4, y, f'{len(piezas)} symbols', ha='left', va='center', fontsize=16, color=ORANGE, weight='bold')
    ax.text(0, 4.95, 'BPE by hand on the first sentence of Don Quijote (spaces split words here)', fontsize=15, color=GRAY)
    # bottom: library BPE, vocabulary size vs characters per token
    V = sorted(int(k) for k in z['vocabularios'])
    cpt = [z['vocabularios'][str(v)]['chars_por_token'] for v in V]
    ax2 = fig.add_axes([0.06, 0.08, 0.36, 0.26])
    ax2.bar(range(len(V)), cpt, color=TEAL, width=0.6)
    for i, c in enumerate(cpt):
        ax2.text(i, c + 0.08, f'{c:.2f}', ha='center', va='bottom', fontsize=14, color=NAVY)
    ax2.set_xticks(range(len(V)), [f'|V| = {v:,}' for v in V], fontsize=13)
    ax2.set_ylabel('characters per token\n(validation text)', fontsize=13)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)
    ax3 = fig.add_axes([0.47, 0.06, 0.5, 0.28]); ax3.axis('off')
    ax3.set_xlim(0, 50); ax3.set_ylim(-1.5, 1.6)
    v_ref = 4000 if '4000' in z['vocabularios'] else V[-1]
    piezas = z['vocabularios'][str(v_ref)]['frase']
    ax3.text(0, 1.25, f'byte-level BPE, |V| = {v_ref:,}: {len(piezas)} tokens  (· = a space glued to the token)',
             fontsize=14, color=GRAY)
    fila, x0 = [], 0
    lineas = [[]]
    for p in piezas:                                   # wrap into two lines
        if sum(len(q) * 0.62 + 0.36 for q in lineas[-1]) + len(p) * 0.62 > 48:
            lineas.append([])
        lineas[-1].append(p)
    for r, l in enumerate(lineas[:3]):
        _chips(ax3, 0.55 - r * 0.8, l, x0=0, fs=13, color=TEAL, fill='#dff4ef', alto=0.5)
    fig.savefig(OUT / 's06_f10_bpe.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f10_bpe.png')


def f13():
    ruta = NB / 'muestras_gpt.json'
    if not ruta.exists():
        print('skip f13: no', ruta); return
    z = json.load(open(ruta, encoding='utf-8'))
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0.07, 0.12, 0.42, 0.78])
    ax.plot(z['pasos'], z['train'], color=BLUE, lw=2.5, label='training')
    ax.plot(z['pasos'], z['val'], color=ORANGE, lw=2.5, label='validation (last chapters)')
    ax.axhline(z['ln_V'], color=GRAY, ls='--', lw=1.5)
    ax.text(z['pasos'][-1], z['ln_V'] + 0.08, 'ln|V|: random guessing', ha='right', va='bottom', fontsize=13, color=GRAY)
    ax.set_xlabel('training step'); ax.set_ylabel('cross-entropy (nats / token)')
    ax.legend(frameon=False, fontsize=14, loc='upper right', bbox_to_anchor=(1, 0.9))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    muestras = [(f'after {k} steps', v) for k, v in sorted(z['muestras'].items(), key=lambda kv: int(kv[0]))]
    if z.get('largo'):
        muestras.append((f"long checkpoint, {z['largo']['pasos']:,} steps", z['largo']['muestra']))
    for k, (t, v) in enumerate(muestras):
        y = 0.86 - k * 0.29
        fig.text(0.54, y, t, fontsize=16, color=ORANGE if k < len(muestras) - 1 else TEAL, weight='bold', va='top')
        fig.text(0.54, y - 0.045, textwrap.fill(v.replace('\n', ' '), 62), fontsize=14.5, color=NAVY, va='top',
                 style='italic', linespacing=1.3)
    for k, (paso, _) in enumerate(sorted(z['muestras'].items(), key=lambda kv: int(kv[0]))):
        p = int(paso)
        if p in z['pasos']:
            i = z['pasos'].index(p)
            ax.plot(p, z['val'][i], 'o', ms=10, color=ORANGE, mec='white', mew=2, zorder=5)
    fig.text(0.54, 0.035, 'prompt: "En un lugar de la Mancha" · temperature 0.8 · notebook 6.3', fontsize=12, color=GRAY)
    fig.savefig(OUT / 's06_f13_gpt_samples.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f13_gpt_samples.png')


if __name__ == '__main__':
    for nombre, f in [('f06', f06), ('f10', f10), ('f13', f13)]:
        if SOLO is None or nombre in SOLO:
            f()
