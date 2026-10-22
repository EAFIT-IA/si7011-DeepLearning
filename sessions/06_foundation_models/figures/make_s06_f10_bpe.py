"""S06-F10: byte-pair encoding on a Spanish sentence.

The corpus is the Spanish markdown of the course notebooks (sessions/*/notebooks/*.ipynb), so the figure
needs no download. Top: BPE by hand (60 merges counted over the corpus words) applied to one sentence.
Bottom: byte-level BPE trained with the tokenizers library at three vocabulary sizes, measured in
characters per token on held-out notebooks, and the sentence split by the largest one.

Run from this folder:  python make_s06_f10_bpe.py ../../.. .
"""
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
from matplotlib import pyplot as plt
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('../../..')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.')
FRASE = 'Me senté en el banco del parque a leer el periódico'
N_MERGES = 60
VOCABS = (256, 1000, 4000)
BLUE = '#1f6fb4'
ORANGE = '#d9631e'
NAVY = '#14213d'
TEAL = '#167d78'
GRAY = '#8a94a3'
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'mathtext.fontset': 'dejavusans',
                     'text.color': NAVY, 'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY})


def corpus():
    textos = []
    for f in sorted(glob.glob(str(ROOT / 'sessions/*/notebooks/*.ipynb'))):
        nb = json.load(open(f, encoding='utf-8'))
        t = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'markdown')
        t = re.sub(r'\$\$.*?\$\$', ' ', t, flags=re.S)
        t = re.sub(r'\$[^$\n]*\$', ' ', t)
        t = re.sub(r'`[^`]*`', ' ', t)
        t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)
        t = re.sub(r'<[^>]+>|[#*|>_]', ' ', t)
        textos.append(t)
    return textos


def merges_a_mano(texto, n):
    palabras = Counter(tuple(w) for w in texto.split())
    fusiones = []
    for _ in range(n):
        c = Counter()
        for s, f in palabras.items():
            for a, b in zip(s, s[1:]):
                c[a, b] += f
        par = c.most_common(1)[0][0]
        fusiones.append(par)
        nuevo = Counter()
        for s, f in palabras.items():
            nuevo[tuple(_fusionar(list(s), par))] += f
        palabras = nuevo
    return fusiones


def datos():
    textos = corpus()
    corte = int(0.85 * len(textos))
    train, val = '\n'.join(textos[:corte]), '\n'.join(textos[corte:])
    z = {'frase': FRASE, 'n_chars': len(train), 'fusiones_mini': merges_a_mano(train, N_MERGES), 'vocabularios': {}}
    for v in VOCABS:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator([train], trainers.BpeTrainer(vocab_size=v, show_progress=False,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        n = len(tok.encode(val).ids)
        piezas = [tok.decode([i]) for i in tok.encode(FRASE).ids]
        z['vocabularios'][str(v)] = {'chars_por_token': len(val) / n, 'frase': piezas}
        print(f'|V| = {v:5d}: {len(val) / n:.2f} characters per token · ' + ' | '.join(piezas))
    return z


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
        w = max(len(p), 1) * 0.8 + 0.45
        ax.add_patch(plt.Rectangle((x, y - alto / 2), w, alto, facecolor=fill, edgecolor=color, lw=1.2))
        ax.text(x + w / 2, y, p.replace(' ', '·'), ha='center', va='center', fontsize=fs, color=NAVY)
        x += w + gap
    return x


def f10(z):
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
    ax.text(0, 4.95, f'BPE by hand, merges counted on the text of the course notebooks ({z["n_chars"]:,} characters); spaces split words here', fontsize=15, color=GRAY)
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
    ax3.set_xlim(0, 51)
    ax2.text(0, -0.26, 'below 1 at |V| = 256: é, ó take two bytes each', transform=ax2.transAxes, fontsize=12, color=GRAY); ax3.set_ylim(-1.5, 1.6)
    v_ref = V[-1]
    piezas = z['vocabularios'][str(v_ref)]['frase']
    ax3.text(0, 1.25, f'byte-level BPE, |V| = {v_ref:,}: {len(piezas)} tokens  (· = a space glued to the token)',
             fontsize=14, color=GRAY)
    fila, x0 = [], 0
    lineas = [[]]
    for p in piezas:                                   # wrap into two lines
        if sum(len(q) * 0.8 + 0.51 for q in lineas[-1]) + len(p) * 0.8 > 50.5:
            lineas.append([])
        lineas[-1].append(p)
    for r, l in enumerate(lineas[:3]):
        _chips(ax3, 0.55 - r * 0.8, l, x0=0, fs=13, color=TEAL, fill='#dff4ef', alto=0.5)
    fig.savefig(OUT / 's06_f10_bpe.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f10_bpe.png')


if __name__ == '__main__':
    f10(datos())
