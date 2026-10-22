"""S06 figures drawn from the saved outputs of the notebooks, so slide and notebook agree.

Run from this folder:  python make_s06_figuras_notebooks.py ../notebooks .
Produces (1672x941 each):
  s06_f06_masks_permutation.png   from notebook 6.1, sections 5-6 (atencion_mascaras.npz)
  s06_f13_base_model_answers.png  from notebook 6.3, section 3 (resultados_sft.json)
  s06_f18_masked_loss.png         from notebook 6.3, section 5 (resultados_sft.json)
  s06_f15_clip_zeroshot.png       from notebook 6.4, optional (clip_zeroshot.npz)
  s06_f19_summary.png             from notebook 6.5 (resultados_adaptacion.json)
F10 has its own script (make_s06_f10_bpe.py), with no notebook behind it.
A figure whose input file is missing is skipped with a message. Optional third argument: the
list of figures to draw, e.g. "f13,f18".
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


def _sft():
    ruta = NB / 'resultados_sft.json'
    if not ruta.exists():
        print('skip: no', ruta); return None
    return json.load(open(ruta, encoding='utf-8'))


def f13():
    """A base model, asked a question: the continuations of section 3 of notebook 6.3."""
    z = _sft()
    if z is None:
        return
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    preguntas = list(z['antes'])[:3]
    for k, q in enumerate(preguntas):
        texto, paro = z['antes'][q]
        y = 0.93 - k * 0.305
        fig.text(0.05, y, f'Usuario: {q} ⏎ Asistente:', fontsize=17, color=BLUE, weight='bold', va='top',
                 family='DejaVu Sans Mono')
        cuerpo = textwrap.shorten(texto.replace('\n', ' ⏎ '), 330, placeholder=' …')
        fig.text(0.05, y - 0.055, textwrap.fill(cuerpo, 118), fontsize=14.5, color=NAVY, va='top', style='italic',
                 linespacing=1.35)
        estado = 'stopped: produced <|endoftext|>' if paro else 'did not stop: cut at the 120-token limit'
        n = len(textwrap.fill(cuerpo, 118).split('\n'))
        fig.text(0.05, y - 0.07 - 0.034 * n, estado, fontsize=14, color=TEAL if paro else RED, weight='bold', va='top')
    fig.text(0.05, 0.03, f"{z['modelo']}, the base model before any fine-tuning · greedy · notebook 6.3, section 3",
             fontsize=13, color=GRAY)
    fig.savefig(OUT / 's06_f13_base_model_answers.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f13_base_model_answers.png')


def f18():
    """Per-token loss of the base model on one validation pair, coloured by role, masked positions faded."""
    z = _sft()
    if z is None:
        return
    pt = z['por_token']
    piezas, papel, L = pt['piezas'], pt['papel'], np.array(pt['L'])
    objetivo, rol = piezas[1:], np.array(papel[1:])                # L_t belongs to the target w_{t+1}
    colores = {0: GRAY, 1: BLUE, 2: ORANGE}
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0.06, 0.25, 0.9, 0.6])
    x = np.arange(len(L))
    for r in (0, 1, 2):
        sel = rol == r
        ax.bar(x[sel], L[sel], color=colores[r], alpha=1.0 if r == 2 else 0.35, width=0.75,
               edgecolor=colores[r], lw=1.2)
    m = rol == 2
    J, J_todo = L[m].mean(), L.mean()
    ax.axhline(J_todo, color=GRAY, ls='--', lw=1.8)
    ax.axhline(J, color=ORANGE, lw=2.2)
    fig.text(0.06, 0.885, f'- - -  mean over all {len(L)} positions: {J_todo:.2f}', fontsize=15, color=GRAY)
    fig.text(0.39, 0.885, f'——  masked J, answer only ({m.sum()} positions): {J:.2f}', fontsize=15, color=ORANGE,
             weight='bold')
    ax.set_xticks(x, [p.replace('\n', '⏎').replace(' ', '·') for p in objetivo], rotation=60, ha='right',
                  fontsize=12)
    for t, r in zip(ax.get_xticklabels(), rol):
        t.set_color(colores[r])
    ax.set_xlim(-0.6, len(L) - 0.4)
    ax.set_ylabel('$L_t = -\\log p(w_{t+1})$  (nats)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    for k, (r, nombre) in enumerate([(0, 'template, m = 0'), (1, 'question, m = 0'), (2, 'answer and <|endoftext|>, m = 1')]):
        fig.text(0.06 + k * 0.22, 0.93, '■ ' + nombre, color=colores[r], fontsize=15,
                 weight='bold' if r == 2 else 'normal', alpha=1.0 if r == 2 else 0.8)
    fig.text(0.06, 0.03, f"{z['modelo']} before fine-tuning, one Dolly validation pair; each bar sits on the token "
             'it predicts · notebook 6.3, section 5', fontsize=13, color=GRAY)
    fig.savefig(OUT / 's06_f18_masked_loss.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f18_masked_loss.png')


def f19():
    """Test accuracy against trainable parameters on IMDB (notebook 6.5)."""
    ruta = NB / 'resultados_adaptacion.json'
    if not ruta.exists():
        print('skip f19: no', ruta); return
    z = json.load(open(ruta, encoding='utf-8'))
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0.08, 0.13, 0.6, 0.78])
    colores = [GRAY, TEAL, BLUE, ORANGE]
    nombres = {'LSTM desde cero': 'LSTM from scratch', 'DistilBERT congelado + sonda': 'DistilBERT frozen + probe',
               'DistilBERT ajustado completo': 'DistilBERT fine-tuned', 'DistilBERT + LoRA': 'DistilBERT + LoRA'}
    for k, r in enumerate(z['filas']):
        c = colores[k % len(colores)]
        ax.scatter(r['entrenables'], r['acc_test'], s=260, color=c, zorder=3, edgecolor='white', lw=2)
        ax.annotate(f"{nombres.get(r['modelo'], r['modelo'])}\n{r['acc_test']:.1%}", (r['entrenables'], r['acc_test']),
                    xytext=(14, -6), textcoords='offset points', fontsize=15, color=c, weight='bold', va='top')
    s05 = NB.resolve().parent.parent / '05_sequences' / 'notebooks' / 'resultados_imdb_54.json'   # Session 05, 20 000 reviews
    ref = []
    if s05.exists():
        r5 = json.load(open(s05, encoding='utf-8'))
        ref = [(n, r5[k]['parametros'], r5[k]['test']) for k, n in (('bolsa', 'bag of embeddings'), ('cnn', '1-D CNN'),
                                                                     ('lstm', 'LSTM')) if k in r5]
        for n, par, acc in ref:
            ax.scatter(par, acc, s=170, marker='D', facecolor='white', edgecolor=GRAY, lw=2, zorder=3)
            ax.annotate(f'{n} {acc:.1%}', (par, acc), xytext=(14, 0), textcoords='offset points', ha='left',
                        va='center', fontsize=13, color=GRAY)
        ax.text(ref[0][1] * 0.8, max(a for *_, a in ref) + 0.012, 'Session 05, trained on 20 000 reviews', ha='left',
                fontsize=13, color=GRAY, style='italic')
    ax.set_xscale('log')
    ax.set_xlabel('trainable parameters (log scale)'); ax.set_ylabel('test accuracy')
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.05))
    lo = min([r['acc_test'] for r in z['filas']] + [a for *_, a in ref])
    ax.set_ylim(max(0.45, lo - 0.06), min(1.0, max(r['acc_test'] for r in z['filas']) + 0.04))
    xs = [r['entrenables'] for r in z['filas']]
    ax.set_xlim(min(xs) / 4, max(xs) * 40)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.3)
    y = 0.88
    fig.text(0.72, y, 'time  ·  peak memory', fontsize=15, color=GRAY)
    for k, r in enumerate(z['filas']):
        y -= 0.075
        mem = '' if r['memoria_gb'] != r['memoria_gb'] else f" · {r['memoria_gb']:.1f} GB"
        fig.text(0.72, y, f"{nombres.get(r['modelo'], r['modelo'])}: {r['segundos']:.0f} s{mem}", fontsize=14.5,
                 color=colores[k % len(colores)])
    fig.text(0.08, 0.03, f"IMDB, filled: {z['n_train']:,} training reviews, {z['n_test']:,} test reviews, "
             f"{z['max_len']} tokens (notebook 6.5)" + (" · hollow: Session 05 notebook 5.4, 20 000 training reviews, "
             "the 25 000 test reviews" if ref else ''), fontsize=13, color=GRAY)
    fig.savefig(OUT / 's06_f19_summary.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f19_summary.png')


def f15_resumen():
    """F15 without the per-image file: the EuroSAT table of notebook 6.4 (resultados_fundacionales.json)."""
    ruta = NB / 'resultados_fundacionales.json'
    if not ruta.exists():
        print('skip f15: no clip_zeroshot.npz nor', ruta); return
    z = json.load(open(ruta, encoding='utf-8'))['eurosat']
    zs, tabla = z['zero_shot'], z['tabla']
    get = lambda enc, et: next(r['accuracy'] for r in tabla if r['encoder'] == enc and r['etiquetas'] == et)
    filas = [('folder name: "AnnualCrop"', zs['nombre de la carpeta'], GRAY, 'zero-shot, 0 labels'),
             ('"a photo of annual crop land."', zs['una foto de ...'], GRAY, None),
             ('"a satellite photo of annual crop land."', zs['una foto satelital de ...'], ORANGE, None),
             ('ImageNet ResNet-18 + probe, 1 %', get('ResNet-18 ImageNet + sonda', '1%'), '#9fb7cf', 'linear probe'),
             ('ImageNet ResNet-18 + probe, 10 %', get('ResNet-18 ImageNet + sonda', '10%'), '#9fb7cf', None),
             ('CLIP image features + probe, 1 %', get('CLIP (imagen) + sonda', '1%'), TEAL, None),
             ('CLIP image features + probe, 10 %', get('CLIP (imagen) + sonda', '10%'), TEAL, None)]
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0.36, 0.12, 0.58, 0.78])
    y = np.arange(len(filas))[::-1].astype(float)
    y[3:] -= 0.6                                                # a gap between the two groups
    for yi, (nombre, acc, col, grupo) in zip(y, filas):
        ax.barh(yi, acc, color=col, height=0.7)
        ax.text(acc + 0.01, yi, f'{acc:.1%}', va='center', fontsize=16, color=NAVY, weight='bold')
    ax.set_yticks(y, [f[0] for f in filas], fontsize=14.5)
    ax.axvline(0.1, color=RED, ls='--', lw=1.5)
    ax.text(0.105, y[0] + 0.55, 'chance: 10 %', color=RED, fontsize=13, va='bottom')
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel('validation accuracy, EuroSAT (10 classes)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    fig.text(0.02, 0.93, 'CLIP zero-shot: one text per class, no training', fontsize=16, color=ORANGE, weight='bold')
    fig.text(0.02, 0.555, 'Frozen features + a linear probe, 1 % or 10 % of the labels', fontsize=16, color=TEAL,
             weight='bold')
    fig.text(0.02, 0.03, 'CLIP ViT-B/32 · probes averaged over 3 seeds · split of the S03/S04 integrators · notebook 6.4',
             fontsize=13, color=GRAY)
    fig.savefig(OUT / 's06_f15_clip_zeroshot.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f15_clip_zeroshot.png (summary from resultados_fundacionales.json)')


def f15():
    """CLIP zero-shot on one EuroSAT image (notebook 6.4, optional)."""
    ruta = NB / 'clip_zeroshot.npz'
    if not ruta.exists():
        return f15_resumen()
    z = np.load(ruta)
    clases, sim, prob, v = [str(c) for c in z['clases']], z['similitud'], z['prob'], int(z['verdadera'])
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    a0 = fig.add_axes([0.04, 0.22, 0.3, 0.55])
    a0.imshow(z['imagen'], interpolation='nearest'); a0.axis('off')
    a0.set_title(f'EuroSAT, true class: {clases[v]}', fontsize=17, color=NAVY)
    a1 = fig.add_axes([0.53, 0.1, 0.36, 0.78])
    orden = np.argsort(sim)
    k = int(np.argmax(sim))
    col = [ORANGE if i == k else (TEAL if i == v else '#b9c4d2') for i in orden]
    x0, rango = sim.min() - 0.01, sim.max() - sim.min() + 0.01
    a1.barh(range(len(orden)), sim[orden] - x0, left=x0, color=col)
    a1.set_yticks(range(len(orden)), [clases[i] for i in orden], fontsize=14)
    a1.set_xlim(x0, sim.max() + 0.45 * rango)
    for j, i in enumerate(orden):
        a1.text(sim[i] + 0.02 * rango, j, f'{sim[i]:.3f}  (p = {prob[i]:.2f})', va='center', fontsize=12, color=NAVY)
    a1.set_title('text per class: "a satellite photo of {class}."', fontsize=15, color=GRAY)
    a1.set_xlabel('cosine similarity, image vs class text')
    for sp in ('top', 'right'):
        a1.spines[sp].set_visible(False)
    fig.text(0.04, 0.08, f'prediction: {clases[k]}' + (' (correct)' if k == v else ' (wrong)'), fontsize=18,
             color=ORANGE, weight='bold')
    fig.text(0.04, 0.03, 'CLIP ViT-B/32, no training on EuroSAT · p = softmax(100 cos) · notebook 6.4', fontsize=13,
             color=GRAY)
    fig.savefig(OUT / 's06_f15_clip_zeroshot.png', dpi=100)
    plt.close(fig)
    print('wrote s06_f15_clip_zeroshot.png')


if __name__ == '__main__':
    for nombre, f in [('f06', f06), ('f13', f13), ('f15', f15), ('f18', f18), ('f19', f19)]:
        if SOLO is None or nombre in SOLO:
            f()
