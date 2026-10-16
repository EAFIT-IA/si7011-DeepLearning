"""
Figuras de S04 que salen de los datos de los notebooks (ver figures.md, "Script specifications").

    F03  reconstrucciones_ae.npz + resultados_ae.json        notebook 4.1, sección 4
    F05  vistas_ejemplo.npz                                   notebook 4.4, sección 1
    F07  similitud_n8.npz                                     notebook 4.4, sección 5
    F08  recuperacion_simclr.npz                              notebook 4.5, sección 2
    F09  resultados_eficiencia.json                           notebook 4.5, sección 4
    F10  embeddings_test.npz                                  notebook 4.5, sección 5

Ningún número se escribe a mano: todo se lee de esos archivos. Se generan las figuras cuyos datos
estén en la carpeta; las demás se saltan con un aviso.

Uso:
    python make_s04_figuras_notebooks.py --datos CARPETA_CON_LOS_ARCHIVOS [--solo F03 F09]
Salida: s04_fnn_*.png en esta carpeta, 1672 × 941 px.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

AQUI = Path(__file__).resolve().parent
NAVY, BLUE, TEAL, ORANGE, RED, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#c0392b', '#8a94a3'
PALETA_10 = ['#1f6fb4', '#d9631e', '#2e8b57', '#c0392b', '#7b4fa0', '#8c564b', '#d4579e', '#5b6577', '#b8a019', '#17a2b8']
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'text.color': NAVY,
                     'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY, 'axes.edgecolor': NAVY,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavuserif'})


def lienzo():
    return plt.figure(figsize=(16.72, 9.41), dpi=100)


def guardar(fig, nombre):
    fig.savefig(AQUI / nombre, dpi=100)
    plt.close(fig)
    print('escrita', nombre)


def marco(ax, color, lw=3):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set(color=color, linewidth=lw, visible=True)


# ---------------------------------------------------------------------------------------- F03
def f03(d):
    rec = np.load(d / 'reconstrucciones_ae.npz')
    res = json.load(open(d / 'resultados_ae.json'))['resultados']
    filas = [('original', 'original'), ('AE, $k=2$', 'AE_k=2'), ('AE, $k=8$', 'AE_k=8'),
             ('AE, $k=32$', 'AE_k=32'), ('PCA, $k=32$', 'PCA_k=32')]
    fig = lienzo()
    gs = fig.add_gridspec(5, 11, left=0.14, right=0.97, top=0.93, bottom=0.07, wspace=0.08, hspace=0.12,
                          width_ratios=[1] * 8 + [0.25, 1.15, 1.15])
    for r, (etiqueta, clave) in enumerate(filas):
        for c in range(8):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(rec[clave][c].reshape(28, 28), 0, 1), cmap='gray_r', vmin=0, vmax=1)
            marco(ax, '#d5dae1', 1)
        fig.text(0.125, 1 - (0.07 + (r + 0.5) * 0.86 / 5), etiqueta, ha='right', va='center', fontsize=22)
        if clave != 'original':
            m = res[clave.replace('_', ' ')]
            for col, val in [(9, f"{m['mse']:.4f}"), (10, f"{m['probe']:.3f}")]:
                ax = fig.add_subplot(gs[r, col]); ax.axis('off')
                ax.text(0.5, 0.5, val, ha='center', va='center', fontsize=22)
    for col, titulo in [(9, 'test MSE'), (10, 'probe acc.')]:
        ax = fig.add_subplot(gs[0, col]); ax.axis('off')
        ax.text(0.5, 0.5, titulo, ha='center', va='center', fontsize=18, color=GRAY)
    guardar(fig, 's04_f03_reconstructions_k.png')


# ---------------------------------------------------------------------------------------- F05
def f05(d):
    z = np.load(d / 'vistas_ejemplo.npz')
    x = z['x'].transpose(0, 2, 3, 1) / 255 if z['x'].shape[1] == 3 else z['x'] / 255
    fig = lienzo()
    gs = fig.add_gridspec(3, 8, left=0.08, right=0.98, top=0.95, bottom=0.05, wspace=0.12, hspace=0.12)
    for r, (imgs, color, et) in enumerate([(x, GRAY, r'$\mathbf{x}$'), (z['v1'], TEAL, r'$\tilde{\mathbf{x}}_i$'),
                                           (z['v2'], ORANGE, r'$\tilde{\mathbf{x}}_j$')]):
        for c in range(8):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(imgs[c], 0, 1), interpolation='nearest')
            marco(ax, color, 4)
        fig.text(0.05, 1 - (0.05 + (r + 0.5) * 0.90 / 3), et, ha='center', va='center', fontsize=34)
    guardar(fig, 's04_f05_augmented_views.png')


# ---------------------------------------------------------------------------------------- F07
def f07(d, fila=3):
    z = np.load(d / 'similitud_n8.npz')
    S, vistas = z['S'].astype(float), z['vistas']
    n = len(S); N = n // 2
    fig = lienzo()
    ax = fig.add_axes([0.075, 0.09, 0.50, 0.89])
    lim = np.nanmax(np.abs(S))
    im = ax.imshow(np.ma.masked_invalid(S), cmap='RdBu_r', vmin=-lim, vmax=lim)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor='#e5e8ec', edgecolor=GRAY, hatch='///', lw=0.5))
        j = (i + N) % n
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=3))
    ax.add_patch(Rectangle((-0.5, fila - 0.5), n, 1, fill=False, edgecolor=NAVY, lw=3, ls='--'))   # la fila del panel derecho
    ax.set_xticks([]); ax.set_yticks([])
    for k in range(n):                                     # miniaturas en los dos ejes
        for pos in [(k / n, -0.065, 1 / n, 0.06), (-0.065, 1 - (k + 1) / n, 0.06, 1 / n)]:
            t = ax.inset_axes(pos); t.imshow(np.clip(vistas[k], 0, 1)); t.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('$s_{ij}$', fontsize=22)
    ax.text(n / 2 - 0.5, -1.0, 'diagonal: masked ($k \\neq i$) · orange: the other view', ha='center', fontsize=15, color=GRAY)
    # una fila como softmax
    s = np.where(np.isnan(S[fila]), -np.inf, S[fila])
    p = np.exp(s - s[np.isfinite(s)].max()); p /= p.sum()
    pos = (fila + N) % n
    cand = [k for k in range(n) if k != fila]
    bx = fig.add_axes([0.70, 0.30, 0.28, 0.55])
    bx.bar(range(len(cand)), p[cand], color=[ORANGE if k == pos else GRAY for k in cand])
    bx.set_xticks([]); bx.set_ylabel('softmax over the row', fontsize=16)
    bx.spines[['top', 'right']].set_visible(False)
    bx.set_title(f'dashed row ($i = {fila}$):\nsoftmax over its {n - 1} candidates', fontsize=18)
    for k_, k in enumerate(cand):
        t = bx.inset_axes([k_ / len(cand), -0.13, 1 / len(cand), 0.11]); t.imshow(np.clip(vistas[k], 0, 1)); t.axis('off')
    fig.text(0.84, 0.13, f'$L_i = -\\log p_{{i,j(i)}} = {-np.log(p[pos]):.2f}$', ha='center', fontsize=24, color=ORANGE)
    guardar(fig, 's04_f07_similarity_matrix.png')


# ---------------------------------------------------------------------------------------- F08
def f08(d):
    z = np.load(d / 'recuperacion_simclr.npz')
    if 'img_consultas' not in z.files:
        print('F08: recuperacion_simclr.npz no trae las imágenes; vuelve a correr la sección 2 del notebook 4.5'); return
    clases = list(z['clases'])
    fig = lienzo()
    gs = fig.add_gridspec(5, 8, left=0.03, right=0.99, top=0.93, bottom=0.03, wspace=0.10, hspace=0.38,
                          width_ratios=[1, 0.15] + [1] * 6)
    for r in range(5):
        ax = fig.add_subplot(gs[r, 0]); ax.imshow(z['img_consultas'][r]); marco(ax, NAVY, 4)
        ax.set_title(clases[z['y_consultas'][r]], fontsize=15)
        for c in range(6):
            ok = z['y_vecinos'][r, c] == z['y_consultas'][r]
            ax = fig.add_subplot(gs[r, c + 2]); ax.imshow(z['img_vecinos'][r, c]); marco(ax, TEAL if ok else RED, 4)
            ax.set_title(f"{z['similitud'][r, c]:.2f}", fontsize=15)
    fig.text(0.07, 0.965, 'query', ha='center', fontsize=17, color=GRAY)
    fig.text(0.60, 0.965, '6 nearest training images by cosine similarity of $\\mathbf{h}$ · teal: same class · red: other class',
             ha='center', fontsize=17, color=GRAY)
    guardar(fig, 's04_f08_knn_retrieval.png')


# ---------------------------------------------------------------------------------------- F09
def f09(d):
    j = json.load(open(d / 'resultados_eficiencia.json'))
    x = j['por_clase']
    colores = {'aleatorio': GRAY, 'autoencoder': BLUE, 'SimCLR': TEAL, 'ImageNet': ORANGE}
    nombres = {'aleatorio': 'random $g$', 'autoencoder': 'autoencoder', 'SimCLR': 'SimCLR',
               'ImageNet': 'ImageNet (1.2 M labels)'}
    fig = lienzo()
    ax = fig.add_axes([0.08, 0.11, 0.68, 0.84])
    finales = {}
    for n, r in j['resultados'].items():
        acc = np.array([r[str(k)] for k in x])
        m = acc.mean(1)
        ax.plot(x, m, 'o--' if n == 'ImageNet' else 'o-', color=colores[n], lw=3, ms=9)
        ax.fill_between(x, acc.min(1), acc.max(1), color=colores[n], alpha=0.15, lw=0)
        finales[n] = m[-1]
    # etiquetas directas al final de cada línea, separadas al menos 0.05 para que no se monten
    orden = sorted(finales, key=finales.get, reverse=True)          # de arriba hacia abajo, sin pasar de 1
    ys = [min(finales[orden[0]], 0.98)]
    for n in orden[1:]:
        ys.append(min(finales[n], ys[-1] - 0.05))
    for n, yy in zip(orden, ys):
        ax.text(x[-1] * 1.3, yy, nombres[n], color=colores[n], fontsize=20, va='center', clip_on=False)
    ax.axhline(0.1, color=NAVY, ls=':', lw=1.5); ax.text(x[-1], 0.065, 'chance', fontsize=16, color=GRAY, ha='right')
    ax.set_xscale('log'); ax.set_xticks(x, [str(k) for k in x]); ax.minorticks_off(); ax.set_ylim(0, 1)
    ax.set_xlabel('labels per class', fontsize=22); ax.set_ylabel('test accuracy (linear probe)', fontsize=22)
    ax.tick_params(labelsize=18); ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25)
    guardar(fig, 's04_f09_label_efficiency.png')


# ---------------------------------------------------------------------------------------- F10
CIFAR = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def f10(d):
    z = np.load(d / 'embeddings_test.npz')
    clases = list(z['clases']) if 'clases' in z.files else CIFAR
    fig = lienzo()
    for k, (n, t) in enumerate([('aleatorio', 'random $g$'), ('autoencoder', 'autoencoder'), ('SimCLR', 'SimCLR')]):
        ax = fig.add_axes([0.02 + k * 0.33, 0.16, 0.30, 0.76])
        ax.scatter(*z[n].T, c=[PALETA_10[c] for c in z['y']], s=7, lw=0)
        ax.set_title(t, fontsize=24); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set(color='#d5dae1')
    manijas = [plt.Line2D([], [], marker='o', ls='', color=PALETA_10[c], ms=11) for c in range(10)]
    fig.legend(manijas, clases, loc='lower center', ncol=10, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.04),
               columnspacing=1.2, handletextpad=0.3)
    fig.text(0.985, 0.012, 't-SNE keeps neighbors, not distances', ha='right', fontsize=15, color=GRAY)
    guardar(fig, 's04_f10_tsne_embeddings.png')


FIGURAS = {'F03': (f03, ['reconstrucciones_ae.npz', 'resultados_ae.json']), 'F05': (f05, ['vistas_ejemplo.npz']),
           'F07': (f07, ['similitud_n8.npz']), 'F08': (f08, ['recuperacion_simclr.npz']),
           'F09': (f09, ['resultados_eficiencia.json']), 'F10': (f10, ['embeddings_test.npz'])}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--datos', required=True, type=Path)
    ap.add_argument('--solo', nargs='*', default=list(FIGURAS))
    a = ap.parse_args()
    for k in a.solo:
        fn, archivos = FIGURAS[k]
        faltan = [f for f in archivos if not (a.datos / f).exists()]
        if faltan:
            print(f'{k}: faltan {faltan}, se salta')
        else:
            fn(a.datos)
