"""
Figuras conceptuales de S04 que se calculan, no se dibujan a mano (ver figures.md):

    F14  cambio de coordenadas: anillos en x, una recta en h            sin datos
    F15  autoencoder lineal = PCA: ejemplo 2-D y cosenos en Fashion-MNIST   Fashion-MNIST (.gz originales)
    F16  tareas pretexto hechas a mano y sus atajos                     una imagen de CIFAR-10
    F17  el atajo del color: emparejar vistas por histograma             imágenes de CIFAR-10 de los notebooks
    F18  colapso sin negativos                                          sin datos (encoder de juguete)
    F19  tres maneras de leer una representación                        sin datos

Las aumentaciones de F17 son las del notebook 4.4 (copiadas abajo). Los números de F15, F17 y F18
salen de correr este script; ninguno se escribe a mano.

Uso:
    python make_s04_figuras_conceptos.py --fmnist DIR_GZ --cifar DIR_NPZ [--solo F14 F18]
    DIR_GZ: train-images-idx3-ubyte.gz y t10k-images-idx3-ubyte.gz
    DIR_NPZ: recuperacion_simclr.npz y vistas_ejemplo.npz (salidas de los notebooks 4.5 y 4.4)
Salida: s04_fnn_*.png en esta carpeta, 1672 × 941 px.
"""
import argparse
import gzip
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

AQUI = Path(__file__).resolve().parent
NAVY, BLUE, TEAL, ORANGE, RED, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#c0392b', '#8a94a3'
LBLUE, LTEAL, LGRAY, LORANGE = '#d6e6f5', '#d3ebe9', '#eceff3', '#fbe3d4'
PALETA_6 = ['#1f6fb4', '#d9631e', '#2e8b57', '#c0392b', '#7b4fa0', '#b8a019']
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'text.color': NAVY,
                     'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY, 'axes.edgecolor': NAVY,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavuserif'})


def lienzo():
    return plt.figure(figsize=(16.72, 9.41), dpi=100)


def guardar(fig, nombre):
    fig.savefig(AQUI / nombre, dpi=100)
    plt.close(fig)
    print('escrita', nombre)


def limpiar(ax, ejes=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if not ejes:
        ax.axis('off')


def flecha_fig(fig, x0, y0, x1, y1, texto=None, color=NAVY, dy=0.025):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle='-|>',
                                   mutation_scale=22, color=color, lw=2))
    if texto:
        fig.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, texto, ha='center', va='bottom', fontsize=20, color=color)


# ------------------------------------------------------------------------------------ F14
def f14():
    rng = np.random.default_rng(0)
    n = 220
    r = np.concatenate([np.clip(rng.normal(0.9, 0.18, n), 0.4, 1.3), np.clip(rng.normal(2.1, 0.2, n), 1.7, 2.7)])
    a = rng.uniform(-np.pi, np.pi, 2 * n)
    y = np.repeat([0, 1], n)
    X = np.c_[r * np.cos(a), r * np.sin(a)]
    H = np.c_[np.linalg.norm(X, axis=1), np.degrees(np.arctan2(X[:, 1], X[:, 0]))]
    fig = lienzo()
    a1 = fig.add_axes([0.07, 0.20, 0.33, 0.66]); a2 = fig.add_axes([0.64, 0.20, 0.31, 0.66])
    for c, col, nombre in [(0, BLUE, 'class A'), (1, ORANGE, 'class B')]:
        a1.scatter(*X[y == c].T, s=22, color=col, alpha=0.85, lw=0, label=nombre)
        a2.scatter(*H[y == c].T, s=22, color=col, alpha=0.85, lw=0)
    for ang in (20, 75, 140):                                    # rectas que fallan
        t = np.radians(ang); d = np.array([np.cos(t), np.sin(t)])
        p = np.array([0.3, -0.2]) + np.linspace(-4, 4, 2)[:, None] * d
        a1.plot(p[:, 0], p[:, 1], color=GRAY, ls='--', lw=1.4)
    a1.set_xlim(-3.2, 3.2); a1.set_ylim(-3.2, 3.2); a1.set_aspect('equal')
    a1.set_xlabel('$x_1$', fontsize=22); a1.set_ylabel('$x_2$', fontsize=22, rotation=0, labelpad=16)
    a2.axvline(1.5, color=NAVY, lw=2.2)
    a2.set_xlim(0, 3.2); a2.set_ylim(-190, 190); a2.set_yticks([-180, -90, 0, 90, 180])
    a2.set_xlabel(r'$h_1 = \Vert \mathbf{x} \Vert$', fontsize=22)
    a2.set_ylabel(r'$h_2$ = angle of $\mathbf{x}$ (°)', fontsize=20)
    for ax in (a1, a2):
        limpiar(ax); ax.tick_params(labelsize=14)
    a1.legend(loc='upper right', fontsize=16, frameon=False, handletextpad=0.2, bbox_to_anchor=(1.12, 1.08))
    fig.text(0.235, 0.07, 'input space: no line separates the classes', ha='center', fontsize=19)
    fig.text(0.795, 0.07, 'representation: one line does', ha='center', fontsize=19)
    flecha_fig(fig, 0.43, 0.53, 0.545, 0.53, r'$\mathbf{h} = g(\mathbf{x})$')
    fig.text(0.4875, 0.47, 'same points,\nnew coordinates', ha='center', va='top', fontsize=15, color=GRAY)
    guardar(fig, 's04_f14_change_of_coordinates.png')


# ------------------------------------------------------------------------------------ F15
def leer_idx(ruta):
    with gzip.open(ruta, 'rb') as f:
        datos = f.read()
    n = int.from_bytes(datos[4:8], 'big')
    return np.frombuffer(datos, np.uint8, offset=16).reshape(n, 784)


def f15(dir_fmnist):
    # (a) ejemplo 2-D: AE lineal con k = 1 entrenado por descenso de gradiente frente al eje de PCA
    torch.manual_seed(0)
    C = torch.tensor([[1.0, 0.0], [0.0, 0.32]])
    R = torch.tensor([[math.cos(0.55), -math.sin(0.55)], [math.sin(0.55), math.cos(0.55)]])
    P = (torch.randn(300, 2) @ C @ R.T) * 1.3 + torch.tensor([0.2, 0.1])
    mu = P.mean(0)
    pc = torch.linalg.svd(P - mu, full_matrices=False)[2][0]
    g2, d2 = nn.Linear(2, 1), nn.Linear(1, 2)
    with torch.no_grad():                                        # empieza lejos del eje, para ver el giro
        d2.weight.copy_(torch.tensor([[-0.5], [0.9]])); g2.weight.copy_(torch.tensor([[-0.4, 0.6]]))
    opt = torch.optim.SGD([*g2.parameters(), *d2.parameters()], lr=0.02)
    tray = []
    for t in range(400):
        if t in (0, 10, 25, 50, 100, 399):
            tray.append(F.normalize(d2.weight.detach()[:, 0], dim=0).clone())
        L = ((d2(g2(P)) - P) ** 2).sum(1).mean()
        opt.zero_grad(); L.backward(); opt.step()
    ang = math.degrees(math.acos(abs(float(tray[-1] @ pc))))

    # (b) Fashion-MNIST, k = 32: lo mismo que el notebook 4.1, sección 3
    d = Path(dir_fmnist)
    Xtr = torch.tensor(leer_idx(d / 'train-images-idx3-ubyte.gz'), dtype=torch.float32) / 255
    Xte = torch.tensor(leer_idx(d / 't10k-images-idx3-ubyte.gz'), dtype=torch.float32) / 255
    K = 32
    m = Xtr.mean(0)
    V = torch.linalg.svd(Xtr - m, full_matrices=False)[2][:K].T
    mse_pca = (((Xte - m) @ V @ V.T + m - Xte) ** 2).mean().item()
    torch.manual_seed(0)
    g, dd = nn.Linear(784, K), nn.Linear(K, 784)
    opt = torch.optim.Adam([*g.parameters(), *dd.parameters()], lr=3e-3)
    paso = 0
    while paso < 9600:
        for b in torch.randperm(len(Xtr)).split(128):
            xb = Xtr[b]; L = ((dd(g(xb)) - xb) ** 2).mean()
            opt.zero_grad(); L.backward(); opt.step(); paso += 1
            if paso == 9600:
                break
    with torch.no_grad():
        mse_ae = ((dd(g(Xte)) - Xte) ** 2).mean().item()
    Q, _ = torch.linalg.qr(dd.weight.detach())
    cos = torch.linalg.svdvals(V.T @ Q).numpy()
    print(f'F15 · ángulo final 2-D {ang:.2f}° · MSE PCA {mse_pca:.4f} · AE lineal {mse_ae:.4f} · '
          f'{(cos > 0.9).sum()} de {K} cosenos > 0.9, mínimo {cos.min():.3f}')

    fig = lienzo()
    a1 = fig.add_axes([0.05, 0.13, 0.40, 0.76]); a2 = fig.add_axes([0.56, 0.24, 0.40, 0.54])
    a1.scatter(*P.T, s=14, color=GRAY, alpha=0.55, lw=0)
    for i, v in enumerate(tray):
        a = 0.25 + 0.75 * i / (len(tray) - 1)
        e = (mu + 3.1 * v).numpy()
        a1.annotate('', xy=e, xytext=mu.numpy(), arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=2.2, alpha=a))
    linea = mu + torch.linspace(-4.2, 4.2, 2)[:, None] * pc
    a1.plot(*linea.T, color=NAVY, ls='--', lw=2)
    a1.text(*(mu + 4.0 * pc + torch.tensor([0.1, -0.45])).numpy(), 'first principal\ncomponent', fontsize=16,
            ha='left', va='top')
    e = (mu + 3.1 * tray[0]).numpy()
    a1.text(e[0] - 0.1, e[1] + 0.15, 'decoder direction\nat step 0', fontsize=15, color=ORANGE, ha='center', va='bottom')
    a1.text(0.02, 0.02, f'linear AE with $k=1$, 400 gradient steps: ends {ang:.1f}° from the principal axis',
            transform=a1.transAxes, fontsize=15, color=GRAY)
    a1.set_aspect('equal'); a1.set_xlim(-4.5, 4.8); a1.set_ylim(-3.6, 4.2); a1.axis('off')

    x = np.arange(1, K + 1)
    a2.bar(x, cos, color=[BLUE if c > 0.9 else ORANGE for c in cos], width=0.75)
    a2.axhline(0.9, color=GRAY, ls=':', lw=1.5)
    a2.text(K + 0.6, 0.9, '0.9', va='center', fontsize=14, color=GRAY)
    a2.set_ylim(0, 1.05); a2.set_xlim(0.2, K + 0.8)
    a2.set_xticks([1, 8, 16, 24, 32]); a2.tick_params(labelsize=14)
    a2.set_xlabel('principal angle between the two 32-D subspaces', fontsize=16)
    a2.set_ylabel('cosine (1 = same direction)', fontsize=16)
    limpiar(a2)
    a2.text(0, 1.21, f'Fashion-MNIST, $k = 32$ (notebook 4.1):\n{(cos > 0.9).sum()} of {K} directions match PCA',
            transform=a2.transAxes, fontsize=18, va='bottom')
    a2.text(0, 1.08, f'test MSE: PCA {mse_pca:.4f} · linear autoencoder {mse_ae:.4f}', transform=a2.transAxes,
            fontsize=16, color=GRAY, va='bottom')
    a2.text(0, -0.19, 'the one that misses: components 32 and 33 have almost the same variance,\n'
            'so any mix of the two reconstructs equally well', transform=a2.transAxes, fontsize=15, color=GRAY,
            va='top')
    guardar(fig, 's04_f15_linear_ae_pca.png')


# ------------------------------------------------------------------------------------ CIFAR
def cargar_cifar(dir_npz):
    d = Path(dir_npz)
    z, v = np.load(d / 'recuperacion_simclr.npz'), np.load(d / 'vistas_ejemplo.npz')
    return np.concatenate([z['img_consultas'], z['img_vecinos'].reshape(-1, 32, 32, 3), v['x'].transpose(0, 2, 3, 1)])


def grande(img, lado=192):
    return np.asarray(Image.fromarray(img).resize((lado, lado), Image.BICUBIC))


# ------------------------------------------------------------------------------------ F16
def f16(dir_npz):
    img = cargar_cifar(dir_npz)[32]                              # el caballo: cielo arriba, pasto abajo
    G = grande(img)
    fig = lienzo()
    xs = [0.035, 0.28, 0.525, 0.77]
    tit = ['rotation', 'jigsaw', 'colorization', 'inpainting']
    tarea = ['predict which rotation\nwas applied', 'predict the original\nposition of each tile',
             'predict the colors\nfrom the gray image', 'predict the missing\nsquare']
    atajo = ['sky on top, grass below:\nthe horizon gives it away', 'edges and textures\ncontinue across borders',
             'grass is green, sky is pale:\nlocal texture fixes the color', 'copy the texture\nof the border']
    W = 0.195
    for k, x0 in enumerate(xs):
        fig.add_artist(FancyBboxPatch((x0, 0.06), W, 0.86, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc='none', ec='#cfd5dd', lw=1.4, zorder=0))
        fig.text(x0 + W / 2, 0.875, tit[k], ha='center', fontsize=22, weight='bold')
        fig.text(x0 + W / 2, 0.30, tarea[k], ha='center', va='top', fontsize=16, linespacing=1.25)
        fig.text(x0 + W / 2, 0.19, 'shortcut', ha='center', va='top', fontsize=14, color=ORANGE, weight='bold')
        fig.text(x0 + W / 2, 0.155, atajo[k], ha='center', va='top', fontsize=15, color=GRAY, linespacing=1.25)
    lado = 0.135; cy = 0.58                                      # imágenes cuadradas: ancho en fracción de figura
    alto = lado * 16.72 / 9.41

    def poner(x, y, im, w, gris=False):
        ax = fig.add_axes([x, y, w, w * 16.72 / 9.41]); ax.imshow(im, cmap='gray' if gris else None, vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#cfd5dd')
        return ax
    # rotación: 2 × 2
    w = 0.072
    for i, (gr, et) in enumerate([(0, '0°'), (1, '90°'), (2, '180°'), (3, '270°')]):
        cx, cyy = xs[0] + 0.022 + (i % 2) * (w + 0.008), 0.41 + (1 - i // 2) * (w * 16.72 / 9.41 + 0.035)
        ax = poner(cx, cyy, np.rot90(G, gr), w)
        ax.set_title('?', fontsize=15, color=GRAY, pad=3)
    # rompecabezas: 3 × 3 barajado
    rng = np.random.default_rng(4); orden = rng.permutation(9); t = 192 // 3; w = 0.044; sep = 0.004
    for i, o in enumerate(orden):
        f, c = divmod(o, 3)
        tile = G[f * t:(f + 1) * t, c * t:(c + 1) * t]
        fi, ci = divmod(i, 3)
        ax = poner(xs[1] + 0.025 + ci * (w + sep), 0.40 + (2 - fi) * (w * 16.72 / 9.41 + sep), tile, w)
        ax.text(4, 12, str(o + 1), fontsize=12, color='white', weight='bold', va='top')
    # colorear
    gris = (0.299 * G[..., 0] + 0.587 * G[..., 1] + 0.114 * G[..., 2]).astype(np.uint8)
    poner(xs[2] + (W - lado) / 2, cy - alto / 2 + 0.02, gris, lado, gris=True)
    # rellenar
    hueco = G.copy(); hueco[56:136, 56:136] = 160
    ax = poner(xs[3] + (W - lado) / 2, cy - alto / 2 + 0.02, hueco, lado)
    ax.add_patch(Rectangle((56, 56), 80, 80, fill=False, ec=ORANGE, lw=2.5))
    fig.text(0.5, 0.012, 'each task worked a little; the network learns the shortcut first, and the shortcut is not the image',
             ha='center', va='bottom', fontsize=16, color=GRAY)
    guardar(fig, 's04_f16_pretext_tasks.png')


# ------------------------------------------------------------------------------------ F17
# aumentaciones del notebook 4.4 (sin la normalización final)
def recorte_espejo(x, escala=(0.2, 1.0), razon=(3 / 4, 4 / 3)):
    B = len(x)
    area = torch.empty(B).uniform_(*escala)
    r = torch.exp(torch.empty(B).uniform_(math.log(razon[0]), math.log(razon[1])))
    w, h = (area * r).sqrt().clamp(max=1), (area / r).sqrt().clamp(max=1)
    cx = (torch.rand(B) * 2 - 1) * (1 - w)
    cy = (torch.rand(B) * 2 - 1) * (1 - h)
    espejo = torch.where(torch.rand(B) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = w * espejo, cx, h, cy
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def gris_t(x):
    return (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).expand_as(x)


def color_jitter(x, b=0.4, c=0.4, s=0.4, hue=0.1, p=0.8):
    B = len(x); u = lambda a: torch.empty(B, 1, 1, 1).uniform_(1 - a, 1 + a)
    y = x * u(b)
    m = gris_t(y).mean((1, 2, 3), keepdim=True); y = (y - m) * u(c) + m
    gy = gris_t(y); y = (y - gy) * u(s) + gy
    yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    ang = torch.empty(B).uniform_(-hue, hue) * 2 * math.pi
    rot = torch.zeros(B, 3, 3); rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = ang.cos(), -ang.sin(), ang.sin(), ang.cos()
    M = torch.linalg.inv(yiq) @ rot @ yiq
    y = torch.einsum('bij,bjhw->bihw', M, y).clamp(0, 1)
    return torch.where(torch.rand(B, 1, 1, 1) < p, y, x)


def vista(xb, color=True):
    x = recorte_espejo(xb.float() / 255)
    if color:
        x = color_jitter(x)
        x = torch.where(torch.rand(len(x), 1, 1, 1) < 0.2, gris_t(x), x)
    return x.clamp(0, 1)


def histograma(x, b=4):
    q = (x.clamp(0, 0.999) * b).long()
    idx = q[:, 0] * b * b + q[:, 1] * b + q[:, 2]
    H = torch.stack([torch.bincount(i.flatten(), minlength=b ** 3) for i in idx]).float()
    return H / H.sum(1, keepdim=True)


def f17(dir_npz):
    imgs = cargar_cifar(dir_npz)
    X = torch.tensor(imgs).permute(0, 3, 1, 2)
    torch.manual_seed(0)
    acierto = {}
    for color in (False, True):                                  # ¿la otra vista es la de histograma más parecido?
        a = []
        for _ in range(200):
            h1, h2 = histograma(vista(X, color)), histograma(vista(X, color))
            S = torch.minimum(h1[:, None], h2[None]).sum(-1)     # intersección de histogramas
            a.append((S.argmax(1) == torch.arange(len(X))).float().mean().item())
        acierto[color] = float(np.mean(a))
    n = len(X)
    print(f'F17 · {n} imágenes · sin color jitter {acierto[False]:.3f} · con color jitter {acierto[True]:.3f} · azar {1 / n:.3f}')

    torch.manual_seed(7)
    A, B = 43 - 8 + 6, 43 - 8 + 2                                # rana y pájaro de vistas_ejemplo (clases 6 y 2)
    par = X[[A, A, B]]
    vs = vista(par, color=False)
    hs = histograma(vs)
    b = 4; centros = [((i // 16 + .5) / b, ((i // 4) % 4 + .5) / b, (i % 4 + .5) / b) for i in range(64)]

    fig = lienzo()
    nombres = [r'$\tilde{\mathbf{x}}_i$  view of image 1', r'$\tilde{\mathbf{x}}_j$  other view of image 1',
               r'a view of image 2']
    for k in range(3):
        y0 = 0.67 - k * 0.29
        ax = fig.add_axes([0.04, y0, 0.12, 0.12 * 16.72 / 9.41 * 0.98])
        ax.imshow(grande((vs[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8), 128)); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set(color=TEAL if k < 2 else GRAY, lw=3)
        ah = fig.add_axes([0.20, y0 + 0.02, 0.33, 0.17])
        ah.bar(np.arange(64), hs[k].numpy(), color=centros, width=0.9, edgecolor='#b0b7c1', lw=0.3)
        ah.set_xticks([]); ah.set_yticks([]); ah.set_ylim(0, float(hs.max()) * 1.1)
        for s in ('top', 'right', 'left'):
            ah.spines[s].set_visible(False)
        ah.spines['bottom'].set_color(GRAY)
        ah.set_title(nombres[k], loc='left', fontsize=16, pad=4)
    fig.text(0.20, 0.035, 'color histogram, 4 × 4 × 4 bins of RGB; crops and flips, no color jitter', fontsize=14, color=GRAY)

    ab = fig.add_axes([0.66, 0.24, 0.31, 0.56])
    vals = [acierto[False], acierto[True], 1 / n]
    nom = ['crop + flip\n(no color jitter)', 'crop + flip\n+ color jitter\n(notebook 4.4)', 'chance']
    ab.bar([0, 1, 2], vals, color=[ORANGE, BLUE, GRAY], width=0.6)
    for i, v in enumerate(vals):
        ab.text(i, v + 0.02, f'{100 * v:.0f} %', ha='center', fontsize=20, weight='bold')
    ab.set_xticks([0, 1, 2]); ab.set_xticklabels(nom, fontsize=15)
    ab.set_ylim(0, 1.05); ab.set_yticks([0, 0.5, 1]); ab.set_yticklabels(['0', '50 %', '100 %'], fontsize=14)
    limpiar(ab)
    ab.text(-0.45, 1.24, 'Find the other view by color histogram alone', transform=ab.transData, fontsize=19, va='bottom')
    ab.text(-0.45, 1.13, f'{n} CIFAR-10 images, 2 views each, 200 random draws', transform=ab.transData, fontsize=15,
            color=GRAY, va='bottom')
    guardar(fig, 's04_f17_color_shortcut.png')


# ------------------------------------------------------------------------------------ F18
def f18():
    centros = torch.tensor([[math.cos(a), math.sin(a)] for a in torch.linspace(0, 2 * math.pi, 7)[:-1]]) * 2.0
    torch.manual_seed(1)
    prueba_c = torch.arange(6).repeat_interleave(40)
    prueba = centros[prueba_c] + 0.6 * torch.randn(240, 2)
    res = {}
    for perdida in ('nce', 'pos'):
        torch.manual_seed(0)
        g = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 2))
        opt = torch.optim.Adam(g.parameters(), lr=3e-3)
        disp = []
        for t in range(1500):
            x = centros[torch.randint(0, 6, (64,))] + 0.6 * torch.randn(64, 2)
            v1, v2 = x + torch.randn_like(x), x + torch.randn_like(x)   # vistas: la entrada + ruido
            u1, u2 = F.normalize(g(v1), dim=1), F.normalize(g(v2), dim=1)
            if perdida == 'pos':
                L = -(u1 * u2).sum(1).mean()                     # solo acercar los pares positivos
            else:
                u = torch.cat([u1, u2]); s = u @ u.T / 0.2; s.fill_diagonal_(-1e9)
                m = len(u1); obj = torch.cat([torch.arange(m, 2 * m), torch.arange(m)])
                L = F.cross_entropy(s, obj)                      # InfoNCE con τ = 0.2
            opt.zero_grad(); L.backward(); opt.step()
            with torch.no_grad():
                disp.append(F.normalize(g(prueba), dim=1).std(0).sum().item())
        with torch.no_grad():
            res[perdida] = (F.normalize(g(prueba), dim=1).numpy(), np.array(disp))
    print(f"F18 · dispersión final · InfoNCE {res['nce'][1][-1]:.3f} · solo positivos {res['pos'][1][-1]:.4f}")

    fig = lienzo()
    rng = np.random.default_rng(0)
    for k, (clave, tit) in enumerate([('nce', 'with negatives (InfoNCE)'), ('pos', 'positive pairs only')]):
        ax = fig.add_axes([0.03 + k * 0.29, 0.17, 0.26, 0.26 * 16.72 / 9.41])
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color='#cfd5dd', lw=1.5)
        U = res[clave][0]
        rr = 1 + 0.06 * prueba_c.numpy() / 5 + rng.normal(0, 0.012, len(U))   # cada grupo en su carril
        ax.scatter(U[:, 0] * rr, U[:, 1] * rr, s=26, c=[PALETA_6[c] for c in prueba_c], lw=0, alpha=0.85)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal'); ax.axis('off')
        fig.text(0.03 + k * 0.29 + 0.13, 0.86, tit, ha='center', fontsize=20, weight='bold')
        fig.text(0.03 + k * 0.29 + 0.13, 0.81, r'$\mathbf{u}$ of 240 test inputs, 6 groups',
                 ha='center', fontsize=15, color=GRAY)
    fig.text(0.305, 0.10, 'every input lands on the same point:\nthe two views always match, and $\\mathbf{u}$ says nothing',
             ha='left', va='top', fontsize=15, color=RED)
    ax = fig.add_axes([0.66, 0.20, 0.31, 0.58])
    pasos = np.arange(1, 1501)
    ax.plot(pasos, res['nce'][1], color=TEAL, lw=2.4)
    ax.plot(pasos, res['pos'][1], color=RED, lw=2.4)
    ax.text(1500, res['nce'][1][-1] + 0.06, 'with negatives', ha='right', fontsize=16, color=TEAL)
    ax.text(1500, res['pos'][1][-1] + 0.06, 'positive pairs only', ha='right', fontsize=16, color=RED)
    ax.set_xscale('log'); ax.set_xlim(1, 1500); ax.set_ylim(-0.03, 1.55)
    ax.set_xlabel('training step', fontsize=16); ax.tick_params(labelsize=14)
    limpiar(ax)
    ax.text(0, 1.14, r'The symptom: spread of $\mathbf{u}$ across inputs', transform=ax.transAxes, fontsize=19, va='bottom')
    ax.text(0, 1.06, 'sum of the per-dimension standard deviations', transform=ax.transAxes, fontsize=15,
            color=GRAY, va='bottom')
    fig.text(0.5, 0.012, 'toy shared encoder 2 → 64 → 2 on the circle; views = input + Gaussian noise; same seed, same steps',
             ha='center', va='bottom', fontsize=14, color=GRAY)
    guardar(fig, 's04_f18_collapse.png')


# ------------------------------------------------------------------------------------ F19
def f19():
    fig = lienzo()
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.3); ax.axis('off')

    def caja(x, y, w, h, fc, ec, texto, lw=2, hatch=None, color_t=NAVY, fs=18):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.4', fc=fc, ec=ec, lw=lw,
                                    hatch=hatch))
        ax.text(x + w / 2, y + h / 2, texto, ha='center', va='center', fontsize=fs, color=color_t)

    def flecha(x0, y0, x1, y1, color=NAVY, estilo='-|>', ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=16, color=color, lw=1.8,
                                     linestyle=ls))
    # leyenda
    caja(30, 50.6, 3.2, 2.0, '#f3f5f8', GRAY, '', lw=1.5, hatch='///')
    ax.text(34.0, 51.6, 'frozen', va='center', fontsize=17)
    caja(44, 50.6, 3.2, 2.0, LORANGE, ORANGE, '', lw=2)
    ax.text(48.0, 51.6, 'trained with the labels', va='center', fontsize=17)
    ax.plot([65.5, 68.5], [51.6, 51.6], color=RED, lw=2, ls=(0, (4, 2)))
    ax.text(69.2, 51.6, 'gradient', va='center', fontsize=17)

    cols = [('$k$-NN', 3), ('linear probe', 36), ('fine-tuning', 69)]
    for i, (nombre, x0) in enumerate(cols):
        ax.add_patch(FancyBboxPatch((x0, 3), 28, 44, boxstyle='round,pad=0,rounding_size=0.8', fc='white',
                                    ec='#cfd5dd', lw=1.4))
        ax.text(x0 + 14, 44.2, nombre, ha='center', fontsize=24, weight='bold')
        cx = x0 + 14
        ax.text(cx, 39.0, r'$\mathbf{x}$', ha='center', va='center', fontsize=22)
        flecha(cx, 37.8, cx, 35.6)
        g_fc, g_ec, g_h = ('#f3f5f8', BLUE, '///') if i < 2 else (LORANGE, ORANGE, None)
        caja(cx - 6, 29.6, 12, 6, g_fc, g_ec, '', lw=2.4, hatch=g_h)
        ax.text(cx, 32.6, '$g$', ha='center', va='center', fontsize=26, color=BLUE,
                bbox=dict(fc='white', ec='none', pad=1.5))
        flecha(cx, 29.4, cx, 27.0)
        ax.text(cx, 25.8, r'$\mathbf{h}\in\mathbb{R}^{512}$', ha='center', va='center', fontsize=19, color=ORANGE)
        flecha(cx, 24.4, cx, 22.0)
        if i == 0:
            caja(cx - 9, 16.2, 18, 5.6, LGRAY, GRAY, 'vote of the $k$ nearest\nlabeled $\\mathbf{h}$', lw=1.6, fs=16)
        else:
            caja(cx - 6, 16.2, 12, 5.6, LORANGE, ORANGE, 'linear $W$', lw=2.2)
        flecha(cx, 16.0, cx, 13.8)
        ax.text(cx, 12.6, r'$\hat y$', ha='center', va='center', fontsize=22)
        if i == 2:                                               # el gradiente vuelve hasta g
            ax.add_patch(FancyArrowPatch((cx + 8.0, 17.5), (cx + 8.0, 33.0), arrowstyle='-|>', mutation_scale=16,
                                         color=RED, lw=2, linestyle=(0, (4, 2))))
        if i == 1:
            ax.add_patch(FancyArrowPatch((cx + 8.0, 13.5), (cx + 8.0, 21.5), arrowstyle='-|>', mutation_scale=16,
                                         color=RED, lw=2, linestyle=(0, (4, 2))))
    filas = [('trains', ['nothing', '5 130 weights', 'all 11.2 M + 5 130']),
             ('measures', ['neighborhoods in $\\mathbf{h}$', 'linear decodability', '$g$ as a starting point']),
             ('cost', ['embed once, then search', 'seconds on fixed $\\mathbf{h}$', 'a full training run'])]
    for j, (et, vals) in enumerate(filas):
        y = 9.4 - j * 2.6
        for i, (_, x0) in enumerate(cols):
            ax.text(x0 + 1.5, y, et, ha='left', va='center', fontsize=14, color=GRAY)
            ax.text(x0 + 8.0, y, vals[i], ha='left', va='center', fontsize=16)
    guardar(fig, 's04_f19_three_readings.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmnist', default=None)
    ap.add_argument('--cifar', default=None)
    ap.add_argument('--solo', nargs='*', default=None)
    a = ap.parse_args()
    quiere = lambda f: a.solo is None or f in a.solo
    if quiere('F14'): f14()
    if quiere('F15'):
        f15(a.fmnist) if a.fmnist else print('F15 saltada: falta --fmnist')
    if quiere('F16'):
        f16(a.cifar) if a.cifar else print('F16 saltada: falta --cifar')
    if quiere('F17'):
        f17(a.cifar) if a.cifar else print('F17 saltada: falta --cifar')
    if quiere('F18'): f18()
    if quiere('F19'): f19()
