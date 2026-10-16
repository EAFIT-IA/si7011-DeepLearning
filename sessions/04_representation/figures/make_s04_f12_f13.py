"""
S04-F12 (arquitecturas de los notebooks) y S04-F13 (costos de cómputo).

Los modelos se definen igual que en los notebooks 4.1-4.5; los parámetros y los FLOPs se miden con
torch.utils.flop_counter sobre una imagen, no se escriben a mano. FLOPs de entrenamiento = 3 × FLOPs
hacia adelante (adelante + atrás), multiplicado por las imágenes vistas en cada preentrenamiento.

Uso:  python make_s04_f12_f13.py          (necesita torch y torchvision; no necesita GPU ni datos)
Salida: s04_f12_architectures.png, s04_f13_compute_costs.png, 1672 × 941 px.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import resnet18

AQUI = Path(__file__).resolve().parent
NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
LBLUE, LTEAL, LGRAY = '#d6e6f5', '#d3ebe9', '#eceff3'
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'text.color': NAVY,
                     'axes.labelcolor': NAVY, 'xtick.color': NAVY, 'ytick.color': NAVY, 'axes.edgecolor': NAVY,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'mathtext.fontset': 'dejavuserif'})


# ------------------------------------------------------------------------------------ medir
def flops(f):
    with FlopCounterMode(display=False) as m:
        f()
    return m.get_total_flops()


def npar(*ms):
    return sum(p.numel() for m in ms for p in m.parameters())


def resnet18_cifar():                                   # la de los notebooks 4.4 y 4.5
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False); m.maxpool = nn.Identity(); m.fc = nn.Identity()
    return m


def medir():
    torch.manual_seed(0)
    R = {}
    # 4.1: MLP 784-256-32, 4 800 pasos × 128
    g = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 32))
    d = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 784))
    x = torch.randn(1, 784)
    R['mlp'] = dict(pg=npar(g), pd=npar(d), fg=flops(lambda: g(x)), paso=flops(lambda: d(g(x))), imgs=4800 * 128)
    # 4.2: autoencoder convolucional, 3 000 pasos × 128
    g = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                      nn.Flatten(), nn.Linear(64 * 49, 32))
    d = nn.Sequential(nn.Linear(32, 64 * 49), nn.ReLU(), nn.Unflatten(1, (64, 7, 7)),
                      nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(), nn.ConvTranspose2d(32, 1, 3, 2, 1, 1))
    x = torch.randn(1, 1, 28, 28)
    R['conv'] = dict(pg=npar(g), pd=npar(d), fg=flops(lambda: g(x)), paso=flops(lambda: d(g(x))), imgs=3000 * 128)
    # 4.3: MAE, d = 64, encoder de 4 bloques, decoder de 1, 16 patches de 7×7; 3 000 pasos × 256
    bloque = lambda: nn.TransformerEncoderLayer(64, 4, 128, 0.0, batch_first=True, norm_first=True)
    emb, head = nn.Linear(49, 64), nn.Linear(64, 49)
    enc = nn.Sequential(*[bloque() for _ in range(4)], nn.LayerNorm(64))
    dec = nn.Sequential(bloque(), nn.LayerNorm(64))

    def paso_mae(visibles):
        t = enc(emb(torch.randn(1, visibles, 49)))
        lugares = torch.zeros(1, 16, 64); lugares[:, :visibles] = t
        head(dec(lugares))
    R['mae'] = dict(pg=npar(emb, enc), pd=npar(dec, head), fg=flops(lambda: enc(emb(torch.randn(1, 16, 49)))),
                    paso75=flops(lambda: paso_mae(4)), paso25=flops(lambda: paso_mae(12)), imgs=3000 * 256)
    # 4.4-4.5: ResNet-18 para CIFAR, cabeza p, decoder del autoencoder de 4.5
    g = resnet18_cifar().eval()
    p = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128)).eval()
    d = nn.Sequential(nn.Linear(512, 256 * 16), nn.ReLU(), nn.Unflatten(1, (256, 4, 4)),
                      nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                      nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                      nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid()).eval()
    x, h = torch.randn(1, 3, 32, 32), torch.randn(1, 512)
    fg = flops(lambda: g(x))
    R['resnet'] = dict(pg=npar(g), pp=npar(p), fg=fg,
                       etapas={n: npar(getattr(g, n)) for n in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']},
                       simclr_paso=2 * (fg + flops(lambda: p(h))), simclr_imgs=30 * 50000,     # 2 vistas, 30 épocas
                       ae_paso=fg + flops(lambda: d(h)), ae_imgs=10 * 50000)
    # referencia: ResNet-18 supervisada en ImageNet, 224 × 224, 90 épocas × 1 281 167 imágenes
    gi = resnet18(weights=None).eval()
    R['imagenet'] = dict(pg=npar(gi), fg=flops(lambda: gi(torch.randn(1, 3, 224, 224))), imgs=90 * 1281167)
    return R


def legible(v, unidad=''):
    for e, s in [(1e18, 'E'), (1e15, 'P'), (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k')]:
        if v >= e:
            n = v / e
            return f'{n:.0f} {s}{unidad}' if n >= 10 else f'{n:.1f} {s}{unidad}'
    return f'{v:.0f} {unidad}'


# ------------------------------------------------------------------------------------ F12
MID = '#9cc3e6'                                                  # tokens recién proyectados
SALIDA = '#c5ccd6'                                               # patches predichos


class Fila:
    """Dibuja una fila de izquierda a derecha; self.x es el borde derecho de lo último dibujado."""
    A = 4.6                                                      # largo de cada flecha

    def __init__(self, ax, y, x=2.0):
        self.ax, self.y, self.x = ax, y, x

    def caja(self, w, h, fc, ec, abajo=None, arriba=None, dentro=None, lw=1.6, color_abajo=NAVY):
        ax, y = self.ax, self.y
        ax.add_patch(FancyBboxPatch((self.x, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.25',
                                    fc=fc, ec=ec, lw=lw))
        if abajo:
            ax.text(self.x + w / 2, y - h / 2 - 0.45, abajo, ha='center', va='top', fontsize=13, color=color_abajo)
        if arriba:
            ax.text(self.x + w / 2, y + h / 2 + 0.35, arriba, ha='center', va='bottom', fontsize=13)
        if dentro:
            ax.text(self.x + w / 2, y, dentro, ha='center', va='center', fontsize=14, linespacing=1.3)
        x0 = self.x; self.x += w
        return x0

    def mapa(self, lado, canales, fc, ec, abajo, arriba=None, alto=0.25, ancho=0.6):
        h = 2.0 + lado * alto
        w = 1.0 + ancho * float(torch.log2(torch.tensor(float(canales))))
        return self.caja(w, h, fc, ec, abajo, arriba)

    def vector(self, dim, color, abajo):
        h = 0.9 * float(torch.log2(torch.tensor(float(dim))))
        return self.caja(0.9, h, color, color, abajo, color_abajo=color)

    def tokens(self, colores, abajo, lado=0.5):
        n = len(colores); paso = lado * 1.15; alto = n * paso
        for i, c in enumerate(colores):
            self.ax.add_patch(Rectangle((self.x, self.y + alto / 2 - (i + 1) * paso), lado, lado, fc=c, ec='none'))
        self.ax.text(self.x + lado / 2, self.y - alto / 2 - 0.45, abajo, ha='center', va='top', fontsize=13)
        x0 = self.x; self.x += lado
        return x0

    def imagen(self, lado, abajo, arriba, rejilla=0):
        x0 = self.caja(lado, lado, LGRAY, GRAY, abajo, arriba)
        for k in range(1, rejilla):
            f = x0 + lado * k / rejilla
            self.ax.plot([f, f], [self.y - lado / 2, self.y + lado / 2], color='white', lw=1.6)
            self.ax.plot([x0, x0 + lado], [self.y - lado / 2 + lado * k / rejilla] * 2, color='white', lw=1.6)
        return x0

    def flecha(self, texto=None, color=NAVY, largo=None):
        L = largo or self.A
        x0, x1 = self.x + 0.4, self.x + L - 0.4
        self.ax.add_patch(FancyArrowPatch((x0, self.y), (x1, self.y), arrowstyle='-|>', mutation_scale=13,
                                          color=color, lw=1.4))
        if texto:
            self.ax.text((x0 + x1) / 2, self.y + 0.5, texto, ha='center', va='bottom', fontsize=12, color=GRAY,
                         linespacing=1.1)
        self.x += L


def llave(ax, x0, x1, y, texto, color):
    ax.plot([x0, x0, x1, x1], [y + 0.5, y, y, y + 0.5], color=color, lw=1.6)
    ax.text((x0 + x1) / 2, y - 0.4, texto, ha='center', va='top', fontsize=14, color=color)


def titulo(ax, y, texto, sub):
    t = ax.text(2.0, y, texto, ha='left', va='center', fontsize=17, weight='bold')
    ax.annotate(sub, xy=(1, 0.5), xycoords=t, xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', fontsize=14, color=GRAY)


def nota(ax, y, lineas, x=70.0):
    for i, (t, c) in enumerate(lineas):
        ax.text(x, y - 1.9 * i, t, ha='left', va='center', fontsize=14, color=c)


def f12(R):
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.3); ax.axis('off')

    # fila 1: autoencoder convolucional (4.2)
    c = R['conv']
    titulo(ax, 54.6, 'Notebook 4.2 · convolutional autoencoder', '· Fashion-MNIST')
    F = Fila(ax, 46.4); F.A = 5.6
    F.imagen(6.5, r'$1{\times}28{\times}28$', r'$\mathbf{x}$'); F.flecha('conv 3×3')
    g0 = F.mapa(14, 32, LBLUE, BLUE, r'$32{\times}14{\times}14$'); F.flecha('conv 3×3')
    F.mapa(7, 64, LBLUE, BLUE, r'$64{\times}7{\times}7$'); F.flecha('linear')
    F.vector(32, ORANGE, r'$\mathbf{h}\in\mathbb{R}^{32}$'); g1 = F.x; F.flecha('linear')
    d0 = F.mapa(7, 64, LTEAL, TEAL, r'$64{\times}7{\times}7$'); F.flecha('conv-T')
    F.mapa(14, 32, LTEAL, TEAL, r'$32{\times}14{\times}14$'); d1 = F.x; F.flecha('conv-T')
    F.imagen(6.5, r'$1{\times}28{\times}28$', r'$\hat{\mathbf{x}}$')
    llave(ax, g0, g1, 39.2, f'encoder $g$ · {legible(c["pg"])} parameters', BLUE)
    llave(ax, d0, d1, 39.2, f'decoder $d$ · {legible(c["pd"])} parameters', TEAL)
    nota(ax, 48.3, [('784 pixels → 32 numbers → 784 pixels', NAVY),
                    ('stride 2 halves the side; conv-T doubles it', GRAY)])

    # fila 2: MAE (4.3)
    m = R['mae']
    titulo(ax, 35.4, 'Notebook 4.3 · masked autoencoder', '· 16 patches of 7×7, tokens of 64')
    F = Fila(ax, 26.4); F.A = 5.0
    F.imagen(6.5, r'$28{\times}28$', r'$\mathbf{x}$', rejilla=4); F.flecha('patch\n+ linear')
    F.tokens([MID] * 16, r'$16{\times}64$'); F.flecha('keep 4')
    g0 = F.tokens([BLUE] * 4, r'$4{\times}64$'); F.flecha(largo=2.4)
    F.caja(8.0, 5.6, LBLUE, BLUE, dentro='4 transformer\nblocks', lw=2); F.flecha(largo=2.4)
    F.tokens([BLUE] * 4, r'$4{\times}64$'); g1 = F.x; F.flecha('+ 12 [M]')
    col = [GRAY] * 16
    for i in (1, 7, 8, 14):
        col[i] = BLUE
    d0 = F.tokens(col, r'$16{\times}64$'); F.flecha(largo=2.4)
    F.caja(4.0, 4.0, LTEAL, TEAL, dentro='1\nblock', lw=2); F.flecha('linear')
    F.tokens([SALIDA] * 16, r'$16{\times}49$'); d1 = F.x; F.flecha('unpatch')
    F.imagen(6.5, r'$28{\times}28$', r'$\hat{\mathbf{x}}$')
    llave(ax, g0, g1, 19.6, f'encoder $g$ · {legible(m["pg"])} parameters', BLUE)
    llave(ax, d0, d1, 19.6, f'decoder $d$ · {legible(m["pd"])} parameters', TEAL)
    nota(ax, 29.4, [('pretraining: $g$ sees 4 tokens, $d$ all 16', NAVY),
                    (r'probe: $g$ sees all 16; $\mathbf{h}\in\mathbb{R}^{64}$ is their mean', NAVY),
                    ('$d$ is small and is discarded', GRAY)])

    # fila 3: ResNet-18 para CIFAR (4.4-4.5)
    r, e = R['resnet'], R['resnet']['etapas']
    titulo(ax, 16.4, 'Notebooks 4.4–4.5 · ResNet-18', '· CIFAR-10, 3×3 stem, no max-pool')
    F = Fila(ax, 7.6, x=3.4); F.A = 4.4
    kw = dict(alto=0.2, ancho=0.45)
    F.mapa(32, 3, LGRAY, GRAY, r'$3{\times}32{\times}32$', r'$\mathbf{x}$', **kw); F.flecha('conv 3×3')
    g0 = F.mapa(32, 64, LBLUE, BLUE, r'$64{\times}32{\times}32$', **kw)
    for k, (lado, ch) in enumerate([(32, 64), (16, 128), (8, 256), (4, 512)], 1):
        F.flecha(f'stage {k}')
        F.mapa(lado, ch, LBLUE, BLUE, rf'${ch}{{\times}}{lado}{{\times}}{lado}$', legible(e[f'layer{k}']), **kw)
    F.flecha('mean')
    F.vector(512, ORANGE, r'$\mathbf{h}\in\mathbb{R}^{512}$'); g1 = F.x; F.flecha('$p$', TEAL)
    F.vector(128, TEAL, r'$\mathbf{u}\in\mathbb{R}^{128}$')
    nota(ax, 10.4, [(f'encoder $g$: {legible(r["pg"])} parameters', BLUE),
                    (f'{100 * e["layer4"] / r["pg"]:.0f} % of them in stage 4', BLUE),
                    (f'$p$: 512 → 512 → 128 · {legible(r["pp"])} · SimCLR only', TEAL),
                    ('above each stage: its parameters', GRAY)], x=F.x + 3.0)
    fig.savefig(AQUI / 's04_f12_architectures.png', dpi=100); plt.close(fig)
    print('escrita s04_f12_architectures.png')


# ------------------------------------------------------------------------------------ F13
def f13(R):
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    a1 = fig.add_axes([0.205, 0.19, 0.25, 0.68]); a2 = fig.add_axes([0.70, 0.19, 0.25, 0.68])

    filas1 = [('MLP $g$ · 4.1', R['mlp']['fg'], R['mlp']['pg'], BLUE),
              ('conv $g$ · 4.2', R['conv']['fg'], R['conv']['pg'], BLUE),
              ('MAE $g$, 16 patches · 4.3', R['mae']['fg'], R['mae']['pg'], BLUE),
              ('ResNet-18, 32×32 · 4.4–4.5', R['resnet']['fg'], R['resnet']['pg'], BLUE),
              ('ResNet-18, 224×224 · ImageNet', R['imagenet']['fg'], R['imagenet']['pg'], GRAY)]
    T = lambda e: 3 * e                                         # adelante + atrás ≈ 3 × adelante
    filas2 = [('MLP autoencoder · 4.1', T(R['mlp']['paso']) * R['mlp']['imgs'], '4 800 steps × 128', BLUE),
              ('conv autoencoder · 4.2', T(R['conv']['paso']) * R['conv']['imgs'], '3 000 steps × 128', BLUE),
              ('MAE, mask 75 % · 4.3', T(R['mae']['paso75']) * R['mae']['imgs'], '3 000 steps × 256', BLUE),
              ('MAE, mask 25 % · 4.3', T(R['mae']['paso25']) * R['mae']['imgs'], '3 000 steps × 256', BLUE),
              ('ResNet autoencoder · 4.5', T(R['resnet']['ae_paso']) * R['resnet']['ae_imgs'], '10 epochs × 50 000', BLUE),
              ('SimCLR checkpoint · 4.4', T(R['resnet']['simclr_paso']) * R['resnet']['simclr_imgs'],
               '30 epochs × 50 000 × 2 views', ORANGE),
              ('supervised ImageNet', T(R['imagenet']['fg']) * R['imagenet']['imgs'], '90 epochs × 1.28 M, labels', GRAY)]

    def barras(ax, filas, titulo, xlim, nota):
        n = len(filas)
        for i, (nombre, v, extra, color) in enumerate(filas):
            yy = n - 1 - i
            ax.barh(yy, v, height=0.56, color=color, left=xlim[0])
            ax.text(v * 1.5, yy + 0.02, legible(v, 'FLOPs'), va='center', fontsize=14, weight='bold')
            ax.text(xlim[0] / 1.6, yy + 0.13, nombre, va='center', ha='right', fontsize=14)
            ax.text(xlim[0] / 1.6, yy - 0.27, nota(extra), va='center', ha='right', fontsize=12, color=GRAY)
        ax.set_xscale('log'); ax.set_xlim(*xlim); ax.set_ylim(-0.6, n - 0.4); ax.set_yticks([])
        ax.set_title(titulo, loc='left', fontsize=17, pad=14, x=-0.62)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRAY); ax.tick_params(axis='x', labelsize=13, colors=GRAY)
        ax.grid(axis='x', color='#e3e6ea', lw=0.8); ax.set_axisbelow(True)

    barras(a1, filas1, 'Forward pass, one image, encoder $g$', (1e5, 1e11),
           lambda p: f'{legible(p)} parameters')
    barras(a2, filas2, 'Whole pretraining, forward + backward', (1e11, 1e20), lambda s: s)
    sim, img = filas2[5][1], filas2[6][1]
    fig.text(0.205, 0.045, 'At 32×32 with no downsampling at the stem, the same\n11 M weights cost about 1 GFLOP per image.',
             fontsize=14, color=GRAY, va='bottom')
    fig.text(0.70, 0.045, f'SimCLR here ≈ 1/{img / sim:.0f} of supervised ImageNet.\nTwo views per image double every step.',
             fontsize=14, color=GRAY, va='bottom')
    fig.savefig(AQUI / 's04_f13_compute_costs.png', dpi=100); plt.close(fig)
    print('escrita s04_f13_compute_costs.png')


if __name__ == '__main__':
    R = medir()
    f12(R)
    f13(R)
