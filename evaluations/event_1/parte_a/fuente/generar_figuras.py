"""Real runs for the Part A exam figures (Fashion-MNIST, CPU, about 1 minute).
Run from this folder: python generar_figuras.py .  (writes fig1–fig4 and facts.json)"""
import sys, os, time, math, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)
plt.rcParams.update({'figure.dpi': 200, 'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
ROOT = 'data'   # Fashion-MNIST; torchvision downloads it here if missing
tr = datasets.FashionMNIST(ROOT, train=True, download=True)
X = tr.data.reshape(-1, 784).float() / 255; y = tr.targets
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(1))
mu, sd = X[perm[:20000]].mean(), X[perm[:20000]].std()
X = (X - mu) / sd
Xtr, ytr = X[perm[:20000]], y[perm[:20000]]
Xva, yva = X[perm[50000:]], y[perm[50000:]]
loss_fn = nn.CrossEntropyLoss()
facts = {}

def mlp(sizes, init=None):
    L = []
    for a, b in zip(sizes[:-2], sizes[1:-1]):
        L += [nn.Linear(a, b), nn.ReLU()]
    m = nn.Sequential(*L, nn.Linear(sizes[-2], sizes[-1]))
    if init:
        for l in m:
            if isinstance(l, nn.Linear): init(l.weight); nn.init.zeros_(l.bias)
    return m

he = lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')

def fit(m, opt, n, epochs, bs=128, step_log=False):
    Xn, yn = Xtr[:n], ytr[:n]; h = {'train': [], 'val': [], 'steps': [], 'sec': []}
    for e in range(epochs):
        t = time.time(); m.train(); p = torch.randperm(n); tot = 0
        for i in range(0, n, bs):
            b = p[i:i + bs]; loss = loss_fn(m(Xn[b]), yn[b]); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
            if step_log: h['steps'].append(loss.item())
        h['sec'].append(time.time() - t)
        m.eval()
        with torch.no_grad(): h['val'].append(loss_fn(m(Xva), yva).item())
        h['train'].append(tot / n)
    return h

def curves(ax, h, title):
    ep = range(1, len(h['train']) + 1)
    ax.plot(ep, h['train'], label='entrenamiento'); ax.plot(ep, h['val'], label='validación')
    ax.set_title(title); ax.set_xlabel('época'); ax.set_ylabel('pérdida'); ax.legend()

# ---- Figure 1: two runs to diagnose
torch.manual_seed(0); m = mlp([784, 1024, 1024, 10], he)
hA = fit(m, torch.optim.Adam(m.parameters(), 1e-3), 5000, 30)
torch.manual_seed(0); m = mlp([784, 256, 256, 10], he)
hB = fit(m, torch.optim.SGD(m.parameters(), 2e-4), 20000, 30)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
curves(ax[0], hA, 'Corrida A'); curves(ax[1], hB, 'Corrida B'); fig.tight_layout(); fig.savefig(f'{OUT}/fig1_curvas.png'); plt.close(fig)
facts['A'] = dict(train_final=hA['train'][-1], val_min=min(hA['val']), val_min_epoch=1 + hA['val'].index(min(hA['val'])), val_final=hA['val'][-1])
facts['B'] = dict(train_final=hB['train'][-1], val_final=hB['val'][-1])

# ---- Figure 2: three unlabeled initializations (ReLU, 15 layers, width 256)
x = Xtr[:512]
schemes = {'I': he, 'II': None, 'III': nn.init.xavier_normal_}      # I = He, II = PyTorch default, III = Xavier
fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
for name, init in schemes.items():
    torch.manual_seed(0); m = mlp([784] + [256] * 15 + [10], init)
    stds = []; hooks = [l.register_forward_hook(lambda mod, i, o: stds.append(o.std().item())) for l in m if isinstance(l, nn.ReLU)]
    m.zero_grad(); loss = loss_fn(m(x), ytr[:512]); loss.backward()
    for hk in hooks: hk.remove()
    g = [l.weight.grad.norm().item() for l in m if isinstance(l, nn.Linear)][:-1]
    ax[0].plot(range(1, 16), stds, marker='o', ms=3, label=f'inicialización {name}')
    ax[1].plot(range(1, 16), g, marker='o', ms=3, label=f'inicialización {name}')
    facts[f'init_{name}'] = dict(std_first=stds[0], std_last=stds[-1], grad_first=g[0], grad_last=g[-1], loss0=loss.item())
for a, t, yl in [(ax[0], 'Hacia adelante', r'std($a_l$)'), (ax[1], 'Hacia atrás', r'$\|\nabla_{W_l} J\|$')]:
    a.set_yscale('log'); a.set_xlabel('capa $l$'); a.set_ylabel(yl); a.set_title(t)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f'{OUT}/fig2_capas.png'); plt.close(fig)

# ---- Figure 3: batch size and gradient noise
runs = {}
for name, bs in [('P', 1024), ('Q', 16)]:
    torch.manual_seed(0); m = mlp([784, 256, 256, 10], he)
    runs[name] = fit(m, torch.optim.SGD(m.parameters(), 0.05), 20000, 3, bs=bs, step_log=True)
    facts[f'bs_{name}'] = dict(batch=bs, steps_per_epoch=math.ceil(20000 / bs), sec_per_epoch=sum(runs[name]['sec']) / 3, train_final=runs[name]['train'][-1])
fig, ax = plt.subplots(figsize=(7, 2.6))
for name in ['Q', 'P']:
    s = runs[name]['steps']; ax.plot([3 * (i + 1) / len(s) for i in range(len(s))], s, label=f'corrida {name}', alpha=0.85, lw=0.8 if name == 'Q' else 1.6)
ax.set_xlabel('época'); ax.set_ylabel('pérdida del mini-lote'); ax.set_title('Pérdida en cada paso, mismo η = 0.05, 3 épocas'); ax.legend()
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_lote.png'); plt.close(fig)

# ---- Figure 4: depth degradation
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for d, a in [(6, ax[0]), (40, ax[1])]:
    torch.manual_seed(0); m = mlp([784] + [128] * d + [10], he)
    h = fit(m, torch.optim.Adam(m.parameters(), 1e-3), 20000, 10)
    curves(a, h, f'{d} capas ocultas'); facts[f'depth_{d}'] = dict(train_final=h['train'][-1], val_final=h['val'][-1])
ymax = max(a.get_ylim()[1] for a in ax)
for a in ax: a.set_ylim(0, ymax)
fig.tight_layout(); fig.savefig(f'{OUT}/fig4_profundidad.png'); plt.close(fig)

json.dump(facts, open(f'{OUT}/facts.json', 'w'), indent=1)
print(json.dumps(facts, indent=1))
