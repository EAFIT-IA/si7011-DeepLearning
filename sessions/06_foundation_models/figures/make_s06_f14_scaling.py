"""S06-F14 · Loss against compute — reproducible 16:9 matplotlib figure.

Run: python make_s06_f14_scaling.py
Produces s06_f14_scaling.png (1672x941) in this folder.

No training: the curves come from a published fit of the loss of language models as a
function of parameters N and training tokens D,

    L(N, D) = E + A / N^alpha + B / D^beta,     C = 6 N D  (training FLOPs).

Coefficients: the Chinchilla fit as re-estimated by Besiroglu, Erdil, Barnett and You (2024),
"Chinchilla Scaling: A replication attempt". The original "approach 3" fit of Hoffmann et al.
(2022) puts the optimum at 40-100 tokens per parameter, inconsistent with the paper's own
approaches 1-2 (about 20); the re-estimate agrees with them, which is what slide 53 states.
The loss is in nats per token on the Chinchilla training distribution: compare shapes, not values.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parent
E, A, B, ALPHA, BETA = 1.8172, 482.01, 2085.43, 0.3478, 0.3658

NAVY = '#14213d'
BLUE = '#1f6fb4'
TEAL = '#167d78'
ORANGE = '#d9631e'
GRAY = '#8a94a3'
RED = '#c0392b'

mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17, 'mathtext.fontset': 'dejavusans',
                     'axes.edgecolor': '#9aa3ae', 'axes.labelcolor': NAVY, 'xtick.color': NAVY,
                     'ytick.color': NAVY})


def loss(N, D):
    return E + A / N**ALPHA + B / D**BETA


C = np.logspace(17, 25, 400)
fig, ax = plt.subplots(figsize=(16.72, 9.41), dpi=100)
fig.subplots_adjust(left=0.08, right=0.86, bottom=0.12, top=0.95)

# one curve per model size: train longer (more D) at fixed N
sizes = [(1e8, '100 M'), (1e9, '1 B'), (1e10, '10 B'), (1e11, '100 B')]
shades = mpl.colormaps['Blues'](np.linspace(0.45, 0.95, len(sizes)))
for (N, lab), col in zip(sizes, shades):
    D = C / (6 * N)
    ok = D >= N                                       # at least one token per parameter
    ax.plot(C[ok], loss(N, D[ok]), color=col, lw=2.4)
    ax.text(1.15e25, loss(N, D[-1]), f'N = {lab}', ha='left', va='center', fontsize=15, color=col,
            clip_on=False)                            # label at the plateau, in the right margin

# compute-optimal frontier: for each C, the best N
Ns = np.logspace(6, 13, 4000)
best_L, best_ratio = [], []
for c in C:
    Ls = loss(Ns, c / (6 * Ns))
    k = Ls.argmin()
    best_L.append(Ls[k])
    best_ratio.append(c / (6 * Ns[k] ** 2))
ax.plot(C, best_L, color=ORANGE, lw=4, zorder=5, label='compute-optimal: best N for each budget')

# two models with almost the same compute
for name, N, D, col, xy in [('Gopher: 280 B params, 0.3 T tokens', 280e9, 300e9, RED, (1.5e19, 2.12)),
                            ('Chinchilla: 70 B params, 1.4 T tokens', 70e9, 1.4e12, TEAL, (1.5e19, 1.93))]:
    c = 6 * N * D
    ax.plot(c, loss(N, D), 'o', ms=13, color=col, zorder=6, mec='white', mew=2)
    ax.annotate(f'{name}\nloss {loss(N, D):.2f}', (c, loss(N, D)), xytext=xy, fontsize=15, color=col,
                ha='left', va='center', arrowprops=dict(arrowstyle='-', color=col, lw=1.4, shrinkB=8))

ax.axhline(E, color=GRAY, lw=1.5, ls='--')
ax.text(1.4e17, E + 0.01, f'irreducible loss E = {E:.2f}', fontsize=15, color=GRAY, va='bottom')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(1.78, 4.2)
ax.set_xlim(1e17, 1e25)
yt = [2, 2.5, 3, 3.5, 4]
ax.set_yticks(yt, [str(v) for v in yt])
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax.set_xlabel('training compute  C ≈ 6ND  (FLOPs)')
ax.set_ylabel('predicted test loss (nats / token)')
ax.grid(True, which='major', alpha=0.25)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
r0, r1 = best_ratio[np.searchsorted(C, 1e19)], best_ratio[np.searchsorted(C, 1e24)]
ax.legend(loc='upper right', frameon=False, fontsize=16)
ax.text(0.985, 0.86, f'on the orange curve: {r1:.0f}–{r0:.0f} tokens per parameter', transform=ax.transAxes,
        ha='right', va='top', fontsize=15, color=ORANGE)
fig.text(0.86, 0.015, 'fit: Hoffmann et al. (2022), re-estimated by Besiroglu et al. (2024)',
         ha='right', va='bottom', fontsize=12, color=GRAY)
fig.savefig(OUT / 's06_f14_scaling.png', dpi=100)
print('wrote s06_f14_scaling.png', f'tokens/param on the frontier: {r1:.1f} (1e24) to {r0:.1f} (1e19)')
