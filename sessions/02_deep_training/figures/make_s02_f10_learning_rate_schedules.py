"""S02-F10: learning-rate schedules named on the slide (constant, step, exponential, warmup + cosine).
Run from the repository root: python sessions/02_deep_training/figures/make_s02_f10_learning_rate_schedules.py"""
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T, Tw, eta0 = 100, 10, 1.0
t = list(range(T))
schedules = {
    'constant': [eta0] * T,
    'step decay (×0.3 every 30)': [eta0 * 0.3 ** (k // 30) for k in t],
    'exponential decay': [eta0 * math.exp(-0.035 * k) for k in t],
    'warmup + cosine': [eta0 * (k + 1) / Tw if k < Tw else eta0 * 0.5 * (1 + math.cos(math.pi * (k - Tw) / (T - Tw))) for k in t],
}
colors = ['#9aa0a6', '#1f6fb2', '#d58043', '#167d78']

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(11, 3.3))
for (name, eta), c in zip(schedules.items(), colors):
    ax.plot(t, eta, color=c, lw=3.2 if name == 'warmup + cosine' else 2.2,
            ls=(0, (5, 4)) if name == 'constant' else '-', label=name)
ax.axvspan(0, Tw, color='#167d78', alpha=0.08)
ax.text(Tw / 2, 1.06, 'warmup\n$t < T_w$', ha='center', va='bottom', color='#167d78', fontsize=11)
ax.set_xlabel('training progress (% of steps)')
ax.set_ylabel(r'$\eta_t / \eta_0$')
ax.set_ylim(0, 1.25); ax.set_xlim(0, T - 1)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(alpha=0.25)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, handlelength=2.5)
fig.tight_layout()
fig.savefig('sessions/02_deep_training/figures/s02_f10_learning_rate_schedules.svg')
