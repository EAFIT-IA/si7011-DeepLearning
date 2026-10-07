# Generates s01_f06_local_slope.png: the derivative is local information.
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, TEAL, ORANGE, GRAY, BLUE = '#153b63', '#146c72', '#d9631e', '#8a94a3', '#1f6fb4'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
L = lambda w: 0.06 * w**4 - 0.55 * w**2 + 0.35 * w + 2.9
dL = lambda w: 0.24 * w**3 - 1.1 * w + 0.35
w = np.linspace(-3.4, 3.3, 600)

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.text(0.5, 0.93, 'The derivative is local information', ha='center', fontsize=34, weight='bold', color=NAVY)
fig.text(0.5, 0.875, 'At the current value $w_t$, the slope says how $L$ changes nearby, and nothing more.',
         ha='center', fontsize=20, color='#4b596b')
ax = fig.add_axes([0.06, 0.08, 0.6, 0.74])
ax.plot(w, L(w), color='#334155', lw=4)
wt = 0.9; lt = L(wt); s = dL(wt)
seg = np.array([wt - 0.75, wt + 0.75])
ax.plot(seg, lt + s * (seg - wt), color=TEAL, lw=4, solid_capstyle='round')
ax.plot(wt, lt, 'o', color=NAVY, ms=14, zorder=5)
ax.annotate(r'tangent: slope $\dfrac{dL}{dw}(w_t) < 0$', xy=(wt - 0.5, lt + s * (-0.5)), xytext=(-0.9, 4.1),
            fontsize=19, color=TEAL, arrowprops=dict(arrowstyle='-', color=TEAL, lw=1.5))
# gradient step
eta = 1.5; w1 = wt - eta * s
ax.annotate('', xy=(w1, 0.45), xytext=(wt, 0.45),
            arrowprops=dict(arrowstyle='-|>', color=ORANGE, lw=4, mutation_scale=30))
ax.text(w1 + 0.15, 0.45, r'step $-\eta\, dL/dw$', ha='left', va='center', fontsize=19, color=ORANGE)
ax.plot([wt, wt], [0.3, lt], ls=(0, (5, 5)), color=GRAY, lw=1.5)
ax.plot([w1, w1], [0.3, L(w1)], ls=(0, (5, 5)), color=GRAY, lw=1.5)
ax.plot(w1, L(w1), 'o', color=ORANGE, ms=11, zorder=5)
ax.text(wt, 0.12, r'$w_t$', ha='center', fontsize=22, color=NAVY)
ax.text(w1, 0.12, r'$w_{t+1}$', ha='center', fontsize=22, color=ORANGE)
# minima
wg = -2.285; wl = 1.959
ax.plot(wg, L(wg), marker='*', color=BLUE, ms=22, zorder=5)
ax.text(wg, L(wg) - 0.42, 'global minimum', ha='center', fontsize=17, color=BLUE)
ax.plot(wl, L(wl), 'o', mfc='white', mec='#4b596b', mew=2, ms=11, zorder=5)
ax.text(wl + 0.35, L(wl) - 0.4, 'nearest minimum', ha='center', fontsize=17, color='#4b596b')
ax.set_xlim(-3.5, 3.4); ax.set_ylim(0, 4.9)
ax.set_xlabel('$w$', fontsize=24, loc='right'); ax.set_ylabel('$L(w)$', fontsize=24, rotation=0, loc='top', labelpad=-10)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_linewidth(2); ax.spines[sp].set_color('#334155')
ax.set_xticks([]); ax.set_yticks([])

# side box
fig.patches.append(FancyBboxPatch((0.69, 0.27), 0.29, 0.46, boxstyle='round,pad=0,rounding_size=0.015',
                                  transform=fig.transFigure, fc='#eef6f7', ec=TEAL, lw=2))
lines = [('What the gradient tells us', 20, 'bold', NAVY),
         ('• the direction in which $L$', 17, 'normal', '#172033'), ('  decreases at $w_t$', 17, 'normal', '#172033'),
         ('• how steep it is there', 17, 'normal', '#172033'),
         ('What it does not tell us', 20, 'bold', ORANGE),
         ('• where the minimum is', 17, 'normal', '#172033'),
         ('• whether a better minimum', 17, 'normal', '#172033'), ('  exists elsewhere', 17, 'normal', '#172033')]
y = 0.70
for txt, fs, wgt, col in lines:
    if wgt == 'bold' and y < 0.68: y -= 0.02
    fig.text(0.705, y, txt, fontsize=fs, weight=wgt, color=col, va='top')
    y -= 0.048
fig.savefig(sys.argv[1], dpi=100, facecolor='white')
