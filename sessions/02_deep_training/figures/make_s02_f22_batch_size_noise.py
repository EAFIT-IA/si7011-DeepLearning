# Generates s02_f22_batch_size_noise.png (simulated mini-batch gradients; usage: python make_s02_f22_batch_size_noise.py out.png)
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch

NAVY, BLUE, ORANGE, GRAY = '#14213d', '#1f6fb4', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)

mu = np.array([2.4, 1.7])                       # full-batch gradient
# per-example gradient covariance (elongated, rotated)
th = np.deg2rad(30); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
S = R @ np.diag([1.6 ** 2, 0.9 ** 2]) @ R.T      # Cov_i(grad L_i)
rng = np.random.default_rng(3)
fills = ['#e8f1fa', '#fdf0e6', '#e9f5ee']
batches = [4, 16, 64]
pw, ph, gap, left, bottom = 0.295, 0.60, 0.0225, 0.0375, 0.27
for j, B in enumerate(batches):
    x0 = left + j * (pw + gap)
    fig.patches.append(FancyBboxPatch((x0, bottom), pw, ph, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc=fills[j], ec='#c9d3df', lw=1.2, zorder=-5))
    ax = fig.add_axes([x0 + 0.01, bottom + 0.01, pw - 0.02, ph - 0.11])
    ax.set_xlim(-0.35, 4.35); ax.set_ylim(-0.45, 3.55); ax.set_aspect('equal'); ax.axis('off')
    C = S / B
    G = rng.multivariate_normal(mu, C, size=40)
    G = G - G.mean(0) + mu                      # show the cloud centred on the full gradient
    for g in G:
        ax.annotate('', xy=g, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=BLUE, alpha=0.35,
                                                             lw=0.9, mutation_scale=8, shrinkA=0, shrinkB=0))
    ax.scatter(G[:, 0], G[:, 1], s=6, color=BLUE, alpha=0.6, zorder=3)
    vals, vecs = np.linalg.eigh(C)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * 2 * np.sqrt(vals[1]), 2 * 2 * np.sqrt(vals[0]), angle=ang, fill=False,
                         ec=ORANGE, lw=2.2, ls=(0, (5, 3)), zorder=4))
    ax.annotate('', xy=mu, xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=3.2,
                                                          mutation_scale=22, shrinkA=0, shrinkB=0), zorder=5)
    ax.plot(0, 0, 'o', color=NAVY, ms=9, zorder=6)
    lab = (3.05, 0.05)
    ax.annotate(r'$\nabla_\theta J$', xy=0.62 * mu, xytext=lab, fontsize=22, color=NAVY, zorder=7,
                ha='left', va='center', arrowprops=dict(arrowstyle='-', color=NAVY, lw=0.9, shrinkA=4, shrinkB=2))
    if j == 0:
        ax.text(lab[0], lab[1] - 0.38, 'full-batch gradient', fontsize=12.5, color=NAVY, ha='left', va='center')
    ax.text(-0.2, 3.1, r'$g_t$', fontsize=22, color=BLUE)
    ax.text(0.25, 3.12, 'mini-batch gradients' if j == 0 else '', fontsize=12.5, color=BLUE, va='center')
    fig.text(x0 + pw / 2, bottom + ph - 0.055, rf'$|\mathcal{{B}}| = {B}$', ha='center', va='center',
             fontsize=28, color=NAVY)
    fig.text(x0 + pw / 2, bottom - 0.055, rf'$\mathrm{{std}} \propto 1/\sqrt{{{B}}} = {["0.50","0.25","0.125"][j]}$',
             ha='center', va='center', fontsize=22, color=NAVY)
fig.text(0.9625, 0.155, 'dashed ellipse: 2 std of $g_t$ around $\\nabla_\\theta J$ · simulated', fontsize=11, color=GRAY, ha='right')
fig.patches.append(FancyBboxPatch((0.0375, 0.045), 0.925, 0.085, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
fig.text(0.5, 0.0875, 'Same expected direction, less noise: 4× the batch halves the standard deviation, '
         'at 4× the cost per step.', ha='center', va='center', fontsize=17, color=NAVY, weight='bold')
fig.savefig(__import__('sys').argv[1], dpi=DPI, facecolor='white')
