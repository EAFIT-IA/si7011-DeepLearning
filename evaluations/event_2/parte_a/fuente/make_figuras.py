"""
Figuras de la Parte A del evento evaluativo 2.

Todos los números vienen de las salidas guardadas de los notebooks de la sesión 03 (corrida en Kaggle, GPU):
    fig1_permutacion.png  notebook 3.1, secciones 2–4
    fig2_bn_aug.png       notebook 3.3, secciones 1–3 (época 12)
    fig3_transfer.png     notebook 3.4, sección 4 (resumen de las tres estrategias)
Los nombres de los modelos se ocultan (X/Y, A/B/C, I/II/III): identificarlos es parte de la pregunta.

Uso: python make_figuras.py   → escribe los PNG en esta carpeta.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 11})

# ---------------------------------------------------------------- 1 · permutar los píxeles (notebook 3.1)
# MLP: 1 707 274 parámetros · val_acc 0.441 en orden, 0.438 permutado
# CNN: 66 570 parámetros · val_acc 0.604 en orden, 0.405 permutado
fig, ax = plt.subplots(figsize=(7.2, 3.0))
modelos = ['Modelo X\n1 707 274 parámetros', 'Modelo Y\n66 570 parámetros']
orden, perm = [0.441, 0.604], [0.438, 0.405]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, orden, w, label='píxeles en su orden', color='tab:blue')
b2 = ax.bar(x + w / 2, perm, w, label='píxeles permutados', color='tab:orange')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.axhline(0.1, color='gray', ls=':', lw=1); ax.text(1.55, 0.112, 'azar', color='gray', fontsize=9, ha='right')
ax.set_xticks(x); ax.set_xticklabels(modelos); ax.set_ylim(0, 0.72); ax.set_ylabel('accuracy de validación')
ax.set_title('CIFAR-10, 10 épocas, el mismo loop', fontsize=11)
ax.legend(loc='upper left', fontsize=9); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(AQUI / 'fig1_permutacion.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2 · BatchNorm y augmentation (notebook 3.3)
# VGG: acc 0.502 · val 0.501 | VGG + BN: 0.800 · 0.611 | VGG + BN + augmentation: 0.688 · 0.656  (época 12)
corridas = {'Corrida A': (0.688, 0.656), 'Corrida B': (0.502, 0.501), 'Corrida C': (0.800, 0.611)}
fig, ax = plt.subplots(figsize=(7.2, 3.0))
x = np.arange(3)
tr = [v[0] for v in corridas.values()]; va = [v[1] for v in corridas.values()]
b1 = ax.bar(x - w / 2, tr, w, label='entrenamiento', color='tab:blue')
b2 = ax.bar(x + w / 2, va, w, label='validación', color='tab:orange')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f'{b.get_height():.3f}', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(list(corridas)); ax.set_ylim(0, 0.95); ax.set_ylabel('accuracy, época 12')
ax.set_title('CIFAR-10, la misma red de bloques VGG, 12 épocas', fontsize=11)
ax.legend(loc='upper left', fontsize=9, ncol=2); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(AQUI / 'fig2_bn_aug.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3 · tres estrategias (notebook 3.4)
# k = 5:     probe 0.762 · fine-tuning 0.588 · desde cero 0.086
# k = 20:    probe 0.834 · fine-tuning 0.789 · desde cero 0.174
# k = todas: probe 0.849 · fine-tuning 0.872 · desde cero 0.305   (≈ 80 imágenes por raza)
est = {'Estrategia I': [0.588, 0.789, 0.872], 'Estrategia II': [0.086, 0.174, 0.305],
       'Estrategia III': [0.762, 0.834, 0.849]}
k = [5, 20, 80]
OFF = {'Estrategia I': [-0.07, -0.07, 0.03], 'Estrategia II': [0.03, 0.03, 0.03],
       'Estrategia III': [0.03, 0.03, -0.07]}   # etiquetas sin solaparse
fig, ax = plt.subplots(figsize=(8.2, 3.3))
for (nombre, v), m in zip(est.items(), ['o', 's', '^']):
    ax.plot(k, v, marker=m, lw=2, label=nombre)
    for xi, yi, dy in zip(k, v, OFF[nombre]):
        ax.text(xi, yi + dy, f'{yi:.3f}', ha='center', fontsize=9)
ax.axhline(1 / 37, color='gray', ls=':', lw=1); ax.text(9, 0.04, 'azar (37 clases)', color='gray', fontsize=9)
ax.set_xscale('log'); ax.set_xticks(k); ax.set_xticklabels(['5', '20', '≈ 80 (todas)'])
ax.set_xlim(4, 100); ax.set_ylim(0, 1.0)
ax.set_xlabel('imágenes de entrenamiento por raza'); ax.set_ylabel('accuracy de validación')
ax.set_title('Oxford-IIIT Pets, 37 razas, ResNet-18', fontsize=11)
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(AQUI / 'fig3_transfer.png', dpi=200); plt.close(fig)
print('listo')
