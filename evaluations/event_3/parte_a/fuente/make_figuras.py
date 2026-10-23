"""
Figuras de la Parte A del evento evaluativo 3.

Todos los números vienen de los resultados guardados por los notebooks de las sesiones 05 y 06 (corridas en Kaggle):
    fig1_gradiente.png  notebook 5.3: |dL/dx_t| por hora de la ventana, RNN, LSTM y GRU entrenadas en Jena
    fig2_mascaras.png   notebook 6.1: la misma autoatención con tres máscaras
    fig3_perdida.png    notebook 6.3: la pérdida de cada token de un par de Dolly, Qwen2.5-0.5B base
Los nombres se ocultan (A/B/C, P/Q/R): identificarlos es parte de la pregunta.

Uso: python make_figuras.py   → escribe los PNG en esta carpeta.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[3]
S05 = RAIZ / 'sessions/05_sequences/notebooks'
S06 = RAIZ / 'sessions/06_foundation_models/notebooks'
plt.rcParams.update({'font.size': 11})

# ---------------------------------------------------------------- 1 · gradiente en el tiempo (notebook 5.3)
g = np.load(S05 / 'gradiente_tiempo_lstm_gru.npz')
curvas = {'A': g['g_lstm'], 'B': g['g_rnn'], 'C': g['g_gru']}      # A = LSTM, B = RNN, C = GRU
estilos = {'A': ('tab:blue', '-'), 'B': ('tab:red', '-'), 'C': ('tab:green', '--')}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
T = len(g['g_rnn'])
horas = T - 1 - np.arange(T)                 # 0 = la última hora de la ventana
for k, y in curvas.items():
    c, ls = estilos[k]
    y = np.where(y > 0, y, np.nan)
    ax.plot(horas, y, color=c, ls=ls, lw=2, label=f'modelo {k}')
ax.set_yscale('log'); ax.set_ylim(1e-24, 1e-2)
ax.invert_xaxis()
ax.set_xlabel('horas antes del final de la ventana'); ax.set_ylabel(r'$\|\partial L/\partial \mathbf{x}_t\|$')
ax.set_title('Jena, ventanas de 120 h, modelos entrenados, promedio sobre un lote', fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='lower right')
fig.tight_layout(); fig.savefig(AQUI / 'fig1_gradiente.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2 · tres máscaras (notebook 6.1)
a = np.load(S06 / 'atencion_mascaras.npz')
toks = [str(t) for t in a['tokens']]
mapas = {'P': a['A_causal'], 'Q': a['A_sin'], 'R': a['A_pad']}      # P = causal + relleno, Q = ninguna, R = relleno
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6))
for ax, (k, A) in zip(axs, mapas.items()):
    ax.imshow(A, cmap='Blues', vmin=0, vmax=1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f'{A[i, j]:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if A[i, j] > 0.55 else 'black')
    ax.set_xticks(range(len(toks))); ax.set_xticklabels(toks, rotation=60, fontsize=8)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks, fontsize=8)
    ax.set_title(f'máscara {k}', fontsize=11)
axs[0].set_ylabel('consulta (fila $i$)'); axs[1].set_xlabel('clave (columna $j$)')
fig.tight_layout(); fig.savefig(AQUI / 'fig2_mascaras.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3 · pérdida por token (notebook 6.3)
pt = json.load(open(S06 / 'resultados_sft.json'))['por_token']
piezas, papel, L = pt['piezas'], pt['papel'], np.array(pt['L'])
pred = piezas[1:]                     # L[t] es la pérdida de predecir pred[t]
rol = papel[1:]
color = {0: '#9aa3ad', 1: '#5b9bd5', 2: '#e06c1f'}
nombre = {0: 'plantilla', 1: 'pregunta', 2: 'respuesta y fin'}
fig, ax = plt.subplots(figsize=(10.5, 3.8))
x = np.arange(len(L))
ax.bar(x, L, color=[color[r] for r in rol], width=0.8)
etq = [p.replace('\n', '⏎').replace(' ', '·').replace('<|endoftext|>', '<fin>') for p in pred]
ax.set_xticks(x); ax.set_xticklabels(etq, rotation=70, fontsize=8)
for t, r in zip(ax.get_xticklabels(), rol):
    t.set_color(color[r] if r else '#5f6873')
for r in (0, 1, 2):
    ax.bar([0], [0], color=color[r], label=nombre[r])
ax.legend(fontsize=9, ncol=3, loc='upper left')
ax.set_ylabel(r'$L_t=-\log p(w_{t+1})$ (nats)')
ax.set_title('Qwen2.5-0.5B base, antes de ajustar: «Usuario: ¿Qué hace la Constitución? ⏎ Asistente: '
             'Protege los derechos básicos de los estadounidenses»', fontsize=9)
ax.grid(axis='y', alpha=0.3); ax.set_xlim(-0.6, len(L) - 0.4)
fig.tight_layout(); fig.savefig(AQUI / 'fig3_perdida.png', dpi=200); plt.close(fig)

print('promedio de las 26 posiciones', round(L.mean(), 2), '· solo respuesta y fin',
      round(L[np.array(rol) == 2].mean(), 2), f'({int((np.array(rol) == 2).sum())} posiciones)')
