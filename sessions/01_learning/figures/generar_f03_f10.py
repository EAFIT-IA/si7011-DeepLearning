"""
Genera s01_f03_data_model.svg y s01_f10_batch_strategies.svg (SI7011 · S01).

F03: un modelo es una función parametrizada. La arquitectura fija la familia
     ŷ = w x + b; cada θ = (w, b) elige una función concreta.
F10: batch, SGD y mini-batch sobre la misma regresión lineal (numpy, semilla 7).
     Las trayectorias son descenso de gradiente real con el mismo η.

Uso:  python generar_f03_f10.py
"""
import numpy as np

W, H = 1600, 900
AZUL, ROJO, TEAL, NARANJA = "#1f6fb4", "#d1392b", "#146c72", "#d38b33"
TINTA, GRIS, GRIS_CLARO, TITULO = "#172033", "#5b6775", "#c4cbd5", "#153b63"
ESTILO = f"""<style>
.t{{font-family:Arial,Helvetica,sans-serif;fill:{TINTA}}}
.h{{font-family:Arial,Helvetica,sans-serif;font-weight:700;font-size:34px;fill:{TINTA}}}
.s{{font-family:Arial,Helvetica,sans-serif;font-size:22px;fill:{GRIS}}}
.m{{font-family:Georgia,'Times New Roman',serif;font-style:italic;fill:{TINTA}}}
</style>"""


def sub(t, fs=22):
    return f'<tspan baseline-shift="sub" font-size="{fs}">{t}</tspan>'


def caja(x, y, w, h, fill, stroke, r=18):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>'


def flecha(x1, y1, x2, y2, color="#334155", w=4):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" '
            f'marker-end="url(#pt)"/>')


DEFS = '''<defs><marker id="pt" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
<path d="M0,0 L12,6 L0,12 z" fill="#334155"/></marker></defs>'''


def svg(cuerpo):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            f'<rect width="100%" height="100%" fill="white"/>{ESTILO}{DEFS}{cuerpo}</svg>')


# ======================================================================
# F03 · Model as a parameterized function
# ======================================================================
def f03():
    g = []
    # --- fila superior: diagrama x → f_θ → ŷ con θ entrando desde arriba
    g.append(caja(170, 150, 220, 130, "#e4eefc", "#7aa6dc"))
    g.append(f'<text x="280" y="205" text-anchor="middle" class="t" font-size="26" font-weight="700">input</text>')
    g.append(f'<text x="280" y="250" text-anchor="middle" class="m" font-size="38">x</text>')
    g.append(caja(620, 150, 360, 130, "#e3f5ea", "#83c9a0"))
    g.append(f'<text x="800" y="205" text-anchor="middle" class="t" font-size="26" font-weight="700">model</text>')
    g.append(f'<text x="800" y="252" text-anchor="middle" class="m" font-size="40">f{sub("θ", 26)}(x)</text>')
    g.append(caja(1210, 150, 220, 130, "#fdeee3", NARANJA))
    g.append(f'<text x="1320" y="205" text-anchor="middle" class="t" font-size="26" font-weight="700">prediction</text>')
    g.append(f'<text x="1320" y="250" text-anchor="middle" class="m" font-size="38">ŷ</text>')
    g.append(flecha(400, 215, 605, 215))
    g.append(flecha(990, 215, 1195, 215))
    # θ: perillas
    g.append(f'<text x="800" y="70" text-anchor="middle" class="m" font-size="34">θ = (w, b)</text>')
    g.append(flecha(800, 85, 800, 140, color="#146c72"))
    g.append(f'<text x="1010" y="110" class="s">parameters: learned from data</text>')
    g.append(f'<text x="800" y="320" text-anchor="middle" class="s">the architecture fixes the family  ŷ = w x + b</text>')

    # --- fila inferior: misma familia, tres θ distintos sobre los mismos datos
    r = np.random.default_rng(11)
    xs = np.array([-2.6, -1.9, -1.1, -0.4, 0.3, 1.0, 1.8, 2.5])
    ys = np.round(1.2 * xs + 2.0 + r.normal(0, 0.55, xs.size), 2)
    thetas = [(-0.6, 3.0, ROJO), (0.6, 1.0, NARANJA), (1.15, 2.1, TEAL)]
    for k, (w, b, col) in enumerate(thetas):
        x0, y0, pw, ph = 140 + k * 470, 400, 400, 330
        g.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="16" fill="#f8fafc" stroke="#d6dde6" stroke-width="2"/>')
        px = lambda v: x0 + 30 + (v + 3) / 6 * (pw - 60)
        py = lambda v: y0 + ph - 50 - (v + 2) / 9 * (ph - 110)
        g.append(f'<line x1="{px(-3)}" y1="{py(0)}" x2="{px(3)}" y2="{py(0)}" stroke="{GRIS_CLARO}" stroke-width="2"/>')
        g.append(f'<line x1="{px(0)}" y1="{py(-2)}" x2="{px(0)}" y2="{py(7)}" stroke="{GRIS_CLARO}" stroke-width="2"/>')
        for a, c in zip(xs, ys):
            g.append(f'<circle cx="{px(a):.1f}" cy="{py(c):.1f}" r="7" fill="{AZUL}"/>')
        xa, xb = -3, 3
        ya, yb = np.clip(w * xa + b, -2, 7), np.clip(w * xb + b, -2, 7)
        xa2 = (ya - b) / w if w else xa
        xb2 = (yb - b) / w if w else xb
        g.append(f'<line x1="{px(max(xa, min(xa2, xb2))):.1f}" y1="{py(w * max(xa, min(xa2, xb2)) + b):.1f}" '
                 f'x2="{px(min(xb, max(xa2, xb2))):.1f}" y2="{py(w * min(xb, max(xa2, xb2)) + b):.1f}" '
                 f'stroke="{col}" stroke-width="5"/>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 40}" text-anchor="middle" class="m" font-size="26" fill="{col}">'
                 f'<tspan fill="{col}">θ{sub(k + 1, 18)} = ({w:g}, {b:g})</tspan></text>')
        verdict = ["wrong trend", "right trend, wrong level", "close fit"][k]
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph - 15}" text-anchor="middle" class="s">{verdict}</text>')
    g.append(f'<text x="800" y="800" text-anchor="middle" class="t" font-size="28" font-weight="700">'
             f'Same model, different θ: different functions. Training searches for a good θ.</text>')
    return svg(''.join(g))


# ======================================================================
# F10 · Batch, SGD and mini-batch (trayectorias reales)
# ======================================================================
def f10():
    r = np.random.default_rng(7)
    N = 64
    x = r.uniform(-2, 2, N)
    y = 1.5 * x - 0.5 + r.normal(0, 0.8, N)
    A = np.c_[x, np.ones(N)]
    w_opt = np.linalg.lstsq(A, y, rcond=None)[0]

    def J(w, b):
        return np.mean((y - (w * x + b)) ** 2)

    def grad(th, idx):
        e = A[idx] @ th - y[idx]
        return 2 * A[idx].T @ e / len(idx)

    def correr(bsz, eta=0.08, pasos=40, semilla=3):
        rr = np.random.default_rng(semilla)
        th = np.array([-1.5, 2.0])
        P = [th.copy()]
        for _ in range(pasos):
            idx = np.arange(N) if bsz == N else rr.choice(N, bsz, replace=False)
            th = th - eta * grad(th, idx)
            P.append(th.copy())
        return np.array(P)

    casos = [("Batch", N, r"|B| = N = 64", "exact gradient, smooth path", "64 examples per update", AZUL),
             ("SGD", 1, r"|B| = 1", "very noisy estimate", "1 example per update", ROJO),
             ("Mini-batch", 8, r"1 &lt; |B| &lt; N  (here 8)", "noisy but cheap", "8 examples per update", TEAL)]
    g = []
    wr, br = (-2.0, 3.0), (-2.5, 2.5)
    for k, (nombre, bsz, form, desc, coste, col) in enumerate(casos):
        x0, y0, pw, ph = 50 + k * 510, 70, 480, 690
        g.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" rx="20" fill="#f8fafc" stroke="#d6dde6" stroke-width="2"/>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 50}" text-anchor="middle" class="h" fill="{col}"><tspan fill="{col}">{nombre}</tspan></text>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 92}" text-anchor="middle" class="m" font-size="26">{form}</text>')
        # puntos del dataset: muestra cuáles entran en el lote
        rr = np.random.default_rng(3)
        sel = set(range(N)) if bsz == N else set(rr.choice(N, bsz, replace=False).tolist())
        for i in range(N):
            cx = x0 + 40 + (i % 16) * 26.5
            cy = y0 + 130 + (i // 16) * 24
            on = i in sel
            g.append(f'<circle cx="{cx:.1f}" cy="{cy}" r="8" fill="{col if on else "#e2e8f0"}"/>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 245}" text-anchor="middle" class="s">one update uses the highlighted examples</text>')
        # contornos de J(w, b) y trayectoria
        cx0, cy0, cw, ch = x0 + 40, y0 + 270, pw - 80, 300
        px = lambda w: cx0 + (w - wr[0]) / (wr[1] - wr[0]) * cw
        py = lambda b: cy0 + ch - (b - br[0]) / (br[1] - br[0]) * ch
        g.append(f'<rect x="{cx0}" y="{cy0}" width="{cw}" height="{ch}" fill="white" stroke="{GRIS_CLARO}" stroke-width="1.5"/>')
        H2 = A.T @ A / N
        vals, vecs = np.linalg.eigh(H2)
        J0 = J(*w_opt)
        for nivel in (0.3, 1, 2.5, 5, 9):
            t = np.linspace(0, 2 * np.pi, 120)
            u = np.vstack([np.cos(t), np.sin(t)]) * np.sqrt(nivel)
            pts = (vecs @ np.diag(1 / np.sqrt(vals)) @ vecs.T @ u).T + w_opt
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(p[0]):.1f},{py(p[1]):.1f}" for i, p in enumerate(pts)
                         if wr[0] <= p[0] <= wr[1] and br[0] <= p[1] <= br[1])
            if d:
                g.append(f'<path d="{d}" fill="none" stroke="#bcd3ee" stroke-width="1.6"/>')
        P = correr(bsz)
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(np.clip(p[0], *wr)):.1f},{py(np.clip(p[1], *br)):.1f}" for i, p in enumerate(P))
        g.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="3"/>')
        g.append(f'<circle cx="{px(P[0][0]):.1f}" cy="{py(P[0][1]):.1f}" r="8" fill="white" stroke="{col}" stroke-width="3"/>')
        g.append(f'<text x="{px(w_opt[0]):.1f}" y="{py(w_opt[1]) + 9:.1f}" text-anchor="middle" class="t" font-size="26">×</text>')
        g.append(f'<text x="{cx0 + cw - 8}" y="{cy0 + ch - 10}" text-anchor="end" class="m" font-size="22">w</text>')
        g.append(f'<text x="{cx0 + 10}" y="{cy0 + 26}" class="m" font-size="22">b</text>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 615}" text-anchor="middle" class="t" font-size="24">{desc}</text>')
        g.append(f'<text x="{x0 + pw / 2}" y="{y0 + 650}" text-anchor="middle" class="s">{coste}</text>')
    g.append(f'<text x="800" y="815" text-anchor="middle" class="m" font-size="30">'
             f'g{sub("t")} = ∇{sub("θ")}J{sub("B")}(θ{sub("t")}),   θ{sub("t+1")} = θ{sub("t")} − η g{sub("t")}</text>')
    g.append(f'<text x="800" y="860" text-anchor="middle" class="s">'
             f'Same data, same start, same η = 0.08, 40 updates. × marks the least-squares minimum of J(w, b).</text>')
    return svg(''.join(g))


if __name__ == "__main__":
    open("s01_f03_data_model.svg", "w", encoding="utf-8").write(f03())
    open("s01_f10_batch_strategies.svg", "w", encoding="utf-8").write(f10())
    print("ok")
