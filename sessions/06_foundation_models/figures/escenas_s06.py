"""
Escenas Manim · SI7011 · Sesión 6: Transformers and Foundation Models

    A01_SelfAttention   "banco" compara su query con todas las keys, softmax, promedio de los
                        values; después todas las filas a la vez como una matriz T × T, y la
                        máscara causal apaga el triángulo superior.
    A02_Block           Cinco tokens atraviesan un bloque pre-norm: LN → atención (los tokens
                        intercambian información) → +; LN → MLP (cada token por su cuenta) → +.
    A03_Generation      Generación autorregresiva: distribución sobre el vocabulario, muestreo,
                        el token se agrega y la ventana avanza; luego temperatura baja y alta.
    A04_MaskedLoss      La pérdida del fine-tuning: cada posición predice el token siguiente,
                        p(objetivo) → L_t, la máscara deja solo la respuesta, J = 2.17.

Los números de A01 son los de las figuras S06-F03 (puntajes ilustrativos, pesos = su softmax).
A03 usa las probabilidades de S06-F08; A04 las de la diapositiva 69 (ilustrativas).
Todo lo demás se calcula con numpy al construir la escena (semillas fijas).
Notación del curso: X filas = tokens, Q K V, d_k, A pesos de atención, M máscara, L_t, m_t, J.

Render:
    S01_VIDEO=1 manim -qh escenas_s06.py A01_SelfAttention   # video lineal
    manim -qh escenas_s06.py A01_SelfAttention               # con pausas (manim-slides)
Poster (último cuadro): añadir -s.
"""
import os
from pathlib import Path

import numpy as np
from manim import *

AQUI = Path(__file__).resolve().parent

# ---------------------------------------------------------------- estilo (como S02–S05)
FONDO = WHITE
NAVY = "#14213d"
GRIS_TXT = "#5b6b80"
GRIS_CLARO = "#c4cbd5"
GRIS_MASK = "#dfe3e9"
AZUL = "#1f6fb4"
TEAL = "#167d78"
NARANJA = "#d9631e"
ROJO = "#c0392b"
AZ_FILL, AZ_BORDE = "#e4eefc", "#7aa6dc"
VE_FILL = "#dff4ef"
NA_FILL = "#fdeee3"
FUENTE = "Inter"

config.background_color = FONDO

MODO_VIDEO = os.environ.get("S01_VIDEO") == "1"
try:
    if MODO_VIDEO:
        raise ImportError
    from manim_slides import Slide as _Base
    _SLIDES = True
except ImportError:
    _Base = Scene
    _SLIDES = False


class EscenaBase(_Base):
    def pausa(self, t=1.2):
        if _SLIDES:
            self.next_slide()
        else:
            self.wait(t)


def T(s, **kw):
    kw.setdefault("color", NAVY)
    return Text(s, font=FUENTE, **kw)


def M(s, **kw):
    kw.setdefault("color", NAVY)
    return MathTex(s, **kw)


def titulo(s):
    return T(s, font_size=30, weight=BOLD).to_corner(UL, buff=0.35)


def nota(s, **kw):
    kw.setdefault("font_size", 22)
    kw.setdefault("color", NAVY)
    return T(s, **kw).to_edge(DOWN, buff=0.35)


def caja(ancho, alto, fill=AZ_FILL, borde=AZUL, r=0.12):
    return RoundedRectangle(corner_radius=r, width=ancho, height=alto, stroke_color=borde,
                            stroke_width=2, fill_color=fill, fill_opacity=1)


AZULES = ["#e3eefa", "#7fb2e5", "#1f6fb4", "#123f6b"]
NARANJAS = ["#fdebdc", "#f3a46b", "#d9631e", "#8f3a0c"]


def tono(v, escala=AZULES):
    v = float(np.clip(v, 0, 1))
    return interpolate_color(ManimColor(escala[0]), ManimColor(escala[-1]), v) if len(escala) == 2 else \
        _rampa(v, escala)


def _rampa(v, escala):
    k = min(int(v * (len(escala) - 1)), len(escala) - 2)
    t = v * (len(escala) - 1) - k
    return interpolate_color(ManimColor(escala[k]), ManimColor(escala[k + 1]), t)


def columna(vec, ancho=0.42, alto=1.4, escala=AZULES, borde=NAVY, grosor=2):
    """Un vector de token: una columna de celdas (como S06-F01…F05)."""
    n = len(vec)
    celdas = VGroup(*[Rectangle(width=ancho, height=alto / n, stroke_width=0,
                                fill_color=tono(v, escala), fill_opacity=1) for v in vec])
    celdas.arrange(DOWN, buff=0)
    marco = RoundedRectangle(corner_radius=0.05, width=ancho, height=alto, stroke_color=borde,
                             stroke_width=grosor).move_to(celdas)
    return VGroup(celdas, marco)


def recolorear(col, vec, escala=AZULES):
    return [c.animate.set_fill(tono(v, escala)) for c, v in zip(col[0], vec)]


def softmax(s):
    e = np.exp(np.asarray(s, float) - np.max(s))
    return e / e.sum()


# Los vectores de make_s06_f01_f05_tokens.py
E = {
    "me": [0.10, 0.30, 0.50, 0.70], "senté": [0.85, 0.20, 0.65, 0.15], "en": [0.30, 0.60, 0.25, 0.55],
    "el": [0.25, 0.50, 0.35, 0.65], "banco": [0.15, 0.45, 0.70, 0.95],
}
FRASE = ["me", "senté", "en", "el", "banco"]


# ======================================================================
# A01 · Self-attention
# ======================================================================
class A01_SelfAttention(EscenaBase):
    def construct(self):
        tit = titulo("Self-attention: one token, then all of them")
        self.add(tit)
        xs = np.linspace(-5.2, 0.4, 5)
        y_tok = -2.35
        toks = VGroup(*[columna(E[w]).move_to([x, y_tok, 0]) for x, w in zip(xs, FRASE)])
        pal = VGroup(*[T(w, font_size=22).next_to(c, DOWN, buff=0.12) for c, w in zip(toks, FRASE)])
        toks[4][1].set_stroke(width=4.5)
        self.play(FadeIn(toks, lag_ratio=0.1), FadeIn(pal, lag_ratio=0.1))
        self.pausa()

        # q, k, v desde proyecciones fijas (aleatorias, semilla 6), escaladas a [0.1, 0.95]
        rng = np.random.default_rng(6)
        WQ, WK, WV = (rng.uniform(0, 1, (4, 4)) for _ in range(3))

        def proy(W, x):
            v = W @ np.asarray(x)
            return (v - v.min()) / (v.max() - v.min() + 1e-9) * 0.85 + 0.1

        K = [proy(WK, E[w]) for w in FRASE]
        V = [proy(WV, E[w]) for w in FRASE]
        q5 = proy(WQ, E["banco"])
        ks = VGroup(*[columna(k, 0.22, 0.75, NARANJAS, grosor=1.5).move_to([x - 0.14, y_tok + 1.35, 0])
                      for x, k in zip(xs, K)])
        vs = VGroup(*[columna(v, 0.22, 0.75, NARANJAS, grosor=1.5).move_to([x + 0.14, y_tok + 1.35, 0])
                      for x, v in zip(xs, V)])
        kl = VGroup(*[M(rf"\mathbf k_{j + 1}", font_size=22, color=NARANJA).next_to(c, UP, buff=0.06)
                      for j, c in enumerate(ks)])
        vl = VGroup(*[M(rf"\mathbf v_{j + 1}", font_size=22, color=NARANJA).next_to(c, UP, buff=0.06)
                      for j, c in enumerate(vs)])
        qc = columna(q5, 0.22, 0.75, NARANJAS, grosor=3).move_to([xs[4] + 0.55, y_tok + 1.35, 0])
        ql = M(r"\mathbf q_5", font_size=24, color=NARANJA).next_to(qc, UP, buff=0.06)
        proys = M(r"\mathbf q = W_Q\mathbf x,\quad \mathbf k = W_K\mathbf x,\quad \mathbf v = W_V\mathbf x",
                  font_size=30, color=TEAL).move_to([3.9, 2.5, 0])
        self.play(FadeIn(proys), *[TransformFromCopy(t, k) for t, k in zip(toks, ks)],
                  *[TransformFromCopy(t, v) for t, v in zip(toks, vs)], TransformFromCopy(toks[4], qc))
        self.play(FadeIn(kl), FadeIn(vl), FadeIn(ql))
        self.pausa()

        # 1. comparar
        puntajes = [-1.0, 1.2, -0.2, -0.2, 1.0]
        lineas = VGroup(*[Line(qc.get_top() + UP * 0.35, k.get_top() + UP * 0.35, stroke_color=GRIS_TXT,
                               stroke_width=1.5) for k in ks])
        cajas = VGroup()
        for x, s in zip(xs, puntajes):
            n = T(f"{s:.1f}".replace("-", "−"), font_size=22)
            b = SurroundingRectangle(n, buff=0.09, corner_radius=0.06, stroke_color=GRIS_CLARO, stroke_width=1.5)
            cajas.add(VGroup(b, n).move_to([x, y_tok + 2.55, 0]))
        f1 = M(r"\text{score}_j = \mathbf q_5^{\top}\mathbf k_j / \sqrt{d_k}", font_size=30).move_to([3.9, 1.55, 0])
        self.play(Create(lineas), FadeIn(f1))
        self.play(FadeIn(cajas, lag_ratio=0.15))
        self.pausa()

        # 2. normalizar
        w = np.round(softmax(puntajes), 2)
        base = y_tok + 2.25
        barras = VGroup(*[Rectangle(width=0.5, height=max(2.4 * wi / w.max(), 0.04), stroke_width=0,
                                    fill_color=NARANJA, fill_opacity=1).move_to([x, base, 0], aligned_edge=DOWN)
                          for x, wi in zip(xs, w)])
        nums = VGroup(*[T(f"{wi:.2f}", font_size=20).next_to(b, UP, buff=0.06) for b, wi in zip(barras, w)])
        f2 = M(r"A_{5j} = \mathrm{softmax}_j(\text{score}_j)", font_size=30).move_to([3.9, 0.85, 0])
        self.play(FadeOut(lineas), FadeOut(ks), FadeOut(kl), FadeOut(qc), FadeOut(ql),
                  ReplacementTransform(cajas, barras), FadeIn(nums), FadeIn(f2))
        self.pausa()

        # 3. promediar los values
        y5 = sum(wi * np.asarray(v) for wi, v in zip(w, V))
        out = columna(y5, 0.42, 1.4, NARANJAS, grosor=3.5).move_to([3.9, -1.55, 0])
        outl = M(r"\mathbf y_5 = \sum_j A_{5j}\,\mathbf v_j", font_size=30).next_to(out, RIGHT, buff=0.3)
        copias = [v.copy().set_opacity(0.15 + 0.85 * wi / w.max()) for v, wi in zip(vs, w)]
        self.play(*[c.animate.move_to(out).scale(0.6) for c in copias], run_time=1.4)
        self.play(*[FadeOut(c) for c in copias], FadeIn(out), FadeIn(outl))
        enfasis = T('"banco" reads mostly "senté"', font_size=24, color=TEAL).move_to([3.9, -0.15, 0])
        self.play(FadeIn(enfasis), Indicate(barras[1], color=NARANJA))
        self.pausa()

        # 4. todas las filas a la vez
        self.play(*[FadeOut(m) for m in self.mobjects if m is not tit])
        rng2 = np.random.default_rng(3)
        S = rng2.normal(0, 1.0, (5, 5))
        S[4] = puntajes
        A = np.apply_along_axis(softmax, 1, S)
        lado = 0.78
        celdas = VGroup(*[Square(lado, stroke_color=WHITE, stroke_width=2, fill_color=tono(A[i, j] / A.max(), NARANJAS),
                                 fill_opacity=1).move_to([j * lado, -i * lado, 0])
                          for i in range(5) for j in range(5)]).move_to([-1.2, -0.55, 0])
        filas = VGroup(*[T(w_, font_size=20).next_to(celdas[i * 5], LEFT, buff=0.15) for i, w_ in enumerate(FRASE)])
        cols = VGroup(*[T(w_, font_size=20).rotate(PI / 4).next_to(celdas[j], UP, buff=0.12) for j, w_ in enumerate(FRASE)])
        f3 = M(r"A = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)", font_size=34).move_to([4.2, 1.2, 0])
        f4 = M(r"Y = AV", font_size=34).next_to(f3, DOWN, buff=0.35)
        sub = T("all T × T scores in one matrix product", font_size=22, color=GRIS_TXT).next_to(f4, DOWN, buff=0.35)
        self.play(FadeIn(celdas, lag_ratio=0.02), FadeIn(filas), FadeIn(cols), FadeIn(f3), FadeIn(f4), FadeIn(sub))
        fila5 = SurroundingRectangle(VGroup(*celdas[20:25]), buff=0.03, stroke_color=NAVY, stroke_width=3.5)
        self.play(Create(fila5))
        self.pausa()

        # 5. máscara causal
        Sc = np.where(np.triu(np.ones((5, 5)), 1) > 0, -np.inf, S)
        Ac = np.apply_along_axis(softmax, 1, Sc)
        anim, cruces = [], VGroup()
        for i in range(5):
            for j in range(5):
                c = celdas[i * 5 + j]
                if j > i:
                    anim.append(c.animate.set_fill(GRIS_MASK))
                    cruces.add(Cross(c, stroke_color=ROJO, stroke_width=2.5, scale_factor=0.35))
                else:
                    anim.append(c.animate.set_fill(tono(Ac[i, j] / Ac.max(), NARANJAS)))
        f5 = M(r"A = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)", font_size=34).move_to(f3)
        caus = T("causal mask: token i reads only j ≤ i", font_size=22, color=ROJO).move_to(sub)
        self.play(*anim, FadeIn(cruces), Transform(f3, f5), Transform(sub, caus), FadeOut(fila5), run_time=1.6)
        self.pausa(2.5)


# ======================================================================
# A02 · One pre-norm block
# ======================================================================
def layer_norm(X):
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-5)


class A02_Block(EscenaBase):
    def construct(self):
        tit = titulo("One Transformer block")
        self.add(tit)
        rng = np.random.default_rng(0)
        X0 = np.array([E[w] for w in FRASE]) * 2 - 1                     # valores en [−1, 1]
        Wa = rng.normal(0, 0.6, (4, 4))
        S = (X0 @ Wa) @ X0.T
        A = np.apply_along_axis(softmax, 1, S)
        X1 = X0 + 0.8 * A @ layer_norm(X0) @ rng.normal(0, 0.5, (4, 4))  # X ← X + MHA(LN(X))
        W1, W2 = rng.normal(0, 0.8, (4, 16)), rng.normal(0, 0.4, (16, 4))
        gelu = lambda z: 0.5 * z * (1 + np.tanh(0.79788 * (z + 0.044715 * z ** 3)))
        X2 = X1 + gelu(layer_norm(X1) @ W1) @ W2                         # X ← X + MLP(LN(X))
        lo, hi = min(X0.min(), X1.min(), X2.min()), max(X0.max(), X1.max(), X2.max())
        norm = lambda X: (X - lo) / (hi - lo)

        xs = np.linspace(-4.6, -0.2, 5)
        y = -1.9
        cols = VGroup(*[columna(v).move_to([x, y, 0]) for x, v in zip(xs, norm(X0))])
        pal = VGroup(*[T(w, font_size=21).next_to(c, DOWN, buff=0.12) for c, w in zip(cols, FRASE)])
        banda = RoundedRectangle(corner_radius=0.2, width=5.4, height=5.9, stroke_width=0, fill_color=AZ_FILL,
                                 fill_opacity=0.55).move_to([-2.4, 0.15, 0])
        etq = T("residual stream: one vector per token", font_size=20, color=AZUL).next_to(banda, UP, buff=0.08)
        self.play(FadeIn(banda), FadeIn(etq), FadeIn(cols, lag_ratio=0.1), FadeIn(pal))
        self.pausa()

        pasos = VGroup(*[T(s, font_size=24, color=c) for s, c in [
            ("LN", GRIS_TXT), ("attention", NARANJA), ("+", NAVY), ("LN", GRIS_TXT), ("MLP", TEAL), ("+", NAVY)]])
        pasos.arrange(UP, buff=0.42).move_to([2.2, 0.0, 0])
        cajas = VGroup(*[SurroundingRectangle(p, buff=0.12, corner_radius=0.08, stroke_color=GRIS_CLARO, stroke_width=1.5)
                         for p in pasos])
        self.play(FadeIn(pasos), FadeIn(cajas))
        marca = SurroundingRectangle(cajas[0], buff=0.05, stroke_color=NAVY, stroke_width=3)

        # LN + atención: los tokens intercambian información
        self.play(Create(marca))
        self.play(*[c[0].animate.set_opacity(0.55) for c in cols], run_time=0.5)
        self.play(*[c[0].animate.set_opacity(1) for c in cols], run_time=0.5)
        self.play(marca.animate.move_to(cajas[1]))
        arcos = VGroup()
        for i in range(5):
            for j in range(5):
                if i != j and A[i, j] > 0.12:
                    arcos.add(CurvedArrow(cols[j].get_top() + UP * 0.08, cols[i].get_top() + UP * 0.08,
                                          angle=-0.9 if j < i else 0.9, color=NARANJA,
                                          stroke_width=1 + 6 * A[i, j], tip_length=0.14))
        self.play(LaggedStart(*[Create(a) for a in arcos], lag_ratio=0.08), run_time=1.6)
        mezcla = T("attention mixes information ACROSS tokens", font_size=22, color=NARANJA).move_to([-2.4, 3.0, 0])
        self.play(FadeIn(mezcla))
        self.play(marca.animate.move_to(cajas[2]), *sum([recolorear(c, v) for c, v in zip(cols, norm(X1))], []),
                  FadeOut(arcos), run_time=1.4)
        self.pausa()

        # LN + MLP: cada token por su cuenta
        self.play(marca.animate.move_to(cajas[3]), FadeOut(mezcla))
        self.play(*[c[0].animate.set_opacity(0.55) for c in cols], run_time=0.5)
        self.play(*[c[0].animate.set_opacity(1) for c in cols], run_time=0.5)
        self.play(marca.animate.move_to(cajas[4]))
        mlps = VGroup(*[caja(0.62, 0.42, VE_FILL, TEAL).next_to(c, UP, buff=0.3) for c in cols])
        mlpl = VGroup(*[T("MLP", font_size=14, color=TEAL).move_to(b) for b in mlps])
        sola = T("the MLP transforms EACH token on its own", font_size=22, color=TEAL).move_to([-2.4, 3.0, 0])
        self.play(FadeIn(mlps), FadeIn(mlpl), FadeIn(sola))
        self.play(*[Indicate(b, color=TEAL, scale_factor=1.15) for b in mlps], run_time=1.0)
        self.play(marca.animate.move_to(cajas[5]), *sum([recolorear(c, v) for c, v in zip(cols, norm(X2))], []),
                  FadeOut(mlps), FadeOut(mlpl), run_time=1.4)
        self.pausa()
        fin = M(r"\text{output: still } (T, d)\text{, ready for the next block}", font_size=28).to_edge(DOWN, buff=0.3)
        self.play(FadeOut(marca), FadeIn(fin))
        self.pausa(2.5)


# ======================================================================
# A03 · Generating text
# ======================================================================
def barras_dist(palabras, probs, ancho_max=3.2, alto=0.36, color=TEAL):
    filas = VGroup()
    for w, p in zip(palabras, probs):
        b = Rectangle(width=max(ancho_max * p, 0.03), height=alto, stroke_width=0, fill_color=color, fill_opacity=1)
        etiqueta = T(w, font_size=20).next_to(b, LEFT, buff=0.15)
        num = T(f"{p:.2f}", font_size=18, color=GRIS_TXT).next_to(b, RIGHT, buff=0.12)
        filas.add(VGroup(etiqueta, b, num))
    filas.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
    for f in filas:                                   # barras alineadas a la izquierda
        f[1].align_to(filas[0][1], LEFT)
        f[2].next_to(f[1], RIGHT, buff=0.12)
        f[0].next_to(f[1], LEFT, buff=0.15)
    return filas


class A03_Generation(EscenaBase):
    def construct(self):
        tit = titulo("Generating text, one token at a time")
        self.add(tit)

        def chip(w, color=AZUL, fill=AZ_FILL):
            t = T(w, font_size=22)
            return VGroup(RoundedRectangle(corner_radius=0.08, width=max(t.width + 0.3, 0.7), height=0.55,
                                           stroke_color=color, stroke_width=2, fill_color=fill, fill_opacity=1), t)

        fila = VGroup(*[chip(w) for w in ["me", "senté", "en", "el"]]).arrange(RIGHT, buff=0.12)
        fila.move_to([-3.6, -2.5, 0])
        modelo = caja(3.2, 1.2).move_to([-3.6, -0.4, 0])
        ml = VGroup(T("decoder", font_size=26, weight=BOLD), T("(causal mask)", font_size=18)).arrange(DOWN, buff=0.06).move_to(modelo)
        flecha_in = Arrow(fila.get_top(), modelo.get_bottom(), buff=0.08, color=NAVY, stroke_width=3)
        self.play(FadeIn(fila, lag_ratio=0.1), FadeIn(modelo), FadeIn(ml), GrowArrow(flecha_in))
        self.pausa()

        # paso 1: la distribución del último token
        c1 = ["banco", "sofá", "suelo", "parque", "."]
        logit1 = np.log([0.46, 0.18, 0.14, 0.12, 0.10])
        p1 = softmax(logit1)
        dist = barras_dist(c1, p1).move_to([3.0, 0.4, 0])
        dl = M(r"p(w_{T+1}\mid w_{\le T})", font_size=30).next_to(dist, UP, buff=0.35)
        flecha_out = Arrow(modelo.get_right(), dist.get_left() + LEFT * 0.9, buff=0.1, color=NAVY, stroke_width=3)
        sololast = T("logits of the last position → softmax", font_size=18, color=GRIS_TXT).next_to(flecha_out, DOWN, buff=0.08)
        self.play(GrowArrow(flecha_out), FadeIn(sololast), FadeIn(dl), FadeIn(dist, lag_ratio=0.15))
        self.pausa()
        elegido = SurroundingRectangle(dist[0], buff=0.06, stroke_color=NARANJA, stroke_width=3)
        sam = T("sample", font_size=22, color=NARANJA).next_to(elegido, RIGHT, buff=0.3)
        self.play(Create(elegido), FadeIn(sam))
        nuevo = chip("banco", NARANJA, NA_FILL).move_to(dist[0][0])
        self.play(nuevo.animate.next_to(fila, RIGHT, buff=0.12), run_time=1.2)
        fila.add(nuevo)
        self.play(fila.animate.move_to([-3.6, -2.5, 0]), FadeOut(elegido), FadeOut(sam))
        self.pausa()

        # paso 2: la ventana creció un token
        c2 = ["y", "de", ".", "a", ","]
        p2 = np.array([0.31, 0.22, 0.18, 0.09, 0.06])
        dist2 = barras_dist(c2, p2).move_to(dist, aligned_edge=LEFT)
        self.play(Transform(dist, dist2))
        elegido = SurroundingRectangle(dist[0], buff=0.06, stroke_color=NARANJA, stroke_width=3)
        self.play(Create(elegido))
        nuevo2 = chip("y", NARANJA, NA_FILL).move_to(dist[0][0])
        self.play(nuevo2.animate.next_to(fila, RIGHT, buff=0.12), FadeOut(elegido), run_time=1.0)
        fila.add(nuevo2)
        self.play(fila.animate.move_to([-3.6, -2.5, 0]))
        lazo = T("the next input is the token just produced", font_size=22, color=NARANJA).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(lazo))
        self.pausa()

        # temperatura: los mismos logits del paso 1
        self.play(FadeOut(dist), FadeOut(lazo), FadeOut(sololast))
        tau_t = M(r"\tau = 1.0", font_size=34, color=NARANJA).move_to([3.0, -2.2, 0])
        d1 = barras_dist(c1, p1).move_to([3.0, 0.4, 0])
        self.play(FadeIn(d1), FadeIn(tau_t))
        self.pausa()
        for tau, txt in [(0.5, "low τ: sharper, safe, repetitive"), (1.6, "high τ: flatter, varied, incoherent")]:
            pt = softmax(logit1 / tau)
            nd = barras_dist(c1, pt).move_to(d1, aligned_edge=LEFT)
            nt = M(rf"\tau = {tau}", font_size=34, color=NARANJA).move_to(tau_t)
            lab = T(txt, font_size=22, color=NARANJA).to_edge(DOWN, buff=0.3)
            self.play(Transform(d1, nd), Transform(tau_t, nt), FadeIn(lab), run_time=1.3)
            self.pausa()
            self.play(FadeOut(lab))
        form = M(r"p \propto \exp(z / \tau)", font_size=32).next_to(tau_t, UP, buff=0.3)
        self.play(FadeIn(form))
        self.pausa(2.0)


# ======================================================================
# A04 · The masked next-token loss
# ======================================================================
class A04_MaskedLoss(EscenaBase):
    def construct(self):
        tit = titulo("The fine-tuning loss, token by token")
        self.add(tit)
        toks = ["<user>", "¿Qué", "es", "un", "átomo?", "<assistant>", "Partícula", "mínima", "<end>"]
        rol = ["t", "q", "q", "q", "q", "t", "a", "a", "a"]
        estilo = {"t": (GRIS_TXT, GRIS_MASK), "q": (AZUL, AZ_FILL), "a": (TEAL, VE_FILL)}
        chips = VGroup()
        for w, r in zip(toks, rol):
            t = T(w, font_size=19)
            borde, fill = estilo[r]
            chips.add(VGroup(RoundedRectangle(corner_radius=0.07, width=t.width + 0.22, height=0.5, stroke_color=borde,
                                              stroke_width=2, fill_color=fill, fill_opacity=1), t))
        chips.arrange(RIGHT, buff=0.1).move_to([0, -2.85, 0])
        if chips.width > 13.2:
            chips.scale_to_fit_width(13.2)
        leyenda = VGroup(*[VGroup(Square(0.22, stroke_color=b, fill_color=f, fill_opacity=1, stroke_width=2),
                                  T(n, font_size=17, color=GRIS_TXT)).arrange(RIGHT, buff=0.1)
                           for (b, f), n in [(estilo["t"], "template"), (estilo["q"], "question"), (estilo["a"], "answer")]])
        leyenda.arrange(RIGHT, buff=0.4).to_corner(UR, buff=0.4)
        self.play(FadeIn(chips, lag_ratio=0.08), FadeIn(leyenda))
        self.pausa()

        # una distribución por posición, desplazada un paso
        p = [0.20, 0.60, 0.50, 0.01, 0.98, 0.05, 0.10, 0.30]
        L = -np.log(p)
        flechas = VGroup(*[CurvedArrow(chips[t].get_top() + UP * 0.05, chips[t + 1].get_top() + UP * 0.05, angle=-1.3,
                                       color=GRIS_TXT, stroke_width=2, tip_length=0.12) for t in range(8)])
        msg = M(r"\text{position } t \text{ predicts token } t+1:\quad L_t = -\log p(w_{t+1}\mid w_{\le t})", font_size=28)
        msg.move_to([0, 2.55, 0])
        self.play(LaggedStart(*[Create(f) for f in flechas], lag_ratio=0.1), FadeIn(msg), run_time=1.6)
        self.pausa()

        # p(objetivo) → barras L_t sobre el token objetivo
        base_y = -2.15
        escala = 0.62
        barras, nums = VGroup(), VGroup()
        for t in range(8):
            obj = chips[t + 1]
            color = estilo[rol[t + 1]][0]
            b = Rectangle(width=min(obj.width * 0.6, 0.6), height=max(L[t] * escala, 0.03), stroke_width=0,
                          fill_color=color, fill_opacity=0.85).move_to([obj.get_x(), base_y, 0], aligned_edge=DOWN)
            barras.add(b)
            nums.add(T(f"{L[t]:.2f}", font_size=17).next_to(b, UP, buff=0.05))
        self.play(FadeOut(flechas), LaggedStart(*[GrowFromEdge(b, DOWN) for b in barras], lag_ratio=0.12),
                  FadeIn(nums, lag_ratio=0.12), run_time=1.8)
        self.pausa()

        def linea(valor, color, texto):
            y = base_y + valor * escala
            l = DashedLine([chips[1].get_left()[0] - 0.2, y, 0], [chips[-1].get_right()[0] + 0.1, y, 0],
                           color=color, stroke_width=2.5, dash_length=0.12)
            return VGroup(l, T(texto, font_size=19, color=color).next_to(l, RIGHT, buff=0.1).shift(LEFT * 1.6 + UP * 0.22))

        todo = linea(L.mean(), GRIS_TXT, f"mean of all: {L.mean():.2f}")
        self.play(Create(todo))
        aviso = T('the largest loss is on "átomo?": the user\'s question', font_size=20, color=ROJO).move_to([0, 1.85, 0])
        self.play(FadeIn(aviso), Indicate(barras[3], color=ROJO))
        self.pausa()

        # máscara: solo la respuesta
        m_txt = M(r"m_t = 1 \text{ only if } w_{t+1} \text{ is in the answer:}\quad J = \frac{\sum_t m_t L_t}{\sum_t m_t}",
                  font_size=28).move_to(msg)
        self.play(Transform(msg, m_txt), FadeOut(aviso), FadeOut(todo),
                  *[barras[t].animate.set_fill(GRIS_CLARO, 0.6) for t in range(5)],
                  *[nums[t].animate.set_color(GRIS_CLARO) for t in range(5)], run_time=1.4)
        J = L[5:].mean()
        lj = linea(J, TEAL, f"J = {J:.2f}")
        self.play(Create(lj))
        self.pausa()

        # después de LoRA (ilustrativo)
        p_post = [0.40, 0.60, 0.80]
        L_post = -np.log(p_post)
        nuevas = [Rectangle(width=barras[5 + k].width, height=L_post[k] * escala, stroke_width=0, fill_color=TEAL,
                            fill_opacity=0.85).move_to(barras[5 + k], aligned_edge=DOWN) for k in range(3)]
        lj2 = linea(L_post.mean(), TEAL, f"J = {L_post.mean():.2f}")
        tras = T("after LoRA (illustrative): the answer becomes likely", font_size=21, color=TEAL).move_to([0, 1.85, 0])
        self.play(*[Transform(barras[5 + k], nuevas[k]) for k in range(3)],
                  *[nums[5 + k].animate.become(T(f"{L_post[k]:.2f}", font_size=17).next_to(nuevas[k], UP, buff=0.05))
                    for k in range(3)],
                  Transform(lj, lj2), FadeIn(tras), run_time=1.6)
        self.pausa(2.5)
