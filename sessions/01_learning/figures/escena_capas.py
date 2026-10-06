"""
E8_CapasTransformanEspacio · SI7011
Cómo cada capa de una MLP angosta (2–2–2–2–1, tanh) deforma el plano de
entrada hasta que, en la última capa oculta, las clases quedan separadas
por una recta.

Cada capa se anima en dos tiempos:
    z_l = W_l a_{l-1} + b_l   (afín: las rectas de la rejilla siguen rectas)
    a_l = tanh(z_l)           (no lineal: la rejilla se curva; tanh es un caso de φ)
Al final se dibuja la recta w₄ᵀa₃ + b₄ = 0 y se recorre la red hacia atrás,
de modo que esa recta se convierte en la frontera curva en el espacio x.

La red está ENTRENADA (numpy, BCE, Adam, init ortogonal, parada temprana)
con semilla 8 sobre dos lunas; llega a 100 % en entrenamiento. Todas las
capas tienen 2 unidades para que cada espacio intermedio se pueda dibujar.

Render:
    S01_VIDEO=1 manim -qh escena_capas.py E8_CapasTransformanEspacio
    manim -qh escena_capas.py E8_CapasTransformanEspacio      # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo
FONDO = WHITE
TINTA = "#222222"
GRIS = "#9a9a9a"
REJILLA = "#c9c9c9"
AZUL = "#1f6fb4"
ROJO = "#d1392b"
NARANJA = "#f07c12"
FUENTE = "sans-serif"

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


# ---------------------------------------------------------------- datos y red
def datos_lunas(n=100, ruido=0.1, semilla=0):
    r = np.random.default_rng(semilla)
    t1, t2 = r.uniform(0, np.pi, n), r.uniform(0, np.pi, n)
    a = np.c_[np.cos(t1), np.sin(t1)]
    b = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.r_[a, b] + r.normal(0, ruido, (2 * n, 2))
    y = np.r_[np.zeros(n), np.ones(n)]
    return X - X.mean(0), y


def adelante(P, X):
    H = [X]
    for W, b in P[:-1]:
        H.append(np.tanh(H[-1] @ W.T + b))
    return H, (H[-1] @ P[-1][0].T + P[-1][1]).ravel()


def entrenar_mlp(X, y, capas=(2, 2, 2), semilla=8, lr=0.01, margen=1.0, extra=300, max_pasos=20000):
    """BCE desde logits + Adam. Para cuando todo punto tiene |logit| > margen (+ `extra` pasos)."""
    r = np.random.default_rng(semilla)
    dims = [2, *capas, 1]
    P = []
    for i, o in zip(dims[:-1], dims[1:]):
        Q, _ = np.linalg.qr(r.normal(size=(max(i, o), max(i, o))))
        P.append([Q[:o, :i] * 1.2, r.normal(0, 0.1, o)])
    m = [[0 * W, 0 * b] for W, b in P]
    v = [[0 * W, 0 * b] for W, b in P]
    fin = None
    for t in range(1, max_pasos):
        H, z = adelante(P, X)
        p = 1 / (1 + np.exp(-z))
        g = ((p - y) / len(y))[:, None]
        grads = []
        for k in range(len(P) - 1, -1, -1):
            W, _ = P[k]
            grads.append((g.T @ H[k], g.sum(0)))
            if k > 0:
                g = (g @ W) * (1 - H[k] ** 2)
        for k, gg in enumerate(grads[::-1]):
            for j in (0, 1):
                m[k][j] = 0.9 * m[k][j] + 0.1 * gg[j]
                v[k][j] = 0.999 * v[k][j] + 0.001 * gg[j] ** 2
                P[k][j] -= lr * (m[k][j] / (1 - 0.9 ** t)) / (np.sqrt(v[k][j] / (1 - 0.999 ** t)) + 1e-8)
        s = np.where(y == 1, z, -z)
        if fin is None and s.min() > margen:
            fin = t + extra
        if fin and t >= fin:
            break
    _, z = adelante(P, X)
    return P, float(((z > 0) == y).mean())


# ---------------------------------------------------------------- geometría de la animación
PANEL_CENTRO = np.array([-2.75, -0.3, 0.0])
PANEL_LADO = 5.75
GX = np.linspace(-2.1, 2.1, 15)
GY = np.linspace(-1.5, 1.5, 11)
M_REJ = 160


def lineas_rejilla():
    t_x = np.linspace(GX[0], GX[-1], M_REJ)
    t_y = np.linspace(GY[0], GY[-1], M_REJ)
    L = [np.c_[t_x, np.full(M_REJ, yy)] for yy in GY]
    L += [np.c_[np.full(M_REJ, xx), t_y] for xx in GX]
    return np.stack(L)  # (n_lineas, M_REJ, 2)


def vista(R):
    """(centro, escala) que encaja la rejilla en el panel conservando el aspecto."""
    pts = R.reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    return (lo + hi) / 2, 0.92 * PANEL_LADO / max(hi - lo)


class E8_CapasTransformanEspacio(EscenaBase):
    def construct(self):
        Text.set_default(color=TINTA, font=FUENTE)
        MathTex.set_default(color=TINTA)

        # ----- red entrenada y todas las etapas precomputadas
        X, y = datos_lunas()
        P, acc = entrenar_mlp(X, y)
        n_capas = len(P) - 1

        def logit(xy):
            return adelante(P, xy)[1]

        # frontera en x (curva de nivel 0), en coordenadas de datos
        imp = ImplicitFunction(
            lambda a, b: logit(np.array([[a, b]]))[0],
            x_range=[GX[0], GX[-1]], y_range=[GY[0], GY[-1]],
            min_depth=6, max_quads=4000,
        )
        tramos = [np.array(sm.get_anchors())[:, :2] for sm in imp.family_members_with_points()]
        tramos = [t for t in tramos if len(t) > 8]

        etapas = []  # cada etapa: dict(pts, rej, fr, tipo)
        pts, rej, fr = X, lineas_rejilla(), tramos
        etapas.append(dict(pts=pts, rej=rej, fr=fr, tipo="x"))
        for W, b in P[:-1]:
            aff = lambda A: A @ W.T + b
            pts, rej, fr = aff(pts), aff(rej), [aff(t) for t in fr]
            etapas.append(dict(pts=pts, rej=rej, fr=fr, tipo="z"))
            pts, rej, fr = np.tanh(pts), np.tanh(rej), [np.tanh(t) for t in fr]
            etapas.append(dict(pts=pts, rej=rej, fr=fr, tipo="h"))
        for e in etapas:
            e["vista"] = vista(e["rej"])

        # ----- mobjects del panel (se actualizan a partir de un estado interpolado)
        def a_pantalla(A, c, s):
            A = (A - c) * s
            return np.c_[A, np.zeros(len(A))] + PANEL_CENTRO

        borde = Square(PANEL_LADO, color=GRIS, stroke_width=1.5).move_to(PANEL_CENTRO)
        rejilla = VGroup(*[VMobject(stroke_color=REJILLA, stroke_width=1.6) for _ in range(len(GX) + len(GY))])
        ejes0 = VGroup(*[DashedLine(LEFT, RIGHT, color=GRIS, stroke_width=1.5, dash_length=0.06) for _ in range(2)])
        puntos = VGroup(*[
            Dot(radius=0.055, color=(AZUL if yi == 1 else ROJO)) for yi in y
        ])
        frontera = VGroup(*[VMobject(stroke_color=NARANJA, stroke_width=6) for _ in tramos])

        estado = {"i": 0, "t": 0.0}

        def interpolar(i, t):
            A, B = etapas[i], etapas[min(i + 1, len(etapas) - 1)]
            mix = lambda a, b: (1 - t) * a + t * b
            c = mix(A["vista"][0], B["vista"][0])
            s = 1 / mix(1 / A["vista"][1], 1 / B["vista"][1])
            return mix(A["pts"], B["pts"]), mix(A["rej"], B["rej"]), \
                [mix(a, b) for a, b in zip(A["fr"], B["fr"])], c, s

        def dibujar(_=None):
            p, R, F, c, s = interpolar(estado["i"], estado["t"])
            for d, q in zip(puntos, a_pantalla(p, c, s)):
                d.move_to(q)
            for linea, r in zip(rejilla, R):
                linea.set_points_as_corners(a_pantalla(r, c, s))
            for linea, f in zip(frontera, F):
                linea.set_points_as_corners(a_pantalla(f, c, s))
            # ejes del espacio actual (por el origen), recortados al panel
            o = a_pantalla(np.zeros((1, 2)), c, s)[0]
            h = PANEL_LADO / 2
            ox = np.clip(o[0], PANEL_CENTRO[0] - h, PANEL_CENTRO[0] + h)
            oy = np.clip(o[1], PANEL_CENTRO[1] - h, PANEL_CENTRO[1] + h)
            ejes0[0].put_start_and_end_on([PANEL_CENTRO[0] - h, oy, 0], [PANEL_CENTRO[0] + h, oy, 0])
            ejes0[1].put_start_and_end_on([ox, PANEL_CENTRO[1] - h, 0], [ox, PANEL_CENTRO[1] + h, 0])

        dibujar()

        def avanzar(i, run_time, rate_func=smooth, atras=False):
            def upd(_, a):
                estado["i"], estado["t"] = (i - 1, 1 - a) if atras else (i, a)
                dibujar()
            return UpdateFromAlphaFunc(VGroup(), upd, run_time=run_time, rate_func=rate_func)

        # ----- columna derecha: cadena de capas, ecuación, nota
        COL_X = 3.75
        nombres = [r"x"] + [rf"a_{{{l}}}" for l in range(1, n_capas + 1)] + [r"\hat{y}"]
        chips = VGroup()
        for nm in nombres:
            tx = MathTex(nm, font_size=34)
            caja = RoundedRectangle(width=0.95, height=0.7, corner_radius=0.12,
                                    stroke_color=GRIS, stroke_width=1.5)
            chips.add(VGroup(caja, tx))
        chips.arrange(RIGHT, buff=0.32).move_to([COL_X, 2.85, 0])
        flechas = VGroup(*[
            Arrow(a.get_right(), b.get_left(), buff=0.04, stroke_width=2.5, color=GRIS,
                  max_tip_length_to_length_ratio=0.35)
            for a, b in zip(chips[:-1], chips[1:])
        ])
        resalte = SurroundingRectangle(chips[0], color=NARANJA, buff=0.06, corner_radius=0.14, stroke_width=4)

        titulo = Text("Each layer transforms the space", font_size=34, weight=BOLD).to_corner(UL, buff=0.35)

        def ecuacion(tex, sub):
            e = MathTex(tex, font_size=40)
            n = Text(sub, font_size=22, color=GRIS, line_spacing=0.9)
            return VGroup(e, n).arrange(DOWN, buff=0.3).move_to([COL_X, 1.1, 0])

        def eq_capa(l, tipo):
            prev = r"x" if l == 1 else rf"a_{{{l-1}}}"
            if tipo == "z":
                return ecuacion(rf"z_{{{l}}} = W_{{{l}}}{prev} + b_{{{l}}}",
                                "affine: rotates, stretches and shifts;\nstraight lines stay straight")
            return ecuacion(rf"a_{{{l}}} = \tanh(z_{{{l}}})",
                            "tanh: bends the plane and\nsquashes it into (−1, 1)²")

        def nota(txt):
            return Text(txt, font_size=24, line_spacing=0.95).move_to([COL_X, -1.6, 0])

        # =============================================================
        # 1 · Espacio de entrada
        # =============================================================
        self.play(FadeIn(titulo), Create(borde), run_time=0.8)
        self.play(Create(rejilla), FadeIn(ejes0), run_time=1.4)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in puntos], lag_ratio=0.004), run_time=1.4)
        self.play(FadeIn(chips), FadeIn(flechas), Create(resalte))

        eq = ecuacion(r"x \in \mathbb{R}^2", "input space")
        nt = nota("No straight line separates\nthe two classes in x.")
        self.play(FadeIn(eq), FadeIn(nt))

        # intento de recta que gira y no logra separar
        L = 4.2
        prueba = DashedLine(LEFT * L, RIGHT * L, color=TINTA, stroke_width=3, dash_length=0.12)
        prueba.move_to(PANEL_CENTRO)
        self.play(Create(prueba), run_time=0.6)
        self.play(Rotate(prueba, PI, about_point=PANEL_CENTRO), run_time=3.0, rate_func=linear)
        self.play(FadeOut(prueba), run_time=0.4)
        self.pausa(1.0)

        # =============================================================
        # 2 · Capa por capa
        # =============================================================
        for l in range(1, n_capas + 1):
            i_z = 2 * l - 2  # etapa de partida (x o h⁽ˡ⁻¹⁾)
            eq_z = eq_capa(l, "z")
            self.play(
                FadeOut(eq), FadeIn(eq_z),
                resalte.animate.move_to(chips[l]).set_stroke(opacity=0.5),
                FadeOut(nt) if l == 1 else Wait(0),
                run_time=0.7,
            )
            eq = eq_z
            self.play(avanzar(i_z, run_time=2.4))
            self.pausa(0.6)

            eq_h = eq_capa(l, "h")
            self.play(FadeOut(eq), FadeIn(eq_h), resalte.animate.set_stroke(opacity=1), run_time=0.6)
            eq = eq_h
            self.play(avanzar(i_z + 1, run_time=2.8))
            if l == 1:
                nt = nota("The moons start\nto untangle.")
                self.play(FadeIn(nt))
            elif l < n_capas:
                self.play(FadeOut(nt))
                nt = nota("Each layer keeps\npulling the classes apart.")
                self.play(FadeIn(nt))
            self.pausa(1.0)

        # =============================================================
        # 3 · En la última capa basta una recta
        # =============================================================
        estado["i"], estado["t"] = len(etapas) - 1, 0.0
        eq_y = ecuacion(rf"\hat{{y}} = \sigma\!\left(w_{{{n_capas+1}}}^\top a_{{{n_capas}}} + b_{{{n_capas+1}}}\right)",
                        f"the output layer is linear:\none straight line in a{chr(0x2080+n_capas)}")
        nt_new = nota("In the last hidden layer\nthe classes are linearly\nseparable.")
        self.play(FadeOut(eq), FadeIn(eq_y), FadeOut(nt), FadeIn(nt_new),
                  resalte.animate.move_to(chips[-1]), run_time=0.8)
        eq, nt = eq_y, nt_new

        dibujar()
        self.play(Create(frontera), run_time=1.6)
        self.pausa(2.0)

        # =============================================================
        # 4 · De vuelta al espacio de entrada
        # =============================================================
        nt_new = nota("The same line, seen in x,\nis a curved boundary.")
        self.play(FadeOut(nt), FadeIn(nt_new), resalte.animate.move_to(chips[0]), run_time=0.6)
        nt = nt_new
        for i in range(len(etapas) - 1, 0, -1):
            self.play(avanzar(i, run_time=1.1, atras=True, rate_func=smooth))
        self.wait(1.0)

        cierre = Text("Hidden layers bend the space; the output layer only draws a straight line.",
                      font_size=26).to_edge(DOWN, buff=0.25)
        precision = Text(f"Trained 2–2–2–2–1 MLP · training accuracy {acc:.0%}", font_size=20, color=GRIS)
        precision.next_to(borde, UP, buff=0.12).align_to(borde, RIGHT)
        self.play(FadeIn(cierre, shift=UP * 0.2), FadeIn(precision))
        self.wait(3.0)
