"""
E7_AproximacionReLU · SI7011
Suma de componentes ReLU que aproxima una curva suave (1 → 2 → 4 → 8 → 16).

La aproximación con K componentes es la interpolante lineal por tramos de f
en K + 1 nodos equiespaciados, escrita exactamente como
    g_K(x) = c + Σ_{i=1}^{K} a_i ReLU(x − b_i),
con b_i = nodos[0..K-1], c = f(b_1) y a_i = cambio de pendiente en b_i.
Los nodos están anidados: al duplicar K, los quiebres nuevos son los puntos
medios de los anteriores. No hay entrenamiento: solo crece la capacidad.

Render:
    S01_VIDEO=1 manim -qh escena_relu.py E7_AproximacionReLU   # video lineal
    manim -qh escena_relu.py E7_AproximacionReLU               # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo
FONDO = WHITE
TINTA = "#222222"
GRIS = "#9a9a9a"
AZUL = "#1f6fb4"      # curva objetivo
VERDE = "#2e9e44"     # aproximación
NARANJA = "#f07c12"   # quiebres nuevos
FUENTE = "sans-serif"

config.background_color = FONDO

# ---------------------------------------------------------------- base con pausas
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


# ---------------------------------------------------------------- matemática
X_MIN, X_MAX = -3.0, 3.0
N_MUESTRAS = 961  # paso 6/960 = 0.00625: incluye exactamente todos los nodos de K = 16


def f(x):
    return np.sin(1.3 * x + 0.6) + 0.3 * x


def nodos(K):
    return np.linspace(X_MIN, X_MAX, K + 1)


def coeficientes(K):
    """c, a_i, b_i de g_K(x) = c + Σ a_i ReLU(x − b_i)."""
    n = nodos(K)
    y = f(n)
    pend = np.diff(y) / np.diff(n)
    a = np.diff(np.concatenate([[0.0], pend]))
    return y[0], a, n[:-1]


def g(K, x):
    c, a, b = coeficientes(K)
    return c + (a[None, :] * np.maximum(0.0, x[:, None] - b[None, :])).sum(axis=1)


XS = np.linspace(X_MIN, X_MAX, N_MUESTRAS)


class E7_AproximacionReLU(EscenaBase):
    def construct(self):
        Text.set_default(color=TINTA, font=FUENTE)

        # ---------- ejes compartidos por toda la secuencia
        ejes = Axes(
            x_range=[-3.2, 3.2, 1],
            y_range=[-2, 2, 1],
            x_length=10.5,
            y_length=5.0,
            tips=False,
            axis_config={"color": TINTA, "stroke_width": 2, "include_ticks": False},
        ).shift(DOWN * 0.55)
        rot_x = Text("x", font_size=28, slant=ITALIC).next_to(ejes.x_axis.get_end(), DR, buff=0.1)

        def curva_g(K, color=VERDE):
            pts = [ejes.c2p(x, y) for x, y in zip(XS, g(K, XS))]
            return VMobject(stroke_color=color, stroke_width=5).set_points_as_corners(pts)

        objetivo = ejes.plot(f, x_range=[X_MIN, X_MAX, 0.01], color=AZUL, stroke_width=7)
        objetivo.set_stroke(opacity=0.85)

        # leyenda
        def item(color, texto):
            linea = Line(ORIGIN, RIGHT * 0.55, color=color, stroke_width=6)
            return VGroup(linea, Text(texto, font_size=24)).arrange(RIGHT, buff=0.2)

        leyenda = VGroup(
            item(AZUL, "función objetivo"),
            item(VERDE, "suma de ReLU"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        # zona vacía abajo a la derecha (la curva está cerca de 0 para x > 1.5)
        leyenda.move_to(ejes.c2p(2.0, -1.45))

        # contador
        k_val = ValueTracker(1)
        contador_rot = Text("componentes ReLU:", font_size=28)
        contador_num = Integer(1, font_size=44, color=VERDE)
        contador = VGroup(contador_rot, contador_num).arrange(RIGHT, buff=0.25)
        contador.to_corner(UR, buff=0.4)
        contador_num.add_updater(lambda m: m.set_value(round(k_val.get_value())))

        # marcas de quiebre acumuladas sobre el eje x
        marcas = VGroup()

        def marca(x, color):
            p = ejes.c2p(x, 0)
            return Line(p + DOWN * 0.13, p + UP * 0.13, color=color, stroke_width=5)

        def titulo(texto):
            return Text(texto, font_size=30, weight=BOLD).to_corner(UL, buff=0.4)

        # =============================================================
        # Escena 1 · curva objetivo y aproximación inicial
        # =============================================================
        t1 = titulo("1 · Curva objetivo y aproximación inicial")
        self.play(FadeIn(t1), Create(ejes), FadeIn(rot_x), run_time=1.2)
        self.play(Create(objetivo), run_time=1.8)

        K = 1
        aprox = curva_g(K)
        marcas.add(marca(nodos(K)[0], GRIS))
        self.play(Create(aprox), FadeIn(marcas), run_time=1.5)
        self.play(FadeIn(leyenda), FadeIn(contador))
        nota = Text("Una sola componente: una recta. No hay curvatura.",
                    font_size=26, color=GRIS).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(nota))
        self.pausa(2.0)

        # =============================================================
        # Escena 2 · añadir componentes ReLU: 2, 4, 8
        # =============================================================
        t2 = titulo("2 · Añadir componentes ReLU")
        self.play(FadeOut(t1), FadeIn(t2), FadeOut(nota))

        nota2 = Text("Cada componente nueva agrega un cambio de pendiente.",
                     font_size=26, color=GRIS).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(nota2))

        def agregar(K_nuevo, tiempo):
            nonlocal aprox
            K_prev = K_nuevo // 2
            nuevos = nodos(K_nuevo)[1:-1:2]  # puntos medios = quiebres nuevos

            # 1) destacar en naranja dónde aparecerán los quiebres
            puntos = VGroup(*[
                Dot(ejes.c2p(x, g(K_prev, np.array([x]))[0]), radius=0.09, color=NARANJA)
                for x in nuevos
            ])
            guias = VGroup(*[
                DashedLine(ejes.c2p(x, -2), ejes.c2p(x, 2), color=NARANJA,
                           stroke_width=2.5, dash_length=0.08).set_opacity(0.6)
                for x in nuevos
            ])
            nuevas_marcas = VGroup(*[marca(x, NARANJA) for x in nuevos])
            self.play(
                LaggedStart(*[Create(gl) for gl in guias], lag_ratio=0.08),
                LaggedStart(*[GrowFromCenter(p) for p in puntos], lag_ratio=0.08),
                FadeIn(nuevas_marcas),
                run_time=1.0,
            )
            self.pausa(0.8)

            # 2) la línea verde se dobla; los puntos viajan con ella
            destino = VGroup(*[
                Dot(ejes.c2p(x, f(x)), radius=0.09, color=NARANJA) for x in nuevos
            ])
            nueva = curva_g(K_nuevo)
            self.play(
                Transform(aprox, nueva),
                Transform(puntos, destino),
                k_val.animate.set_value(K_nuevo),
                run_time=tiempo,
                rate_func=smooth,
            )
            self.wait(0.4)

            # 3) retirar el resaltado; las marcas quedan en gris
            self.play(
                FadeOut(guias), FadeOut(puntos),
                nuevas_marcas.animate.set_color(GRIS),
                run_time=0.6,
            )
            marcas.add(*nuevas_marcas)

        for K_nuevo in (2, 4, 8):
            agregar(K_nuevo, tiempo=2.2)
            self.pausa(1.0)

        # =============================================================
        # Escena 3 · aproximación final (16)
        # =============================================================
        t3 = titulo("3 · Aproximación final")
        self.play(FadeOut(t2), FadeIn(t3), FadeOut(nota2))
        agregar(16, tiempo=2.5)
        self.wait(3.0)  # mantener la imagen

        formula = MathTex(
            r"f(x) \approx c + \sum_{i=1}^{16} a_i\,\operatorname{ReLU}(x - b_i)",
            color=TINTA, font_size=40,
        )
        caja = SurroundingRectangle(formula, color=GRIS, buff=0.2, corner_radius=0.08,
                                    stroke_width=1.5, fill_color=WHITE, fill_opacity=0.92)
        # zona vacía arriba a la izquierda (la curva está abajo para x < −0.6)
        bloque = VGroup(caja, formula)
        bloque.next_to(t3, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play(FadeIn(caja), Write(formula), run_time=2.0)
        self.pausa(2.0)

        cierre = Text("La suma de funciones simples puede aproximar una función más compleja.",
                      font_size=28, color=TINTA).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(cierre, shift=UP * 0.2))
        self.wait(3.0)
