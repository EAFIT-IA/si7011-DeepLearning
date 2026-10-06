"""
E9_EjemploActivaciones · SI7011 (Lecture 03, diapositivas 8–10)
Tres grupos rojo–azul–rojo sobre la diagonal. Dos unidades ocultas con
    z₁ =  (x₁ + x₂)/√2 − 1.5,   z₂ = −(x₁ + x₂)/√2 − 1.5.
La parte afín deja los puntos sobre una recta, todavía en orden rojo–azul–rojo,
así que no se pueden separar con una recta. tanh y ReLU doblan esa recta y las
clases quedan separables.

Cuatro paneles, como la figura estática. Las copias de los puntos viajan
de un panel al siguiente:
x → z → tanh(z) y z → ReLU(z).

Render:
    S01_VIDEO=1 manim -qh escena_activaciones.py E9_EjemploActivaciones
    manim -qh escena_activaciones.py E9_EjemploActivaciones    # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo (igual a la figura)
FONDO = WHITE
NAVY = "#1d3a5f"       # títulos
PIZARRA = "#6b7a8f"    # ejes, marcas, etiquetas
GUIA = "#b9c1cc"       # líneas punteadas en 0
ROJO = "#c9565c"
AZUL = "#3b86b4"
VERDE = "#127a6e"      # vectores normales y separadores
FUENTE = "DejaVu Sans"

SEPARADORES = True     # al final dibuja una recta que separa en tanh y ReLU

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


# ---------------------------------------------------------------- datos y capa
def datos_tres_grupos(n=40, semilla=4):
    r = np.random.default_rng(semilla)
    rojo = np.r_[r.normal([3.0, 3.0], 0.6, (n, 2)), r.normal([-3.0, -3.0], 0.6, (n, 2))]
    azul = r.normal([0.0, 0.0], 0.55, (n, 2))
    return rojo, azul


S2 = np.sqrt(2.0)


def capa_afin(X):
    s = (X[:, 0] + X[:, 1]) / S2
    return np.c_[s - 1.5, -s - 1.5]


# ---------------------------------------------------------------- panel tipo matplotlib
class Panel(VGroup):
    """Ejes en L (izquierda y abajo), marcas hacia afuera, guías punteadas en 0."""

    def __init__(self, xlim, ylim, xticks, yticks, lado=3.0, xlab="x_1", ylab="x_2", **kw):
        super().__init__(**kw)
        self.xlim, self.ylim, self.lado = xlim, ylim, lado
        o = np.array([-lado / 2, -lado / 2, 0])
        self._o = o
        abajo = Line(o, o + RIGHT * lado, color=PIZARRA, stroke_width=1.6)
        izq = Line(o, o + UP * lado, color=PIZARRA, stroke_width=1.6)
        self.guias = VGroup()
        if xlim[0] < 0 < xlim[1]:
            self.guias.add(DashedLine(self.c2p(0, ylim[0]), self.c2p(0, ylim[1]),
                                      color=GUIA, stroke_width=1.3, dash_length=0.05))
        if ylim[0] < 0 < ylim[1]:
            self.guias.add(DashedLine(self.c2p(xlim[0], 0), self.c2p(xlim[1], 0),
                                      color=GUIA, stroke_width=1.3, dash_length=0.05))
        marcas = VGroup()
        for v in xticks:
            p = self.c2p(v, ylim[0])
            marcas.add(Line(p, p + DOWN * 0.07, color=PIZARRA, stroke_width=1.4))
            marcas.add(self._num(v).next_to(p, DOWN, buff=0.12))
        for v in yticks:
            p = self.c2p(xlim[0], v)
            marcas.add(Line(p, p + LEFT * 0.07, color=PIZARRA, stroke_width=1.4))
            marcas.add(self._num(v).next_to(p, LEFT, buff=0.12))
        self.rx = MathTex(xlab, color=NAVY, font_size=34).next_to(abajo, DOWN, buff=0.48)
        self.ry = MathTex(ylab, color=NAVY, font_size=34).rotate(PI / 2).next_to(izq, LEFT, buff=0.42)
        self.add(self.guias, abajo, izq, marcas, self.rx, self.ry)

    @staticmethod
    def _num(v):
        t = f"{v:g}".replace("-", "−")
        return Text(t, font=FUENTE, font_size=15, color=PIZARRA)

    def c2p(self, x, y):
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        fy = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.get_origen() + np.array([fx * self.lado, fy * self.lado, 0])

    def get_origen(self):
        # el panel puede haberse movido: el origen sigue a la línea de abajo
        return self._o if not hasattr(self, "_mov") else self._mov

    def colocar(self, centro):
        self.shift(centro)
        self._mov = self._o + centro
        return self


def nube(panel, P, color):
    return VGroup(*[
        Dot(panel.c2p(*p), radius=0.042, fill_color=color, fill_opacity=0.95,
            stroke_color=WHITE, stroke_width=0.8)
        for p in P
    ])


def viajar(origen, panel_dest, P):
    """Copias de una nube que se mueven a las coordenadas P en otro panel."""
    copia = origen.copy()
    destino = [panel_dest.c2p(*p) for p in P]
    anims = [d.animate.move_to(q) for d, q in zip(copia, destino)]
    return copia, anims


class E9_EjemploActivaciones(EscenaBase):
    def construct(self):
        rojo_x, azul_x = datos_tres_grupos()
        rojo_z, azul_z = capa_afin(rojo_x), capa_afin(azul_x)
        relu = lambda Z: np.maximum(0, Z)

        # ----- cuatro paneles
        LADO, Y0 = 2.6, 0.25
        xs = [-4.9, -1.4, 2.1, 5.6]
        p1 = Panel([-5, 5], [-5, 5], [-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], LADO).colocar([xs[0], Y0, 0])
        p2 = Panel([-7.5, 4.5], [-7.5, 4.5], [-6, -4, -2, 0, 2, 4], [-6, -4, -2, 0, 2, 4], LADO,
                   xlab="z_1", ylab="z_2").colocar([xs[1], Y0, 0])
        p3 = Panel([-1.18, 1.18], [-1.18, 1.18], [-1, 0, 1], [-1, 0, 1], LADO,
                   xlab="a_1", ylab="a_2").colocar([xs[2], Y0, 0])
        p4 = Panel([-0.25, 4.5], [-0.25, 4.5], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], LADO,
                   xlab="a_1", ylab="a_2").colocar([xs[3], Y0, 0])

        def titulo(m, panel):
            return m.next_to(panel.c2p(np.mean(panel.xlim), panel.ylim[1]), UP, buff=0.32)

        t1 = titulo(Text("Input space", font=FUENTE, weight=BOLD, color=NAVY, font_size=21), p1)
        t2 = titulo(MathTex(r"z = Wx + b", color=NAVY, font_size=38), p2)
        t3 = titulo(MathTex(r"a = \tanh(z)", color=NAVY, font_size=38), p3)
        t4 = titulo(MathTex(r"a = \mathrm{ReLU}(z)", color=NAVY, font_size=38), p4)
        for t in (t2, t3, t4):
            t.align_to(t1, DOWN)

        formula = MathTex(
            r"z_1 = (x_1 + x_2)/\sqrt{2} - 1.5,", r"\qquad",
            r"z_2 = -(x_1 + x_2)/\sqrt{2} - 1.5",
            color=NAVY, font_size=34,
        ).to_edge(DOWN, buff=0.8)

        # =============================================================
        # 1 · Espacio de entrada
        # =============================================================
        r1, a1 = nube(p1, rojo_x, ROJO), nube(p1, azul_x, AZUL)
        self.play(FadeIn(t1), Create(p1), run_time=1.2)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in [*r1, *a1]], lag_ratio=0.01), run_time=1.4)
        self.pausa(1.0)

        # rectas z₁ = 0, z₂ = 0 y sus vectores normales
        rectas, etiquetas, normales = VGroup(), VGroup(), VGroup()
        for signo, nombre in ((1, "z_1 = 0"), (-1, "z_2 = 0")):
            c = signo * 1.5 * S2  # x₁ + x₂ = c
            a, b = max(-5, c - 5), min(5, c + 5)
            rectas.add(Line(p1.c2p(a, c - a), p1.c2p(b, c - b), color=PIZARRA, stroke_width=2.2))
            et = MathTex(nombre, color=PIZARRA, font_size=24).rotate(-PI / 4)
            xe = 0.15 if signo == 1 else -2.75
            et.move_to(p1.c2p(xe, c - xe)).shift(UR * 0.3)
            etiquetas.add(et)
            pie = np.array([c / 2, c / 2])
            punta = pie + signo * np.array([1.0, 1.0])
            normales.add(Arrow(p1.c2p(*pie), p1.c2p(*punta), buff=0, color=VERDE, stroke_width=3.5,
                               max_tip_length_to_length_ratio=0.28))
        self.play(Create(rectas), FadeIn(etiquetas), run_time=1.4)
        self.play(GrowArrow(normales[0]), GrowArrow(normales[1]))
        self.play(Write(formula), run_time=1.6)
        self.pausa(1.5)

        # =============================================================
        # 2 · Parte afín: x → z
        # =============================================================
        self.play(FadeIn(t2), Create(p2), run_time=1.0)
        r2, ar = viajar(r1, p2, rojo_z)
        a2, aa = viajar(a1, p2, azul_z)
        self.add(r2, a2)
        self.play(*ar, *aa, run_time=2.6, rate_func=smooth)
        self.pausa(1.5)

        # =============================================================
        # 3 · tanh: z → a
        # =============================================================
        self.play(FadeIn(t3), Create(p3), run_time=1.0)
        r3, ar = viajar(r2, p3, np.tanh(rojo_z))
        a3, aa = viajar(a2, p3, np.tanh(azul_z))
        self.add(r3, a3)
        self.play(*ar, *aa, run_time=2.6, rate_func=smooth)
        self.pausa(1.5)

        # =============================================================
        # 4 · ReLU: z → a
        # =============================================================
        self.play(FadeIn(t4), Create(p4), run_time=1.0)
        r4, ar = viajar(r2, p4, relu(rojo_z))
        a4, aa = viajar(a2, p4, relu(azul_z))
        self.add(r4, a4)
        self.play(*ar, *aa, run_time=2.6, rate_func=smooth)
        self.pausa(1.5)

        # =============================================================
        # 5 · (opcional) una recta ya basta
        # =============================================================
        if SEPARADORES:
            # tanh: a₁ + a₂ = −0.55   ·   ReLU: a₁ + a₂ = 0.65
            def separador(panel, c, lo, hi):
                return DashedLine(panel.c2p(lo, c - lo), panel.c2p(hi, c - hi),
                                  color=VERDE, stroke_width=3, dash_length=0.08)
            s3 = separador(p3, -0.55, -1.18, 0.63)
            s4 = separador(p4, 0.65, -0.25, 0.9)
            self.play(Create(s3), Create(s4), run_time=1.2)
        self.wait(3.0)
