"""
E10_GrafoForwardBackward · SI7011
Grafo computacional de una MLP con dos capas ocultas: forward y backward.

    x → [Lineal] → z₁ → [No lineal] → a₁ → [Lineal] → z₂ → [No lineal] → a₂ → [Lineal] → ŷ → [𝓛(ŷ, y)]

Forward: un pulso recorre el grafo, cada bloque se ilumina y quedan guardados
z₁, a₁, z₂, a₂ (los rótulos de las flechas).
Backward: un pulso naranja regresa desde 𝓛. En cada bloque aparece su regla
local (notación adjunta v̄ = ∂𝓛/∂v) y se iluminan los valores del forward que
la regla reutiliza.

Render:
    S01_VIDEO=1 manim -qh escena_grafo.py E10_GrafoForwardBackward
    manim -qh escena_grafo.py E10_GrafoForwardBackward     # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo (igual a la figura)
FONDO = WHITE
NAVY = "#14213d"
GRIS_TXT = "#5b6b80"
FLECHA = "#5b6577"
AZ_FILL, AZ_BORDE = "#e4eefc", "#7aa6dc"
VE_FILL, VE_BORDE = "#e3f5ea", "#83c9a0"
NARANJA = "#d9631e"
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


# ---------------------------------------------------------------- geometría
Y_BLOQUE = 1.45
H_BLOQUE = 0.86
GAP = 0.66


class Bloque(VGroup):
    def __init__(self, texto, ancho, fill, borde, **kw):
        super().__init__(**kw)
        self.caja = RoundedRectangle(width=ancho, height=H_BLOQUE, corner_radius=0.1,
                                     fill_color=fill, fill_opacity=1, stroke_color=borde, stroke_width=2)
        self.texto = texto
        self.borde = borde
        self.add(self.caja, texto)


class E10_GrafoForwardBackward(EscenaBase):
    def construct(self):
        # ----- bloques
        def lineal():
            return Bloque(T("Lineal", font_size=25, weight=BOLD), 1.5, AZ_FILL, AZ_BORDE)

        def nolineal():
            return Bloque(T("No lineal", font_size=25, weight=BOLD), 1.8, VE_FILL, VE_BORDE)

        L1, N1, L2, N2, L3 = lineal(), nolineal(), lineal(), nolineal(), lineal()
        perdida = Bloque(M(r"\mathcal{L}(\hat y, y)", font_size=32), 1.45, NA_FILL, NARANJA)
        rx = M("x", font_size=46)
        cadena = [L1, N1, L2, N2, L3, perdida]

        fila = VGroup(rx, *cadena).arrange(RIGHT, buff=GAP)
        rx.shift(RIGHT * 0.12)
        fila.move_to([0, Y_BLOQUE, 0])

        # flechas forward y rótulos de lo que se guarda
        nombres = ["z_1", "a_1", "z_2", "a_2", r"\hat y"]
        f_ent = Arrow(rx.get_right(), L1.get_left(), buff=0.08, color=FLECHA, stroke_width=2.5,
                      max_tip_length_to_length_ratio=0.25)
        flechas, rotulos = VGroup(), VGroup()
        for a, b, nm in zip(cadena[:-1], cadena[1:], nombres):
            fl = Arrow(a.get_right(), b.get_left(), buff=0.0, color=FLECHA, stroke_width=2.5,
                       max_tip_length_to_length_ratio=0.25)
            flechas.add(fl)
            rotulos.add(M(nm, font_size=38).next_to(fl, UP, buff=0.12))
        r_x = rx  # x también es un valor guardado

        # etiqueta y que entra a la pérdida desde abajo
        y_lbl = M("y", font_size=36).next_to(perdida, DOWN, buff=0.55)
        y_fl = Arrow(y_lbl.get_top(), perdida.get_bottom(), buff=0.06, color=FLECHA, stroke_width=2.5,
                     max_tip_length_to_length_ratio=0.3)

        # llaves y títulos de capa
        def llave(bloques, texto):
            g = VGroup(*bloques)
            br = Brace(g, UP, buff=0.18, color=FLECHA, sharpness=1.2)
            br.stretch_to_fit_height(0.32)
            br.next_to(g, UP, buff=0.18)
            return VGroup(br, T(texto, font_size=25, weight=BOLD).next_to(br, UP, buff=0.12))

        llaves = VGroup(llave([L1, N1], "Capa oculta 1"), llave([L2, N2], "Capa oculta 2"),
                        llave([L3], "Salida"), llave([perdida], "Pérdida"))

        # ecuaciones forward bajo cada bloque
        fwd_tex = [r"z_1 = W_1 x + b_1", r"a_1 = \sigma(z_1)", r"z_2 = W_2 a_1 + b_2",
                   r"a_2 = \sigma(z_2)", r"\hat y = W_3 a_2 + b_3"]
        fwd_sub = ["Transformación afín", "Activación", "Transformación afín",
                   "Activación", "Salida lineal"]
        ecs_f = VGroup()
        for b, tx, sb in zip(cadena, fwd_tex, fwd_sub):
            e = M(tx, font_size=31)
            s = T(sb, font_size=17, color=GRIS_TXT)
            ecs_f.add(VGroup(e, s).arrange(DOWN, buff=0.14).next_to(b, DOWN, buff=0.38))

        titulo = T("Forward: calcular y guardar", font_size=30, weight=BOLD).to_corner(UL, buff=0.35)

        # =============================================================
        # Estructura
        # =============================================================
        self.play(FadeIn(titulo))
        self.play(LaggedStart(FadeIn(rx), *[FadeIn(b, shift=UP * 0.1) for b in cadena], lag_ratio=0.12),
                  run_time=1.6)
        self.play(GrowArrow(f_ent), *[GrowArrow(f) for f in flechas], GrowArrow(y_fl), FadeIn(y_lbl),
                  FadeIn(llaves), run_time=1.2)
        self.pausa(1.0)

        # =============================================================
        # Forward
        # =============================================================
        pulso = Dot(rx.get_right() + RIGHT * 0.05, radius=0.09, color=AZ_BORDE)
        self.play(FadeIn(pulso, scale=0.5), run_time=0.3)
        self.play(pulso.animate.move_to(L1.get_left()), run_time=0.5, rate_func=linear)

        for k, b in enumerate(cadena):
            tinta = b.borde
            self.play(
                pulso.animate.move_to(b.get_center()).set_color(tinta),
                b.caja.animate.set_stroke(width=4.5),
                run_time=0.45, rate_func=linear,
            )
            extra = [FadeIn(ecs_f[k], shift=UP * 0.1)] if k < len(ecs_f) else []
            self.play(*extra, b.caja.animate.set_stroke(width=2), run_time=0.6)
            if k < len(flechas):
                self.play(pulso.animate.move_to(flechas[k].get_end()), FadeIn(rotulos[k], shift=DOWN * 0.08),
                          run_time=0.6, rate_func=linear)
        self.play(FadeOut(pulso), Indicate(perdida, color=NARANJA, scale_factor=1.06))

        nota = T("Se guardan  x, z₁, a₁, z₂, a₂, ŷ  para el backward.", font_size=22, color=GRIS_TXT)
        nota.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(nota),
                  LaggedStart(*[Indicate(r, color=AZ_BORDE, scale_factor=1.15) for r in [r_x, *rotulos]],
                              lag_ratio=0.15), run_time=1.6)
        self.pausa(2.0)

        # =============================================================
        # Backward
        # =============================================================
        titulo_b = T("Backward: regla de la cadena, de derecha a izquierda", font_size=30, weight=BOLD,
                     color=NARANJA).to_corner(UL, buff=0.35)
        leyenda = M(r"\bar v \equiv \partial \mathcal{L} / \partial v",
                    font_size=32, color=GRIS_TXT).to_corner(UR, buff=0.4)

        # atenuar el forward un poco: sigue visible porque se reutiliza
        self.play(
            FadeOut(titulo), FadeIn(titulo_b), FadeOut(nota),
            ecs_f.animate.set_opacity(0.45), FadeIn(leyenda),
            run_time=0.9,
        )

        # flechas backward (más abajo que las forward, apuntando a la izquierda)
        dy = DOWN * 0.24
        b_fl = VGroup()
        b_rot = VGroup()
        nombres_b = [r"\bar z_1", r"\bar a_1", r"\bar z_2", r"\bar a_2", r"\bar{\hat y}"]
        for a, b, nm in zip(cadena[:-1], cadena[1:], nombres_b):
            fl = Arrow(b.get_left() + dy, a.get_right() + dy, buff=0.0, color=NARANJA, stroke_width=3,
                       max_tip_length_to_length_ratio=0.25)
            b_fl.add(fl)
            b_rot.add(M(nm, font_size=34, color=NARANJA).next_to(fl, DOWN, buff=0.1))

        # reglas locales (fila A: gradiente hacia la entrada; fila B: parámetros)
        Y_A, Y_B = -0.95, -2.05
        reglas_A = {
            5: r"\bar{\hat y} = \frac{\partial \ell}{\partial \hat y}",
            4: r"\bar a_2 = W_3^{\top} \bar{\hat y}",
            3: r"\bar z_2 = \bar a_2 \odot \sigma'(z_2)",
            2: r"\bar a_1 = W_2^{\top} \bar z_2",
            1: r"\bar z_1 = \bar a_1 \odot \sigma'(z_1)",
        }
        reglas_B = {
            4: (r"\bar W_3 = \bar{\hat y}\, a_2^{\top}", r"\bar b_3 = \bar{\hat y}"),
            2: (r"\bar W_2 = \bar z_2\, a_1^{\top}", r"\bar b_2 = \bar z_2"),
            0: (r"\bar W_1 = \bar z_1\, x^{\top}", r"\bar b_1 = \bar z_1"),
        }
        # qué valor del forward reutiliza cada bloque
        usa = {4: [rotulos[3]], 3: [rotulos[1 + 1]], 2: [rotulos[1]], 1: [rotulos[0]], 0: [r_x]}

        def regla(tex, x, y, size=29):
            return M(tex, font_size=size, color=NARANJA).move_to([x, y, 0])

        cajas_param = VGroup()
        pulso = Dot(perdida.get_center() + dy, radius=0.09, color=NARANJA)
        self.play(FadeIn(pulso, scale=0.5), perdida.caja.animate.set_stroke(width=4.5), run_time=0.4)
        semilla = regla(reglas_A[5], perdida.get_x(), Y_A)
        self.play(Write(semilla), perdida.caja.animate.set_stroke(width=2), run_time=0.9)
        self.pausa(0.8)

        for k in range(4, -1, -1):
            b = cadena[k]
            fl, rt = b_fl[k], b_rot[k]
            # el gradiente viaja por la flecha hacia el bloque k
            self.play(GrowArrow(fl), pulso.animate.move_to(fl.get_end()), run_time=0.6, rate_func=linear)
            self.play(FadeIn(rt, shift=UP * 0.08), pulso.animate.move_to(b.get_center() + dy),
                      b.caja.animate.set_stroke(color=NARANJA, width=4.5), run_time=0.5)
            anims = [b.caja.animate.set_stroke(color=b.borde, width=2)]
            if k in reglas_A:
                anims.append(Write(regla(reglas_A[k], b.get_x(), Y_A)))
            if k in reglas_B:
                wtex, btex = reglas_B[k]
                par = VGroup(regla(wtex, 0, 0, 28), regla(btex, 0, 0, 28)).arrange(DOWN, buff=0.18)
                par.move_to([b.get_x(), Y_B, 0])
                caja = SurroundingRectangle(par, color=NARANJA, buff=0.12, corner_radius=0.08,
                                            stroke_width=1.5, fill_color=NA_FILL, fill_opacity=1)
                cajas_param.add(caja)
                anims += [FadeIn(caja), Write(par)]
                self.add_foreground_mobject(par)
            self.play(*anims,
                      *[Indicate(u, color=NARANJA, scale_factor=1.25) for u in usa.get(k, [])],
                      run_time=1.2)
            self.pausa(0.7)

        self.play(FadeOut(pulso))
        cierre = T("El forward guarda los valores; el backward los reutiliza "
                   "para obtener todos los gradientes.", font_size=22, color=NAVY)
        cierre.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(cierre, shift=UP * 0.1),
                  LaggedStart(*[Indicate(c, color=NARANJA, scale_factor=1.05) for c in cajas_param[::-1]],
                              lag_ratio=0.3), run_time=1.8)
        self.wait(3.0)
