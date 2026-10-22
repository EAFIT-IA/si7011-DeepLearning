"""
Escenas Manim · SI7011 · Sesión 5: Sequences

    A01_UnrollingBPTT        La celda recurrente enrollada, luego desenrollada en 8 pasos con la
                             misma W_hh en cada paso; después el gradiente vuelve desde el paso 8 y
                             se debilita paso a paso. Magnitudes: RNN entrenada del notebook 5.2
                             (../notebooks/gradiente_tiempo_rnn.npz), las últimas 8 horas.
    A02_LSTMMemory           Un valor entra en el paso 2. En una RNN se desvanece paso a paso; en la
                             LSTM la compuerta de entrada lo escribe en c_t, la de olvido (≈ 1) lo
                             conserva y la de salida lo lee en el paso 7. Números ilustrativos.
    A03_Seq2seqAttention     Decodificar "el diecinueve de octubre de 2 0 2 6 en la sala dos" → 2026-10-19:
                             en cada carácter, los pesos α_τt sobre los estados del encoder y el
                             carácter emitido; la matriz de S05-F09 se llena fila por fila.
                             Pesos: notebook 5.5 (../notebooks/atencion_fechas.npz, ejemplo 2).

Notación del curso: x_t entrada, h_t estado oculto, c_t celda de la LSTM, f_t, i_t, o_t compuertas,
W_hh pesos recurrentes, s_τ estado del decoder, α_τt pesos de atención, a_τ contexto, L pérdida.

Render:
    S01_VIDEO=1 manim -qh escenas_s05.py A01_UnrollingBPTT   # video lineal
    manim -qh escenas_s05.py A01_UnrollingBPTT               # con pausas (manim-slides)
Poster (último cuadro): añadir -s.
"""
import os
from pathlib import Path

import numpy as np
from manim import *

AQUI = Path(__file__).resolve().parent
NOTEBOOKS = AQUI.parent / "notebooks"

# ---------------------------------------------------------------- estilo (como S02–S04)
FONDO = WHITE
NAVY = "#14213d"
GRIS_TXT = "#5b6b80"
GRIS_EJE = "#5b6577"
GRIS_CLARO = "#c4cbd5"
AZUL = "#1f6fb4"
TEAL = "#167d78"
NARANJA = "#d9631e"
ROJO = "#c0392b"
AZ_FILL, AZ_BORDE = "#e4eefc", "#7aa6dc"
VE_FILL = "#e3f5ea"
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
    kw.setdefault("font", FUENTE)
    return Text(s, **kw)


def M(s, **kw):
    kw.setdefault("color", NAVY)
    return MathTex(s, **kw)


def titulo(s):
    return T(s, font_size=30, weight=BOLD).to_corner(UL, buff=0.35)


def nota(s, **kw):
    kw.setdefault("font_size", 22)
    kw.setdefault("color", NAVY)
    return T(s, **kw).to_edge(DOWN, buff=0.35)


def caja(ancho, alto, fill=AZ_FILL, borde=AZUL, r=0.14):
    return RoundedRectangle(corner_radius=r, width=ancho, height=alto, stroke_color=borde,
                            stroke_width=2, fill_color=fill, fill_opacity=1)


def flecha(a, b, color=GRIS_EJE, ancho=3, punta=0.18):
    return Arrow(a, b, buff=0.05, color=color, stroke_width=ancho, tip_length=punta,
                 max_tip_length_to_length_ratio=0.35, max_stroke_width_to_length_ratio=20)


# ======================================================================
# A01 · Unrolling through time
# ======================================================================
class A01_UnrollingBPTT(EscenaBase):
    def construct(self):
        n = 8
        datos = np.load(NOTEBOOKS / "gradiente_tiempo_rnn.npz")
        g = datos["g"]                                    # ‖∂L/∂x_t‖, t = 1..120, RNN entrenada
        rel = g[-n:] / g[-1]                              # las últimas 8 horas, relativas a la última

        self.add(titulo("Unrolling through time"))

        # --- la celda enrollada
        celda = caja(1.6, 1.0)
        et = M(r"\mathbf h_t", font_size=36).move_to(celda)
        x_in = M(r"\mathbf x_t", font_size=34).next_to(celda, DOWN, buff=0.9)
        h_out = M(r"\mathbf h_t", font_size=34).next_to(celda, UP, buff=0.9)
        f_in = flecha(x_in.get_top(), celda.get_bottom())
        f_out = flecha(celda.get_top(), h_out.get_bottom())
        lazo = CurvedArrow(celda.get_right() + 0.15 * UP, celda.get_right() + 0.15 * DOWN, angle=-4.2,
                           color=NARANJA, stroke_width=4, tip_length=0.18)
        lazo.shift(0.05 * RIGHT)
        w_lazo = M(r"W_{hh}", font_size=32, color=NARANJA).next_to(lazo, RIGHT, buff=0.12)
        enrollada = VGroup(celda, et, x_in, h_out, f_in, f_out, lazo, w_lazo).move_to(0.2 * DOWN)
        ecu = M(r"\mathbf h_t = \tanh\!\big(W_{xh}\,\mathbf x_t + W_{hh}\,\mathbf h_{t-1} + \mathbf b\big)",
                font_size=32).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(enrollada), Write(ecu))
        self.pausa()

        # --- desenrollada en 8 pasos
        xs = np.linspace(-5.6, 5.0, n)
        y0 = 0.2
        cajas = VGroup(*[caja(0.95, 0.75).move_to([x, y0, 0]) for x in xs])
        etq = VGroup(*[M(rf"\mathbf h_{k + 1}", font_size=26).move_to(c) for k, c in enumerate(cajas)])
        entradas = VGroup(*[M(rf"\mathbf x_{k + 1}", font_size=26).move_to([x, y0 - 1.55, 0])
                            for k, x in enumerate(xs)])
        f_x = VGroup(*[flecha(e.get_top(), c.get_bottom(), ancho=2.5) for e, c in zip(entradas, cajas)])
        f_h = VGroup(*[flecha(cajas[k].get_right(), cajas[k + 1].get_left(), color=NARANJA, ancho=4)
                       for k in range(n - 1)])
        w_h = VGroup(*[M(r"W_{hh}", font_size=20, color=NARANJA).next_to(f, UP, buff=0.08) for f in f_h])
        salida = VGroup(M(r"\hat y", font_size=30), M(r"\to L", font_size=30)).arrange(RIGHT, buff=0.1)
        salida.next_to(cajas[-1], UP, buff=0.75)
        f_y = flecha(cajas[-1].get_top(), salida.get_bottom())
        desenrollada = VGroup(cajas, etq, entradas, f_x, f_h, w_h, salida, f_y)

        self.play(FadeOut(enrollada, shift=0.3 * UP), FadeOut(ecu))
        self.play(LaggedStart(*[FadeIn(VGroup(c, e, x, fx)) for c, e, x, fx in zip(cajas, etq, entradas, f_x)],
                              lag_ratio=0.12),
                  LaggedStart(*[GrowArrow(f) for f in f_h], lag_ratio=0.12), run_time=2.2)
        self.play(FadeIn(w_h), FadeIn(salida), GrowArrow(f_y))
        misma = nota("The same weights at every step: weight sharing along time.", color=NARANJA)
        self.play(*[Indicate(w, color=NARANJA, scale_factor=1.35) for w in w_h], FadeIn(misma))
        self.pausa()

        # --- backpropagation through time
        self.play(FadeOut(misma))
        ecu2 = M(r"\frac{\partial \mathbf h_T}{\partial \mathbf h_t} = \prod_{k=t+1}^{T}"
                 r"\operatorname{diag}\!\big(\varphi'(\mathbf z_k)\big)\, W_{hh}", font_size=30)
        ecu2.to_edge(DOWN, buff=0.35)
        self.play(Write(ecu2))
        atras = VGroup()
        valores = VGroup()
        y_g = y0 - 2.55
        for k in range(n - 1, -1, -1):
            r = float(rel[k])
            ancho = max(0.6, 9 * r ** 0.35)
            op = max(0.25, r ** 0.25)
            a = Arrow([xs[k] + 0.55, y_g, 0], [xs[k] - 0.55, y_g, 0], buff=0, color=ROJO, stroke_width=ancho,
                      stroke_opacity=op, tip_length=0.16, max_tip_length_to_length_ratio=0.3)
            a.tip.set_fill(opacity=op)
            if k == n - 1:
                txt = M(r"1", font_size=24, color=ROJO)
            else:
                mant, exp_ = f"{r:.0e}".split("e")
                txt = M(rf"{mant}\!\cdot\!10^{{{int(exp_)}}}", font_size=22, color=ROJO)
            txt.set_opacity(max(0.5, op)).next_to(a, DOWN, buff=0.1)
            atras.add(a); valores.add(txt)
        self.play(FadeOut(entradas), FadeOut(f_x), run_time=0.6)
        rotulo = T("gradient reaching step t, relative to the last step", font_size=19, color=ROJO)
        rotulo.next_to(atras, UP, buff=0.12).align_to(atras, LEFT)
        self.play(FadeIn(rotulo))
        for a, v in zip(atras, valores):
            self.play(GrowArrow(a), FadeIn(v), run_time=0.45)
        self.pausa()
        cierre = VGroup(T("The same factor, repeated:", font_size=22, color=NAVY),
                        T("the signal from the first steps fades geometrically.", font_size=22, color=ROJO)
                        ).arrange(RIGHT, buff=0.18)
        fuente = T("Magnitudes: the trained RNN of notebook 5.2, last 8 hours of the window.",
                   font_size=16, color=GRIS_TXT)
        self.play(FadeOut(ecu2))
        cierre.to_edge(DOWN, buff=0.65)
        fuente.next_to(cierre, DOWN, buff=0.15)
        self.play(FadeIn(cierre), FadeIn(fuente))
        self.pausa(2.0)


# ======================================================================
# A02 · The memory in motion
# ======================================================================
class A02_LSTMMemory(EscenaBase):
    def construct(self):
        n, t_in, t_out = 8, 2, 7
        xs = np.linspace(-4.6, 5.6, n)
        self.add(titulo("The memory in motion"))

        def fila(y, nombre):
            et = T(nombre, font_size=24, weight=BOLD).move_to([-6.25, y, 0])
            cajas = VGroup(*[caja(0.9, 0.62).move_to([x, y, 0]) for x in xs])
            fl = VGroup(*[flecha(cajas[k].get_right(), cajas[k + 1].get_left(), ancho=2.5) for k in range(n - 1)])
            pasos = VGroup(*[T(str(k + 1), font_size=17, color=GRIS_TXT).next_to(c, DOWN, buff=0.1)
                             for k, c in enumerate(cajas)])
            return et, cajas, fl, pasos

        # --- fila 1: RNN
        y_r = 1.75
        et_r, cajas_r, fl_r, pasos_r = fila(y_r, "RNN")
        self.play(FadeIn(et_r), FadeIn(cajas_r), FadeIn(fl_r), FadeIn(pasos_r))
        punto = Dot(cajas_r[t_in - 1].get_bottom() + 0.75 * DOWN, radius=0.13, color=NARANJA)
        entra = T("an important value enters at step 2", font_size=18, color=NARANJA).next_to(punto, RIGHT, buff=0.15)
        self.play(FadeIn(punto, scale=2), FadeIn(entra))
        self.play(punto.animate.move_to(cajas_r[t_in - 1]), run_time=0.6)
        self.remove(punto)
        resto = 1.0
        barras_r = VGroup()
        for k in range(t_in - 1, n):
            b = Rectangle(width=0.5, height=0.5 * resto, stroke_width=0, fill_color=NARANJA, fill_opacity=0.85)
            b.move_to(cajas_r[k]).align_to(cajas_r[k], DOWN).shift(0.06 * UP)
            barras_r.add(b)
            self.play(FadeIn(b), run_time=0.35)
            resto *= 0.45
        apaga = T("fades at every step: × W_hh, × φ′", font_size=19, color=ROJO).next_to(cajas_r, UP, buff=0.18)
        apaga.align_to(cajas_r, RIGHT)
        self.play(FadeOut(entra), FadeIn(apaga))
        self.pausa()

        # --- fila 2: LSTM
        y_l = -1.35
        et_l, cajas_l, fl_l, pasos_l = fila(y_l, "LSTM")
        linea = Line([xs[0] - 0.6, y_l + 0.95, 0], [xs[-1] + 0.6, y_l + 0.95, 0], color=NARANJA, stroke_width=6)
        c_lab = M(r"\mathbf c_t", font_size=28, color=NARANJA).next_to(linea, LEFT, buff=0.12)
        self.play(FadeIn(et_l), FadeIn(cajas_l), FadeIn(fl_l), FadeIn(pasos_l), Create(linea), FadeIn(c_lab))

        def compuerta(x, y, nombre, abierta, color):
            base = RoundedRectangle(corner_radius=0.08, width=0.5, height=0.2, stroke_color=color, stroke_width=1.6,
                                    fill_color=WHITE, fill_opacity=1).move_to([x, y, 0])
            nivel = 0.05 + 0.9 * abierta
            cursor = Dot(radius=0.07, color=color).move_to(base.get_left() + RIGHT * (0.5 * nivel))
            et = M(nombre, font_size=18, color=color).next_to(base, UP, buff=0.04)
            return VGroup(base, cursor, et)

        # escribir en el paso 2
        punto = Dot(cajas_l[t_in - 1].get_bottom() + 0.75 * DOWN, radius=0.13, color=NARANJA)
        self.play(FadeIn(punto, scale=2))
        g_i = compuerta(xs[t_in - 1] - 0.3, y_l + 0.5, r"\mathbf i_2", 1.0, TEAL)
        self.play(FadeIn(g_i), punto.animate.move_to([xs[t_in - 1], y_l + 0.95, 0]), run_time=0.8)
        escribe = T("input gate open: write", font_size=18, color=TEAL).next_to(cajas_l[t_in - 1], DOWN, buff=0.45)
        self.play(FadeIn(escribe))
        self.remove(punto)

        # conservar: f ≈ 1 en cada paso
        valor = 1.0
        barras_c = VGroup()
        olvidos = VGroup()
        for k in range(t_in - 1, n):
            b = Rectangle(width=0.5, height=0.5 * valor, stroke_width=0, fill_color=NARANJA, fill_opacity=0.85)
            b.move_to([xs[k], y_l + 0.95 + 0.25 * valor + 0.06, 0])
            barras_c.add(b)
            anims = [FadeIn(b)]
            if k > t_in - 1:
                g_f = compuerta(xs[k] - 0.62, y_l + 1.55, r"\mathbf f", 0.97, TEAL)
                olvidos.add(g_f); anims.append(FadeIn(g_f))
            self.play(*anims, run_time=0.35)
            valor *= 0.98
        conserva = T("forget gate ≈ 1: keep", font_size=18, color=TEAL).move_to([xs[3], y_l - 0.85, 0])
        self.play(FadeOut(escribe), FadeIn(conserva))
        self.pausa()

        # leer en el paso 7
        g_o = compuerta(xs[t_out - 1] + 0.3, y_l + 0.5, r"\mathbf o_7", 1.0, TEAL)
        lee = Dot([xs[t_out - 1], y_l + 0.95, 0], radius=0.11, color=NARANJA)
        self.play(FadeIn(g_o), FadeIn(lee))
        self.play(lee.animate.move_to(cajas_l[t_out - 1]), run_time=0.7)
        self.play(cajas_l[t_out - 1].animate.set_fill(NA_FILL), Flash(cajas_l[t_out - 1], color=NARANJA))
        leyo = T("output gate open: read it at step 7", font_size=18, color=TEAL).next_to(cajas_l[t_out - 1], DOWN, buff=0.45)
        leyo.shift(0.8 * LEFT)
        self.play(FadeIn(leyo))
        self.pausa()

        cierre = M(r"\mathbf c_t = \mathbf f_t \odot \mathbf c_{t-1} + \mathbf i_t \odot \tilde{\mathbf c}_t"
                   r"\qquad \mathbf h_t = \mathbf o_t \odot \tanh(\mathbf c_t)", font_size=30)
        cierre.to_edge(DOWN, buff=0.35)
        aviso = T("illustrative values", font_size=15, color=GRIS_TXT).to_corner(DR, buff=0.2)
        self.play(Write(cierre), FadeIn(aviso))
        self.pausa(2.0)


# ======================================================================
# A03 · Attention in motion
# ======================================================================
class A03_Seq2seqAttention(EscenaBase):
    def construct(self):
        z = np.load(NOTEBOOKS / "atencion_fechas.npz", allow_pickle=False)
        k_ej = 2
        palabras = str(z["ejemplos"][k_ej]).split()
        salida = str(z["salidas"][k_ej])
        A = z[f"alpha_{k_ej}"]                                   # (10, T)
        Tn, Tau = len(palabras), len(salida)

        self.add(titulo("Attention in motion"))
        x0, dx = -5.15, 0.5
        y_top, dy = 2.35, 0.37
        cw, chh = 0.45, 0.33
        cols = [x0 + j * dx for j in range(Tn)]
        filas = [y_top - i * dy for i in range(Tau)]

        # estados del encoder y palabras
        y_h = filas[-1] - 0.62
        estados = VGroup(*[Square(0.34, stroke_color=AZUL, stroke_width=1.8, fill_color=AZ_FILL, fill_opacity=1)
                           .move_to([x, y_h, 0]) for x in cols])
        h_lab = M(r"\mathbf h_t", font_size=26, color=AZUL).next_to(estados, LEFT, buff=0.25)
        textos = VGroup(*[T(w, font_size=15).rotate(55 * DEGREES) for w in palabras])
        for t, x in zip(textos, cols):
            t.move_to([x, y_h - 0.35, 0], aligned_edge=UR).shift(0.12 * RIGHT)
        enc = T("encoder: one state per input word", font_size=17, color=AZUL).next_to(h_lab, DOWN, buff=0.15).shift(0.4 * RIGHT)
        self.play(FadeIn(textos, lag_ratio=0.05), FadeIn(estados, lag_ratio=0.05), FadeIn(h_lab))

        # matriz vacía
        celdas = VGroup(*[VGroup(*[Rectangle(width=cw, height=chh, stroke_color=GRIS_CLARO, stroke_width=0.8,
                                             fill_color=NARANJA, fill_opacity=0).move_to([x, y, 0])
                                   for x in cols]) for y in filas])
        chars = VGroup(*[T(ch, font_size=22, font="DejaVu Sans Mono").move_to([x0 - 0.6, y, 0])
                         for ch, y in zip(salida, filas)])
        for c in chars:
            c.set_opacity(0)
        tau_lab = M(r"\tau", font_size=26, color=GRIS_TXT).move_to([x0 - 0.6, y_top + 0.45, 0])
        self.play(FadeIn(celdas), FadeIn(tau_lab))

        panel = VGroup(
            M(r"\alpha_{\tau t} = \operatorname{softmax}_t\big(\operatorname{score}(\mathbf s_{\tau-1}, \mathbf h_t)\big)",
              font_size=26),
            M(r"\mathbf a_\tau = \textstyle\sum_t \alpha_{\tau t}\, \mathbf h_t", font_size=26),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        panel.move_to([cols[-1] + 0.75, 1.75, 0], aligned_edge=LEFT)
        if panel.get_right()[0] > 6.9:
            panel.scale_to_fit_width(6.9 - panel.get_left()[0])
        self.play(Write(panel))
        self.pausa()

        marco = SurroundingRectangle(celdas[0], color=NARANJA, buff=0.04, stroke_width=2.5)
        paso_txt = T("", font_size=20)
        for i in range(Tau):
            lento = i < 3
            nuevo_marco = SurroundingRectangle(celdas[i], color=NARANJA, buff=0.04, stroke_width=2.5)
            if i == 0:
                self.play(Create(nuevo_marco), run_time=0.4)
            else:
                self.play(Transform(marco, nuevo_marco), run_time=0.3)
                nuevo_marco = marco
            marco = nuevo_marco
            pesos = A[i]
            anims = [cel.animate.set_fill(NARANJA, opacity=float(min(1.0, 0.08 + 0.92 * p ** 0.8)) if p > 0.02 else 0.0)
                     for cel, p in zip(celdas[i], pesos)]
            brillo = [estados[j].animate.set_fill(NA_FILL if pesos[j] > 0.15 else AZ_FILL) for j in range(Tn)]
            self.play(*anims, *brillo, run_time=0.9 if lento else 0.45)
            if lento:
                j = int(np.argmax(pesos))
                donde = T(f"step {i + 1}: looks at “{palabras[j]}”  (α = {pesos[j]:.2f})", font_size=19, color=NARANJA)
                donde.move_to([cols[-1] + 0.75, 0.35, 0], aligned_edge=LEFT)
                if donde.get_right()[0] > 6.9:
                    donde.scale_to_fit_width(6.9 - donde.get_left()[0])
                self.play(FadeTransform(paso_txt, donde) if len(paso_txt) else FadeIn(donde), run_time=0.4)
                paso_txt = donde
            self.play(chars[i].animate.set_opacity(1), run_time=0.3 if lento else 0.2)
            if lento:
                self.pausa()
        self.play(FadeOut(marco), FadeOut(paso_txt), *[e.animate.set_fill(AZ_FILL) for e in estados])
        cierre = VGroup(T("Each output character reads", font_size=21), T("the states it needs.", font_size=21),
                        T("No one told the network which", font_size=18, color=GRIS_TXT),
                        T("word goes with which character.", font_size=18, color=GRIS_TXT),
                        T("The day digits look at the “de”", font_size=18, color=GRIS_TXT),
                        T("after the day: that state holds it.", font_size=18, color=GRIS_TXT)
                        ).arrange(DOWN, buff=0.1, aligned_edge=LEFT)
        cierre.move_to([cols[-1] + 0.75, -0.6, 0], aligned_edge=LEFT)
        if cierre.get_right()[0] > 6.9:
            cierre.scale_to_fit_width(6.9 - cierre.get_left()[0])
        fuente = T("Weights: seq2seq with additive attention, d = 24, notebook 5.5.", font_size=15,
                   color=GRIS_TXT).to_corner(DR, buff=0.2)
        self.play(FadeIn(cierre), FadeIn(fuente))
        self.pausa(2.5)
