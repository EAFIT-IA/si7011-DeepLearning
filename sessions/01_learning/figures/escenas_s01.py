"""Escenas Manim · SI7011 Deep Learning · Sesión 1: Fundamentos de redes neuronales.

Render (video lineal, 1080p):
    S01_VIDEO=1 manim -qh escenas_s01.py E1_ParametrosPerdida
Render para presentar con pausas (manim-slides):
    manim -qh escenas_s01.py E1_ParametrosPerdida && manim-slides E1_ParametrosPerdida

Mapa escena → diapositivas:
    E1_ParametrosPerdida      V02, V03, V04       diapositivas 7–9 y 14
    E2_PuntajeProbabilidad    V05, V06            diapositivas 10–12
    E3_DescensoGradiente      V07, V08            diapositivas 21–24
    E4_TasaAprendizaje        V09                 diapositiva 25
    E5_Activacion             V12                 diapositivas 28–29
    E6_TransformacionEspacio  V14                 diapositiva 32
"""
import numpy as np
from manim import *  # noqa: F401,F403

from comun import (
    AZUL, AZUL_CLASE, GRIS, GRIS_CLARO, NARANJA, ROJO, TEAL, TINTA,
    EscenaBase, datos_clasificacion, datos_regresion, ejes, elipse_nivel,
    entrenar_mlp_v, grad_mse, lectura, mse, optimo_mse, sigmoide,
)


# ======================================================================
# utilidades de dibujo
# ======================================================================
def recta_recortada(ax, w, b, xr, yr, **kw):
    """Segmento de ŷ = w x + b recortado a la ventana de los ejes."""
    x0, x1 = xr
    y0, y1 = yr
    xs = [x0, x1]
    if abs(w) > 1e-9:
        xs += [(y0 - b) / w, (y1 - b) / w]
    xs = sorted(v for v in xs if x0 - 1e-9 <= v <= x1 + 1e-9)
    validos = [v for v in xs if y0 - 1e-6 <= w * v + b <= y1 + 1e-6]
    if len(validos) < 2:
        return VMobject()
    a, c = min(validos), max(validos)
    return Line(ax.c2p(a, w * a + b), ax.c2p(c, w * c + b), **kw)


def curva(ax, puntos, **kw):
    v = VMobject(**kw)
    v.set_points_smoothly([ax.c2p(*p) for p in puntos])
    return v


def punto(pos, color, r=0.08, anillo=False):
    d = Dot(pos, radius=r, color=color)
    if anillo:
        d.set_stroke(WHITE, width=2.5, background=True)
    return d


# ======================================================================
# E1 · Parámetros, residuos y pérdida (S01-V02, V03, V04) · diapositivas 7–9, 14
# ======================================================================
class E1_ParametrosPerdida(EscenaBase):
    codigo = "S01-V02·V03·V04"
    titulo = "Parameters, residuals and loss"

    def construct(self):
        x, y = datos_regresion()
        w_opt, b_opt, L_opt, H = optimo_mse(x, y)
        w = ValueTracker(0.4)
        b = ValueTracker(0.8)
        XR, YR = (-3, 3), (-2, 7)

        ax = ejes([-3, 3, 1], [-2, 7, 1], 8.2, 5.0, "x", "y")
        ax.move_to(LEFT * 1.6 + DOWN * 0.25)
        puntos = VGroup(*[punto(ax.c2p(xi, yi), AZUL, 0.085) for xi, yi in zip(x, y)])

        recta = always_redraw(lambda: recta_recortada(
            ax, w.get_value(), b.get_value(), XR, YR, color=TEAL, stroke_width=4))
        preds = always_redraw(lambda: VGroup(*[
            Circle(radius=0.075, color=TEAL, stroke_width=3).set_fill(WHITE, 1)
            .move_to(ax.c2p(xi, w.get_value() * xi + b.get_value())) for xi in x]))

        # ---- panel de lecturas
        ecuacion = MathTex(r"\hat y = w\,x + b", font_size=40)
        ecuacion[0][0:2].set_color(TEAL)
        gw, nw = lectura("w", w.get_value())
        gb, nb = lectura("b", b.get_value())
        # bloque de pérdida (aparece más tarde, pero se usa para fijar el ancho del panel)
        gL, nL = lectura(r"\mathrm{MSE}", mse(w.get_value(), b.get_value(), x, y), fs=34, color=ROJO)
        formula = MathTex(r"=\frac{1}{n}\sum_i r_i^2", font_size=30, color=ROJO)
        unidades = Text("units: (units of y)²", font_size=18, color=GRIS)
        panel = VGroup(ecuacion, gw, gb).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        bloque_L = VGroup(gL, formula, unidades).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        todo = VGroup(panel, bloque_L).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        todo.to_edge(RIGHT, buff=0.35).to_edge(UP, buff=1.0)
        nw.add_updater(lambda m: m.set_value(w.get_value()))
        nb.add_updater(lambda m: m.set_value(b.get_value()))

        leyenda_datos = VGroup(
            VGroup(punto(ORIGIN, AZUL), Text("data point  (x, y)", font_size=20)).arrange(RIGHT, buff=0.15),
            VGroup(Circle(radius=0.075, color=TEAL, stroke_width=3), Text("prediction  (x, ŷ)", font_size=20)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        leyenda_datos.next_to(panel, DOWN, buff=0.5, aligned_edge=LEFT)

        # ---------------------------------------------------------- A · modelo
        self.play(Create(ax), FadeIn(ax.etiquetas))
        self.play(LaggedStart(*[GrowFromCenter(p) for p in puntos], lag_ratio=0.08))
        self.leyenda("The data (x, y) stay fixed the whole time.")
        self.play(Create(recta), FadeIn(preds), Write(panel), FadeIn(leyenda_datos))
        self.pausa()

        # ---------------------------------------------------------- A2 · cambiar w
        self.leyenda("Question: if we increase w with b fixed, what happens to each prediction?")
        self.pausa()
        fantasmas = VGroup(*[Circle(radius=0.075, color=GRIS, stroke_width=2).move_to(c.get_center())
                             for c in preds]).set_opacity(0.6)
        pivote = punto(ax.c2p(0, b.get_value()), NARANJA, 0.09)
        pivote_lbl = MathTex("(0,\\,b)", font_size=26, color=NARANJA).next_to(pivote, UL, buff=0.08)
        self.add(fantasmas)
        self.play(w.animate.set_value(1.5), run_time=3)
        flechas = VGroup(*[
            Arrow(ax.c2p(xi, 0.4 * xi + 0.8), ax.c2p(xi, 1.5 * xi + 0.8), buff=0.08,
                  color=NARANJA, stroke_width=3, max_tip_length_to_length_ratio=0.2)
            for xi in x])
        self.play(LaggedStart(*[GrowArrow(f) for f in flechas], lag_ratio=0.06),
                  FadeIn(pivote), FadeIn(pivote_lbl))
        self.leyenda("The line rotates around (0, b): Δŷ = x·Δw. For x < 0 the predictions go down.",
                     resaltar={"Δŷ = x·Δw": TEAL})
        self.pausa()
        self.play(FadeOut(flechas), FadeOut(fantasmas), FadeOut(pivote), FadeOut(pivote_lbl))

        # ---------------------------------------------------------- A3 · cambiar b
        self.leyenda("Question: if we increase b by 2 (w fixed), how much does each prediction change?")
        self.pausa()
        w_act, b_act = w.get_value(), b.get_value()
        fantasmas = VGroup(*[Circle(radius=0.075, color=GRIS, stroke_width=2).move_to(c.get_center())
                             for c in preds]).set_opacity(0.6)
        self.add(fantasmas)
        self.play(b.animate.set_value(b_act + 2), run_time=2.5)
        flechas = VGroup(*[
            Arrow(ax.c2p(xi, w_act * xi + b_act), ax.c2p(xi, w_act * xi + b_act + 2), buff=0.08,
                  color=NARANJA, stroke_width=3, max_tip_length_to_length_ratio=0.2)
            for xi in x])
        self.play(LaggedStart(*[GrowArrow(f) for f in flechas], lag_ratio=0.06))
        self.leyenda("They all go up by exactly 2: b shifts every prediction equally.")
        self.pausa()
        self.play(FadeOut(flechas), FadeOut(fantasmas))

        # ---------------------------------------------------------- B · residuos y MSE
        self.play(w.animate.set_value(0.7), b.animate.set_value(1.3), run_time=1.5)
        residuos = always_redraw(lambda: VGroup(*[
            DashedLine(ax.c2p(xi, yi), ax.c2p(xi, w.get_value() * xi + b.get_value()),
                       color=ROJO, stroke_width=3, dash_length=0.06)
            for xi, yi in zip(x, y)]))
        self.leyenda("Residual r = y − ŷ: vertical distance, not perpendicular to the line.",
                     resaltar={"r = y − ŷ": ROJO})
        self.bring_to_front(puntos)
        self.play(Create(residuos))
        self.bring_to_front(preds, puntos)
        self.pausa()

        def cuadrados():
            g = VGroup()
            for xi, yi in zip(x, y):
                p0 = ax.c2p(xi, yi)
                p1 = ax.c2p(xi, w.get_value() * xi + b.get_value())
                lado = abs(p1[1] - p0[1])
                if lado < 1e-3:
                    continue
                sq = Square(side_length=lado).set_fill(ROJO, 0.16).set_stroke(ROJO, 1.5, opacity=0.8)
                sq.move_to((p0 + p1) / 2 + RIGHT * lado / 2)
                g.add(sq)
            return g

        cuads = always_redraw(cuadrados)
        nL.add_updater(lambda m: m.set_value(mse(w.get_value(), b.get_value(), x, y)))
        self.play(FadeOut(leyenda_datos))
        self.play(FadeIn(cuads), Write(bloque_L))
        self.bring_to_front(residuos, preds, puntos)
        self.leyenda("Each square is r²: an error of 3 contributes 9, an error of 1 contributes 1.")
        self.pausa()

        self.leyenda("Question: if we move w and b toward the least-squares fit, what happens to the squares?")
        self.pausa()
        self.play(w.animate.set_value(w_opt), b.animate.set_value(b_opt), run_time=3)
        self.leyenda("The squares shrink and the MSE drops; it never reaches 0 with these data.")
        self.pausa()

        # ---------------------------------------------------------- C · espacios enlazados
        self.play(w.animate.set_value(0.5), b.animate.set_value(1.3), run_time=1.5)
        self.play(FadeOut(formula), FadeOut(unidades))
        izquierda = VGroup(ax, ax.etiquetas, puntos)
        lecturas = VGroup(gw, gb, gL)
        self.play(
            izquierda.animate.scale(0.62).move_to(LEFT * 3.55 + DOWN * 0.55),
            FadeOut(ecuacion),
            lecturas.animate.arrange(RIGHT, buff=0.6).move_to(LEFT * 3.55 + UP * 2.45),
            run_time=1.5,
        )
        t_datos = Text("Data space", font_size=22, color=AZUL, weight="BOLD")
        t_datos.next_to(ax, UP, buff=0.1)

        axL = ejes([-0.4, 2.6, 0.5], [0, 7, 1], 5.4, 3.6, "w", r"J(w)", decimales=1, fs=18)
        axL.move_to(RIGHT * 3.4 + DOWN * 0.45)
        b_fijo = b.get_value()
        curvaL = axL.plot(lambda t: mse(t, b_fijo, x, y), x_range=[-0.4, 2.6], color=ROJO, stroke_width=4)
        t_par = Text("Parameter space", font_size=22, color=AZUL, weight="BOLD")
        t_par.next_to(axL, UP, buff=0.35)
        nota_b = MathTex(rf"b = {b_fijo:.2f}\ \text{{(fixed)}}", font_size=28).next_to(t_par, DOWN, buff=0.1)
        marcador = always_redraw(lambda: punto(axL.c2p(w.get_value(), mse(w.get_value(), b_fijo, x, y)),
                                               NARANJA, 0.1, anillo=True))
        guia = always_redraw(lambda: DashedLine(
            axL.c2p(w.get_value(), 0), axL.c2p(w.get_value(), mse(w.get_value(), b_fijo, x, y)),
            color=NARANJA, stroke_width=2, dash_length=0.05))
        self.play(FadeIn(t_datos), Create(axL), FadeIn(axL.etiquetas), FadeIn(t_par), FadeIn(nota_b))
        self.play(Create(curvaL))
        self.play(FadeIn(marcador), Create(guia))
        self.leyenda("Each point on the curve is a complete model. The data do not change along it.")
        self.pausa()
        w_b = float(np.mean(x * (y - b_fijo)) / np.mean(x * x))
        self.play(w.animate.set_value(2.3), run_time=2.5)
        self.play(w.animate.set_value(0.0), run_time=3)
        self.play(w.animate.set_value(w_b), run_time=2)
        self.leyenda(f"With b fixed at {b_fijo:.2f}, the best slope is w ≈ {w_b:.2f}.")
        self.pausa()

        # ---- contornos de L(w, b)
        self.play(FadeOut(curvaL), FadeOut(marcador), FadeOut(guia), FadeOut(axL),
                  FadeOut(axL.etiquetas), FadeOut(nota_b))
        # misma escala en w y b (1.1 unidades de pantalla por unidad) para no deformar la geometría
        axC = ejes([-0.4, 2.6, 0.5], [0.0, 4.2, 0.5], 3.3, 4.62, "w", "b", decimales=1, fs=16)
        axC.move_to(RIGHT * 3.6 + DOWN * 0.75)
        niveles = [0.25, 0.5, 1.0, 1.75, 2.75, 4.0]
        tonos = color_gradient([GRIS_CLARO, ROJO], len(niveles))[::-1]
        contornos = VGroup()
        for nv, col in zip(niveles, tonos):
            pts = elipse_nivel(nv, w_opt, b_opt, L_opt, H)
            c = curva(axC, pts, color=col, stroke_width=2.5)
            contornos.add(c)
        etiquetas_nv = VGroup(*[
            MathTex(f"{nv:g}", font_size=18, color=GRIS).move_to(
                axC.c2p(*elipse_nivel(nv, w_opt, b_opt, L_opt, H)[40]))
            for nv in niveles[1:]])
        minimo = MathTex(r"\times", font_size=34, color=TINTA).move_to(axC.c2p(w_opt, b_opt))
        t_cont = Text("level curves of J(w, b)", font_size=18, color=GRIS).next_to(axC, UP, buff=0.05)
        self.play(t_par.animate.next_to(t_cont, UP, buff=0.12), Create(axC), FadeIn(axC.etiquetas),
                  FadeIn(t_cont))
        self.play(LaggedStart(*[Create(c) for c in contornos], lag_ratio=0.15), FadeIn(etiquetas_nv),
                  FadeIn(minimo))
        marcador2 = always_redraw(lambda: punto(axC.c2p(w.get_value(), b.get_value()), NARANJA, 0.1, anillo=True))
        self.play(FadeIn(marcador2))
        self.leyenda("Now w and b both change: each point (w, b) is one line on the left.")
        self.pausa()
        self.play(w.animate.set_value(1.9), b.animate.set_value(1.1), run_time=2.5)
        self.play(w.animate.set_value(0.3), b.animate.set_value(3.2), run_time=3)
        self.play(w.animate.set_value(w_opt), b.animate.set_value(b_opt), run_time=2.5)
        self.leyenda("Same level curve ⇒ same MSE. The MSE of a linear model is convex; a network's loss in general is not.")
        self.pausa()

        # ---------------------------------------------------------- D · dos configuraciones (diap. 14)
        cfgs = [("A", 0.5, 1.3), ("B", 1.5, 2.6)]
        marcas = VGroup()
        rectas_fijas = VGroup()
        for nombre, wa, ba in cfgs:
            self.play(w.animate.set_value(wa), b.animate.set_value(ba), run_time=1.5)
            m = punto(axC.c2p(wa, ba), TEAL, 0.09)
            et = MathTex(rf"{nombre}:\ J={mse(wa, ba, x, y):.2f}", font_size=24, color=TEAL)
            et.next_to(m, RIGHT, buff=0.12)
            r = recta_recortada(ax, wa, ba, XR, YR, color=GRIS, stroke_width=2.5)
            r = DashedLine(r.get_start(), r.get_end(), color=GRIS, stroke_width=2.5, dash_length=0.08)
            rl = MathTex(nombre, font_size=24, color=GRIS).next_to(r.get_end(), RIGHT, buff=0.08)
            self.play(FadeIn(m), Write(et), Create(r), FadeIn(rl))
            marcas.add(m, et)
            rectas_fijas.add(r, rl)
        self.leyenda("Question: how do we find a better configuration without trying every possibility?")
        self.pausa()
        self.wait(0.5)


# ======================================================================
# E2 · Puntaje, sigmoid y BCE (S01-V05, V06) · diapositivas 10–12
# ======================================================================
class E2_PuntajeProbabilidad(EscenaBase):
    codigo = "S01-V05·V06"
    titulo = "From score z to probability to loss"

    def construct(self):
        c0, c1 = datos_clasificacion()
        s = ValueTracker(0.0)          # posición del punto sobre su trayectoria
        et = ValueTracker(1.0)         # etiqueta verdadera del punto (1 o 0)
        A, B = np.array([2.1, 2.0]), np.array([-2.4, -2.3])

        def pos():
            return A + s.get_value() * (B - A)

        def z():
            p = pos()
            return p[0] + p[1]  # w1 = w2 = 1, b = 0

        def pr():
            return float(sigmoide(z()))

        def perdida():
            p = pr()
            return float(-np.log(p)) if et.get_value() > 0.5 else float(-np.log(1 - p))

        # ---- plano de entrada
        ax = ejes([-3, 3, 1], [-3, 3, 1], 5.3, 5.3, "x_1", "x_2", fs=18)
        ax.move_to(LEFT * 3.55 + DOWN * 0.3)
        reg1 = Polygon(ax.c2p(3, -3), ax.c2p(3, 3), ax.c2p(-3, 3), stroke_width=0).set_fill(NARANJA, 0.09)
        reg0 = Polygon(ax.c2p(-3, -3), ax.c2p(3, -3), ax.c2p(-3, 3), stroke_width=0).set_fill(AZUL_CLASE, 0.09)
        frontera = DashedLine(ax.c2p(-3, 3), ax.c2p(3, -3), color=TINTA, stroke_width=3)
        lbl_f = MathTex("z=0", font_size=28).next_to(ax.c2p(-2.4, 2.4), RIGHT, buff=0.15)
        lbl_pos = MathTex("z>0", font_size=26, color=NARANJA).move_to(ax.c2p(2.2, -1.2))
        lbl_neg = MathTex("z<0", font_size=26, color=AZUL_CLASE).move_to(ax.c2p(-2.2, 1.2))
        pts0 = VGroup(*[punto(ax.c2p(*p), AZUL_CLASE, 0.075) for p in c0])
        pts1 = VGroup(*[punto(ax.c2p(*p), NARANJA, 0.075) for p in c1])
        formula = MathTex(r"z = w_1x_1 + w_2x_2 + b", r"\quad (w_1=w_2=1,\ b=0)", font_size=28)
        formula[1].set_color(GRIS).scale(0.85)
        formula.next_to(ax, UP, buff=0.12)
        clases = VGroup(
            VGroup(punto(ORIGIN, NARANJA), Text("class 1", font_size=18)).arrange(RIGHT, buff=0.1),
            VGroup(punto(ORIGIN, AZUL_CLASE), Text("class 0", font_size=18)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3).next_to(ax, DOWN, buff=0.08).shift(RIGHT * 1.6)

        def color_et():
            return NARANJA if et.get_value() > 0.5 else AZUL_CLASE

        movil = always_redraw(lambda: VGroup(
            Circle(radius=0.17, color=TINTA, stroke_width=2.5).move_to(ax.c2p(*pos())),
            punto(ax.c2p(*pos()), color_et(), 0.11),
        ))
        movil_lbl = always_redraw(lambda: MathTex(
            f"y={int(round(et.get_value()))}", font_size=26, color=color_et()
        ).next_to(movil, UR, buff=0.02))

        # ---- lecturas
        gz, nz = lectura("z", z(), fs=30)
        gp, np_ = lectura("p", pr(), decimales=3, fs=30)
        gl, nl = lectura(r"L", perdida(), decimales=3, fs=30, color=ROJO)
        nz.add_updater(lambda m: m.set_value(z()))
        np_.add_updater(lambda m: m.set_value(pr()))
        nl.add_updater(lambda m: m.set_value(perdida()))
        lect = VGroup(gz, gp, gl).arrange(RIGHT, buff=0.55).move_to(RIGHT * 3.4 + UP * 2.75)

        # ---- panel sigmoid
        axS = ejes([-6, 6, 2], [0, 1, 0.5], 5.0, 1.75, "z", "p", decimales=1, fs=16)
        axS.move_to(RIGHT * 3.4 + UP * 1.0)
        curvaS = axS.plot(sigmoide, x_range=[-6, 6], color=TEAL, stroke_width=4)
        umbral = DashedLine(axS.c2p(-6, 0.5), axS.c2p(6, 0.5), color=GRIS, stroke_width=1.5, dash_length=0.06)
        formS = MathTex(r"p=\sigma(z)=\frac{1}{1+e^{-z}}", font_size=24).move_to(axS.c2p(-3.6, 0.8))
        marcaS = always_redraw(lambda: VGroup(
            DashedLine(axS.c2p(z(), 0), axS.c2p(z(), pr()), color=NARANJA, stroke_width=2, dash_length=0.05),
            DashedLine(axS.c2p(z(), pr()), axS.c2p(0, pr()), color=NARANJA, stroke_width=2, dash_length=0.05),
            punto(axS.c2p(z(), pr()), NARANJA, 0.09, anillo=True)))
        refs = VGroup()
        for zz, txt in [(-2, "0.12"), (0, "0.5"), (2, "0.88")]:
            d = punto(axS.c2p(zz, sigmoide(zz)), GRIS, 0.06)
            t = MathTex(txt, font_size=20, color=GRIS).next_to(d, UL if zz < 1 else DR, buff=0.05)
            refs.add(VGroup(d, t))

        # ---- panel BCE
        axB = ejes([0, 1, 0.5], [0, 5, 1], 5.0, 1.75, "p", r"L", decimales=1, fs=16)
        axB.move_to(RIGHT * 3.4 + DOWN * 1.85)
        pmin = np.exp(-5)
        c_y1 = axB.plot(lambda p: -np.log(p), x_range=[pmin, 1 - 1e-4], color=NARANJA, stroke_width=4)
        c_y0 = axB.plot(lambda p: -np.log(1 - p), x_range=[1e-4, 1 - pmin], color=AZUL_CLASE, stroke_width=4)
        lbl_y1 = MathTex(r"y=1:\ -\log p", font_size=22, color=NARANJA)
        lbl_y1.next_to(axB.c2p(0.035, 3.6), RIGHT, buff=0.12)
        lbl_y0 = MathTex(r"y=0:\ -\log(1-p)", font_size=22, color=AZUL_CLASE)
        lbl_y0.next_to(axB.c2p(0.965, 3.6), LEFT, buff=0.12)

        def realce(m):
            activo_y1 = et.get_value() > 0.5
            c_y1.set_stroke(opacity=1.0 if activo_y1 else 0.25)
            lbl_y1.set_opacity(1.0 if activo_y1 else 0.35)
            c_y0.set_stroke(opacity=0.25 if activo_y1 else 1.0)
            lbl_y0.set_opacity(0.35 if activo_y1 else 1.0)
        realzador = Mobject()
        realzador.add_updater(realce)
        marcaB = always_redraw(lambda: punto(axB.c2p(pr(), perdida()), ROJO, 0.09, anillo=True))
        refsB = VGroup(
            VGroup(punto(axB.c2p(0.9, -np.log(0.9)), GRIS, 0.06),
                   MathTex(r"0.9\to 0.105", font_size=19, color=GRIS).next_to(axB.c2p(0.9, -np.log(0.9)), UP, buff=0.12)),
            VGroup(punto(axB.c2p(0.1, -np.log(0.1)), GRIS, 0.06),
                   MathTex(r"0.1\to 2.303", font_size=19, color=GRIS).next_to(axB.c2p(0.1, -np.log(0.1)), RIGHT, buff=0.12)),
        )

        # ---------------------------------------------------------- 1 · puntaje
        self.play(Create(ax), FadeIn(ax.etiquetas), Write(formula))
        self.play(FadeIn(reg1), FadeIn(reg0), LaggedStart(*[GrowFromCenter(p) for p in [*pts0, *pts1]], lag_ratio=0.03),
                  FadeIn(clases))
        self.play(Create(frontera), Write(lbl_f), FadeIn(lbl_pos), FadeIn(lbl_neg))
        self.leyenda("The score z can take any real value. The line z = 0 separates two decisions.")
        self.play(FadeIn(movil), FadeIn(movil_lbl), FadeIn(gz))
        self.pausa()
        self.play(s.animate.set_value(0.72), run_time=3.5)
        self.play(s.animate.set_value(0.25), run_time=2.5)
        self.leyenda("Question: z changes sign when it crosses the boundary. Is z a probability?")
        self.pausa()

        # ---------------------------------------------------------- 2 · sigmoid
        self.play(Create(axS), FadeIn(axS.etiquetas), Create(umbral), Write(formS))
        self.play(Create(curvaS), FadeIn(gp))
        self.play(FadeIn(refs), FadeIn(marcaS))
        self.leyenda("No: the sigmoid σ(z) maps z into (0, 1). The threshold p ≥ 0.5 is the same as z ≥ 0.")
        self.pausa()
        self.play(s.animate.set_value(0.7), run_time=3)
        self.play(s.animate.set_value(0.35), run_time=2)
        self.play(FadeOut(refs))
        self.pausa()

        # ---------------------------------------------------------- 3 · BCE
        self.add(realzador)
        self.play(Create(axB), FadeIn(axB.etiquetas))
        self.play(Create(c_y1), Create(c_y0), FadeIn(lbl_y1), FadeIn(lbl_y0), FadeIn(gl))
        self.play(FadeIn(marcaB), FadeIn(refsB))
        self.leyenda("The point has y = 1: its loss is −log p (natural log).",
                     resaltar={"−log p": NARANJA})
        self.pausa()
        self.play(s.animate.set_value(0.08), run_time=2)
        self.leyenda("Confident and correct: p near 1 and loss near 0.")
        self.pausa()
        self.leyenda("Question: what happens to the loss if the point moves deep into the wrong region?")
        self.pausa()
        self.play(s.animate.set_value(1.0), run_time=5, rate_func=linear)
        self.leyenda("Confident and wrong ⇒ large loss. As p → 0, −log p grows without bound.")
        self.pausa()

        # ---------------------------------------------------------- 4 · cambiar la etiqueta
        self.leyenda("Question: same position and same p, but now the true label is y = 0. What loss do you expect?")
        self.pausa()
        self.play(et.animate.set_value(0.0), run_time=1.5)
        self.leyenda("BCE scores the probability given to the true class: p for y = 1, 1 − p for y = 0.")
        self.pausa()
        self.wait(0.5)


# ======================================================================
# E3 · Derivada, gradiente y actualización (S01-V07, V08) · diapositivas 21–24
# ======================================================================
class E3_DescensoGradiente(EscenaBase):
    codigo = "S01-V07·V08"
    titulo = "Derivative, gradient and update"

    def construct(self):
        L = lambda w: (w - 3) ** 2          # noqa: E731
        dL = lambda w: 2 * (w - 3)          # noqa: E731
        eta = 0.1

        ax = ejes([-0.5, 6, 1], [0, 10, 2], 6.2, 4.5, "w", r"L(w)")
        ax.move_to(LEFT * 3.3 + DOWN * 0.45)
        titulo_L = MathTex(r"L(w)=(w-3)^2", font_size=30).next_to(ax, UP, buff=0.1).shift(RIGHT * 0.5)
        curvaL = ax.plot(L, x_range=[-0.16, 6.16], color=ROJO, stroke_width=4)
        wt = ValueTracker(1.0)

        def tangente(w0, semi=0.7, color=TEAL):
            m = dL(w0)
            return Line(ax.c2p(w0 - semi, L(w0) - semi * m), ax.c2p(w0 + semi, L(w0) + semi * m),
                        color=color, stroke_width=3.5)

        marcador = always_redraw(lambda: punto(ax.c2p(wt.get_value(), L(wt.get_value())), NARANJA, 0.1, anillo=True))
        tang = always_redraw(lambda: tangente(wt.get_value()))

        # panel derecho: texto + pérdida frente a iteración
        axT = ejes([0, 6, 1], [0, 4.5, 1], 4.8, 2.4, "t", r"L_t", fs=16)
        axT.move_to(RIGHT * 3.7 + DOWN * 1.65)
        t_axT = Text("loss vs. iteration", font_size=18, color=GRIS).next_to(axT, UP, buff=0.05)
        zona_texto = RIGHT * 3.6 + UP * 1.4

        def bloque(*lineas, fs=28):
            g = VGroup(*[MathTex(l, font_size=fs) for l in lineas]).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
            if g.width > 6.2:
                g.scale_to_fit_width(6.2)
            return g.move_to(zona_texto)

        # ---------------------------------------------------------- 1 · sensibilidad local
        self.play(Create(ax), FadeIn(ax.etiquetas), Write(titulo_L))
        self.play(Create(curvaL))
        self.play(FadeIn(marcador))
        txt = bloque(r"L(w+\Delta w)\approx L(w)+L'(w)\,\Delta w",
                     r"L'(w)=2(w-3)")
        self.play(Write(txt))
        self.leyenda("Question: at w = 1, should we increase or decrease w to reduce the loss?")
        self.pausa()
        self.play(Create(tang))
        sondas = VGroup()
        for dw, col in [(0.3, TEAL), (-0.3, ROJO)]:
            d = punto(ax.c2p(1 + dw, L(1 + dw)), col, 0.07)
            et = MathTex(rf"L({1 + dw:.1f})={L(1 + dw):.2f}", font_size=22, color=col)
            et.next_to(d, RIGHT if dw > 0 else LEFT, buff=0.12)
            sondas.add(VGroup(d, et))
        txt2 = bloque(r"L(w+\Delta w)\approx L(w)+L'(w)\,\Delta w",
                      r"L'(1)=-4<0")
        self.play(TransformMatchingTex(txt, txt2), FadeIn(sondas))
        txt = txt2
        self.leyenda("L′(1) = −4: a small increase in w reduces the loss. (L′ is not the model's slope w.)")
        self.pausa()
        self.play(FadeOut(sondas))

        # ---------------------------------------------------------- 2 · primera actualización
        txt2 = bloque(r"w_{t+1}=w_t-\eta\,L'(w_t),\qquad \eta=0.1",
                      r"w_1 = 1-0.1\,(-4)=1.4",
                      r"L(1)=4\ \longrightarrow\ L(1.4)=2.56")
        self.play(FadeOut(txt), FadeIn(txt2))
        txt = txt2
        self.play(Create(axT), FadeIn(axT.etiquetas), FadeIn(t_axT))
        hist = [(0, L(1.0))]
        pts_t = VGroup(punto(axT.c2p(0, L(1.0)), NARANJA, 0.07))
        self.play(FadeIn(pts_t))
        paso = Arrow(ax.c2p(1, 0), ax.c2p(1.4, 0), buff=0, color=NARANJA, stroke_width=5,
                     max_tip_length_to_length_ratio=0.45).shift(UP * 0.18)
        paso_lbl = MathTex(r"-\eta\,L'=+0.4", font_size=22, color=NARANJA).next_to(paso, UP, buff=0.05)
        self.play(GrowArrow(paso), FadeIn(paso_lbl))
        self.leyenda("Subtracting a negative gradient increases the parameter.")
        self.pausa()
        rastro = VGroup()

        def dar_paso(t, rapido=False):
            w0 = wt.get_value()
            w1 = w0 - eta * dL(w0)
            rastro.add(punto(ax.c2p(w0, L(w0)), GRIS, 0.055))
            self.add(rastro)
            hist.append((t, L(w1)))
            nuevo = punto(axT.c2p(t, L(w1)), NARANJA, 0.07)
            seg = Line(axT.c2p(*hist[-2]), axT.c2p(*hist[-1]), color=NARANJA, stroke_width=2.5)
            self.play(wt.animate.set_value(w1), Create(seg), FadeIn(nuevo), run_time=0.8 if rapido else 1.8)
            pts_t.add(seg, nuevo)

        dar_paso(1)
        self.play(FadeOut(paso), FadeOut(paso_lbl))
        self.pausa()

        # ---------------------------------------------------------- 3 · segunda actualización
        txt2 = bloque(r"L'(1.4)=-3.2",
                      r"w_2 = 1.4-0.1\,(-3.2)=1.72",
                      r"L(1.72)=1.6384")
        self.play(FadeOut(txt), FadeIn(txt2))
        txt = txt2
        self.leyenda("We recompute the gradient after every update. The second step has size 0.32.")
        dar_paso(2)
        self.pausa()

        # ---------------------------------------------------------- 4 · más iteraciones
        self.leyenda("Question: with η fixed, will the next steps be longer or shorter?")
        self.pausa()
        for t in range(3, 7):
            dar_paso(t, rapido=True)
        txt2 = bloque(r"w_t-3=0.8\,(w_{t-1}-3)", r"w_6\approx 2.476,\quad L_6\approx 0.275")
        self.play(FadeOut(txt), FadeIn(txt2))
        txt = txt2
        self.leyenda("Shorter: |L′| shrinks near the minimum, even though η does not change.")
        self.pausa()

        # ---------------------------------------------------------- 5 · empezar en w0 = 5
        self.play(FadeOut(rastro), *[FadeOut(m) for m in pts_t], FadeOut(txt), FadeOut(axT), FadeOut(t_axT))
        self.leyenda("Question: and if we start at w₀ = 5?")
        self.play(wt.animate.set_value(5.0), run_time=1.5)
        txt = bloque(r"L'(5)=4>0", r"w_1=5-0.1\,(4)=4.6")
        self.pausa()
        paso = Arrow(ax.c2p(5, 0), ax.c2p(4.6, 0), buff=0, color=NARANJA, stroke_width=5,
                     max_tip_length_to_length_ratio=0.45).shift(UP * 0.18)
        self.play(Write(txt), GrowArrow(paso))
        self.play(wt.animate.set_value(4.6), run_time=1.5)
        self.leyenda("Positive gradient ⇒ the step −η·L′ moves w toward smaller values.")
        self.pausa()

        # ---------------------------------------------------------- 6 · dos parámetros (diap. 22)
        self.play(*[FadeOut(m) for m in [ax, titulo_L, curvaL, marcador, tang, txt, paso]])
        self.quitar_leyenda()
        x, y = datos_regresion()
        w_opt, b_opt, L_opt, H = optimo_mse(x, y)
        esc = 1.2  # misma escala en ambos ejes: la geometría euclídea se conserva
        axC = ejes([-0.4, 2.6, 0.5], [0.0, 4.2, 0.5], 3.0 * esc, 4.2 * esc, "w", "b", decimales=1, fs=16)
        axC.move_to(LEFT * 3.0 + DOWN * 0.5)
        niveles = [0.25, 0.5, 1.0, 1.75, 2.75, 4.0]
        tonos = color_gradient([GRIS_CLARO, ROJO], len(niveles))[::-1]
        contornos = VGroup(*[curva(axC, elipse_nivel(nv, w_opt, b_opt, L_opt, H), color=c, stroke_width=2.5)
                             for nv, c in zip(niveles, tonos)])
        sub = Text("MSE of the regression above: level curves of J(w, b)", font_size=18, color=GRIS)
        sub.next_to(axC, UP, buff=0.1)
        self.play(Create(axC), FadeIn(axC.etiquetas), FadeIn(sub))
        self.play(LaggedStart(*[Create(c) for c in contornos], lag_ratio=0.12))

        th = np.array([0.6, 1.0])
        g = grad_mse(*th, x, y)
        u = g / np.linalg.norm(g)
        p0 = axC.c2p(*th)
        largo = 1.1
        f_grad = Arrow(p0, p0 + largo * np.array([u[0], u[1], 0]), buff=0, color=ROJO, stroke_width=5)
        f_desc = Arrow(p0, p0 - largo * np.array([u[0], u[1], 0]), buff=0, color=TEAL, stroke_width=5)
        l_grad = MathTex(r"\nabla J", font_size=28, color=ROJO).next_to(f_grad.get_end(), DOWN, buff=0.08)
        l_desc = MathTex(r"-\nabla J", font_size=28, color=TEAL).next_to(f_desc.get_end(), UP, buff=0.08)
        m0 = punto(p0, NARANJA, 0.1, anillo=True)
        texto = VGroup(
            MathTex(r"\nabla J(w,b)=\begin{bmatrix}\partial J/\partial w\\ \partial J/\partial b\end{bmatrix}", font_size=32),
            MathTex(rf"\nabla J({th[0]:.1f},\,{th[1]:.1f})\approx\begin{{bmatrix}}{g[0]:.2f}\\ {g[1]:.2f}\end{{bmatrix}}", font_size=30),
            Text("Arrows have fixed length: they show direction only.", font_size=18, color=GRIS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.4 + UP * 0.9)
        self.play(FadeIn(m0), Write(texto[0]))
        self.leyenda("Question: at this point, in which direction does the loss grow fastest?")
        self.pausa()
        self.play(GrowArrow(f_grad), FadeIn(l_grad), Write(texto[1]), FadeIn(texto[2]))
        self.play(GrowArrow(f_desc), FadeIn(l_desc))
        self.leyenda("∇J points to the fastest local increase, perpendicular to the level curve. −∇J: descent.")
        self.pausa()

        # trayectoria de descenso por gradiente
        self.play(FadeOut(f_grad), FadeOut(l_grad), FadeOut(f_desc), FadeOut(l_desc))
        regla = MathTex(r"\theta_{t+1}=\theta_t-\eta\,\nabla_\theta J(\theta_t),\quad \eta=0.1", font_size=30)
        regla.next_to(texto, DOWN, buff=0.45, aligned_edge=LEFT)
        gL, nL = lectura(r"J(\theta_t)", mse(*th, x, y), fs=30, color=ROJO)
        gT, nT = lectura("t", 0, decimales=0, fs=30)
        lect = VGroup(gT, gL).arrange(RIGHT, buff=0.6).next_to(regla, DOWN, buff=0.35, aligned_edge=LEFT)
        self.play(Write(regla), FadeIn(lect))
        actual = th.copy()
        for t in range(1, 16):
            nuevo = actual - 0.1 * grad_mse(*actual, x, y)
            seg = Line(axC.c2p(*actual), axC.c2p(*nuevo), color=NARANJA, stroke_width=3)
            d = punto(axC.c2p(*nuevo), NARANJA, 0.05)
            rt = 0.7 if t <= 4 else 0.3
            self.play(Create(seg), FadeIn(d), nT.animate.set_value(t),
                      nL.animate.set_value(mse(*nuevo, x, y)), run_time=rt)
            actual = nuevo
        self.leyenda("Each step uses the gradient at the current point. The path bends toward the minimum.")
        self.pausa()
        self.wait(0.5)


# ======================================================================
# E4 · Tasa de aprendizaje (S01-V09) · diapositiva 25
# ======================================================================
class E4_TasaAprendizaje(EscenaBase):
    codigo = "S01-V09"
    titulo = "Learning rate"

    def construct(self):
        L = lambda w: (w - 3) ** 2  # noqa: E731
        etas = [0.05, 0.4, 1.1]
        comentarios = ["slow", "fast", "oscillates and diverges"]
        xs = [-4.6, 0.0, 4.6]
        T = 6

        regla = MathTex(r"w_{t+1}-3=(1-2\eta)\,(w_t-3),\qquad w_0=1", font_size=32)
        regla.next_to(self.encabezado, DOWN, buff=0.2).set_x(0)
        self.play(Write(regla))

        columnas = []
        for eta, cx in zip(etas, xs):
            axP = ejes([-3, 9, 3], [0, 36, 12], 3.9, 2.0, "w", r"L", fs=15)
            axP.move_to(np.array([cx, 0.75, 0]))
            axT = ejes([0, T, 1], [0, 36, 12], 3.9, 1.35, "t", r"L_t", fs=15)
            axT.move_to(np.array([cx, -1.95, 0]))
            cab = MathTex(rf"\eta={eta:g}", rf"\quad 1-2\eta={1 - 2 * eta:g}", font_size=28)
            cab[1].set_color(GRIS)
            cab.next_to(axP, UP, buff=0.15)
            curvaP = axP.plot(L, x_range=[-3, 9], color=ROJO, stroke_width=3)
            columnas.append(dict(eta=eta, axP=axP, axT=axT, cab=cab, curva=curvaP, w=1.0, hist=[(0, 4.0)]))
        self.play(*[AnimationGroup(Create(c["axP"]), Create(c["axT"]), FadeIn(c["cab"])) for c in columnas])
        self.play(*[Create(c["curva"]) for c in columnas])
        for c in columnas:
            c["dot"] = punto(c["axP"].c2p(1, 4), NARANJA, 0.08, anillo=True)
            c["dotT"] = punto(c["axT"].c2p(0, 4), NARANJA, 0.06)
        self.play(*[FadeIn(c["dot"]) for c in columnas], *[FadeIn(c["dotT"]) for c in columnas])
        gT, nT = lectura("t", 0, decimales=0, fs=30)
        gT.next_to(regla, RIGHT, buff=0.8)
        self.add(gT)
        self.leyenda("Question: same function, same w₀. What will each learning rate η do over 6 steps?")
        self.pausa()

        for t in range(1, T + 1):
            anims = [nT.animate.set_value(t)]
            for c in columnas:
                w0 = c["w"]
                w1 = 3 + (1 - 2 * c["eta"]) * (w0 - 3)
                ax = c["axP"]
                salto = Arrow(ax.c2p(w0, L(w0)), ax.c2p(w1, L(w1)), buff=0.06, color=NARANJA,
                              stroke_width=2.5, max_tip_length_to_length_ratio=0.12, max_stroke_width_to_length_ratio=10)
                nuevo = punto(ax.c2p(w1, L(w1)), NARANJA, 0.08, anillo=True)
                viejo = punto(ax.c2p(w0, L(w0)), GRIS, 0.045)
                c["hist"].append((t, L(w1)))
                seg = Line(c["axT"].c2p(*c["hist"][-2]), c["axT"].c2p(*c["hist"][-1]), color=NARANJA, stroke_width=2.5)
                dT = punto(c["axT"].c2p(t, L(w1)), NARANJA, 0.05)
                anims += [GrowArrow(salto), Transform(c["dot"], nuevo), FadeIn(viejo), Create(seg), FadeIn(dT)]
                c["w"] = w1
            self.play(*anims, run_time=1.0 if t < 3 else 0.7)
        notas = VGroup(*[
            Text(txt, font_size=22, color=TEAL if i < 2 else ROJO, weight="BOLD").next_to(c["axT"], DOWN, buff=0.12)
            for i, (txt, c) in enumerate(zip(comentarios, columnas))])
        self.play(FadeIn(notas))
        self.leyenda("Here it converges if 0 < η < 1. These values belong to this function and scale; they are not a recipe for networks.")
        self.pausa()
        self.wait(0.5)


# ======================================================================
# E5 · Activaciones y no linealidad (S01-V12) · diapositivas 28–29
# ======================================================================
def poligonal(ax, f, quiebres, xr, **kw):
    xs = [xr[0]] + sorted(q for q in quiebres if xr[0] < q < xr[1]) + [xr[1]]
    v = VMobject(**kw)
    v.set_points_as_corners([ax.c2p(t, f(t)) for t in xs])
    return v


class E5_Activacion(EscenaBase):
    codigo = "S01-V12"
    titulo = "Why we need an activation"

    def construct(self):
        w1 = np.array([1.0, -1.0, 1.0])
        b1 = np.array([1.0, 0.5, -1.5])
        w2 = np.array([1.0, 1.0, -1.5])
        b2 = -0.5
        colores = [TEAL, NARANJA, AZUL_CLASE]
        XR = (-3, 3)
        quiebres = list(-b1 / w1)          # −1, 0.5, 1.5
        relu = lambda v: np.maximum(v, 0)  # noqa: E731

        axI = ejes([-3, 3, 1], [-5, 5, 1], 5.4, 3.4, "x", "a_j", fs=16)
        axI.move_to(LEFT * 3.5 + DOWN * 0.55)
        axO = ejes([-3, 3, 1], [-3, 8, 1], 5.4, 3.4, "x", "f(x)", fs=16)
        axO.move_to(RIGHT * 3.4 + DOWN * 0.55)

        tI = MathTex(r"\text{Hidden layer (no activation): } a_j = w_{1j}\,x + b_{1j}", font_size=28).next_to(axI, UP, buff=0.55)
        tO = MathTex(r"\text{Output: } f(x)=\textstyle\sum_j w_{2j}\,a_j + b_2", font_size=28).next_to(axO, UP, buff=0.55)
        pI = MathTex(r"w_1=(1,\,-1,\,1),\ \ b_1=(1,\,0.5,\,-1.5)", font_size=22, color=GRIS).next_to(tI, DOWN, buff=0.1)
        pO = MathTex(r"w_2=(1,\,1,\,-1.5),\ \ b_2=-0.5", font_size=22, color=GRIS).next_to(tO, DOWN, buff=0.1)

        def afin(j):
            return lambda t: w1[j] * t + b1[j]

        def lineas_ocultas(activ=False):
            g = VGroup()
            for j in range(3):
                f = (lambda t, j=j: relu(w1[j] * t + b1[j])) if activ else afin(j)
                g.add(poligonal(axI, f, quiebres if activ else [], XR, color=colores[j], stroke_width=4))
            return g

        def aportes(ax, activ=False):
            g = VGroup()
            for j in range(3):
                f = (lambda t, j=j: w2[j] * relu(w1[j] * t + b1[j])) if activ else \
                    (lambda t, j=j: w2[j] * (w1[j] * t + b1[j]))
                c = poligonal(ax, f, quiebres if activ else [], XR, color=colores[j], stroke_width=2.5)
                g.add(DashedVMobject(c, num_dashes=40, dashed_ratio=0.55))
            return g

        def salida(ax, activ=False):
            if activ:
                f = lambda t: float(np.dot(w2, relu(w1 * t + b1)) + b2)  # noqa: E731
            else:
                f = lambda t: float(np.dot(w2, w1 * t + b1) + b2)  # noqa: E731
            return poligonal(ax, f, quiebres if activ else [], XR, color=ROJO, stroke_width=5)

        # ---------------------------------------------------------- 1 · composición afín
        self.play(Create(axI), Create(axO), Write(tI), Write(tO), FadeIn(pI), FadeIn(pO))
        ocultas = lineas_ocultas()
        self.play(LaggedStart(*[Create(l) for l in ocultas], lag_ratio=0.3))
        et_ocultas = VGroup(*[MathTex(f"a_{j + 1}", font_size=24, color=colores[j]) for j in range(3)])
        for j, e in enumerate(et_ocultas):
            e.next_to(ocultas[j].get_end(), RIGHT, buff=0.08)
        self.play(FadeIn(et_ocultas))
        self.leyenda("Question: if we add these three lines with weights w₂ⱼ, what shape is f(x)?")
        self.pausa()
        ap = aportes(axO)
        self.play(*[TransformFromCopy(ocultas[j], ap[j]) for j in range(3)], run_time=1.5)
        sal = salida(axO)
        self.play(ReplacementTransform(ap.copy(), sal), ap.animate.set_opacity(0.35), run_time=1.5)
        alg = MathTex(r"f(x)=(w_2^\top w_1)\,x+(w_2^\top b_1+b_2)=-1.5\,x+3.25", font_size=28, color=ROJO)
        alg.to_edge(DOWN, buff=1.05)
        self.play(Write(alg))
        self.leyenda("Another line: a composition of affine maps is still affine.")
        self.pausa()

        # ---------------------------------------------------------- 2 · con ReLU
        self.leyenda("Question: and if we apply ReLU to each aⱼ before adding?")
        self.pausa()
        tI2 = MathTex(r"\text{Hidden layer: } a_j = \mathrm{ReLU}(w_{1j}\,x + b_{1j})", font_size=28).move_to(tI)
        axO2 = ejes([-3, 3, 1], [-2.5, 4.5, 1], 5.4, 3.4, "x", "f(x)", fs=16).move_to(axO)
        self.play(FadeOut(alg), TransformMatchingTex(tI, tI2))
        self.play(Transform(ocultas, lineas_ocultas(activ=True)), run_time=2)
        for j, e in enumerate(et_ocultas):
            e.generate_target()
            e.target.next_to(ocultas[j].get_end(), RIGHT, buff=0.08)
        self.play(*[MoveToTarget(e) for e in et_ocultas])
        self.play(Transform(axO, axO2), Transform(ap, aportes(axO2, activ=True)),
                  Transform(sal, salida(axO2, activ=True)), run_time=2)
        guias = VGroup()
        for q in quiebres:
            for axx in (axI, axO2):
                y0, y1 = axx.y_range[0], axx.y_range[1]
                guias.add(DashedLine(axx.c2p(q, y0), axx.c2p(q, y1), color=GRIS, stroke_width=1.5, dash_length=0.06))
        f_relu = lambda t: float(np.dot(w2, relu(w1 * t + b1)) + b2)  # noqa: E731
        codos = VGroup(*[punto(axO2.c2p(q, f_relu(q)), ROJO, 0.08) for q in quiebres])
        self.play(Create(guias), FadeIn(codos))
        alg2 = MathTex(r"\text{breakpoints at } x=-b_{1j}/w_{1j}:\ \ -1,\ 0.5,\ 1.5", font_size=28, color=ROJO)
        alg2.to_edge(DOWN, buff=1.05)
        self.play(Write(alg2))
        self.leyenda("Each ReLU adds one breakpoint: f is piecewise linear. Between two breakpoints the network is still linear.")
        self.pausa()

        # ---------------------------------------------------------- 3 · ReLU y sigmoid (diap. 29)
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self.encabezado and m is not self._leyenda])
        self.quitar_leyenda()
        axR = ejes([-4, 4, 2], [-0.5, 4, 1], 5.2, 3.6, "z", r"\mathrm{ReLU}(z)", fs=16).move_to(LEFT * 3.4 + DOWN * 0.3)
        axS = ejes([-4, 4, 2], [0, 1, 0.5], 5.2, 3.6, "z", r"\sigma(z)", decimales=1, fs=16).move_to(RIGHT * 3.4 + DOWN * 0.3)
        cR = poligonal(axR, lambda t: max(t, 0), [0], (-4, 4), color=TEAL, stroke_width=4)
        cS = axS.plot(sigmoide, x_range=[-4, 4], color=NARANJA, stroke_width=4)
        tR = VGroup(MathTex(r"\mathrm{ReLU}(z)=\max(0,z)", font_size=30),
                    Text("range [0, ∞) · hidden layers", font_size=18, color=GRIS)).arrange(DOWN, buff=0.1).next_to(axR, UP, buff=0.3)
        tS = VGroup(MathTex(r"\sigma(z)=\frac{1}{1+e^{-z}}", font_size=30),
                    Text("range (0, 1) · output probability", font_size=18, color=GRIS)).arrange(DOWN, buff=0.1).next_to(axS, UP, buff=0.3)
        self.play(Create(axR), Create(axS), Write(tR), Write(tS))
        self.play(Create(cR), Create(cS))
        self.leyenda("Question: what values do both functions give at z = −2, 0 and 2?")
        self.pausa()
        marcas = VGroup()
        for zz in (-2, 0, 2):
            r = max(zz, 0)
            s = sigmoide(zz)
            marcas.add(punto(axR.c2p(zz, r), TEAL, 0.07),
                       MathTex(f"{r:g}", font_size=22, color=TEAL).next_to(axR.c2p(zz, r), UP, buff=0.12))
            marcas.add(punto(axS.c2p(zz, s), NARANJA, 0.07),
                       MathTex(f"{s:.2f}" if zz else "0.5", font_size=22, color=NARANJA)
                       .next_to(axS.c2p(zz, s), UL if zz < 1 else DR, buff=0.08))
        self.play(LaggedStart(*[FadeIn(m) for m in marcas], lag_ratio=0.1))
        self.leyenda("ReLU: 0, 0, 2 (no upper bound; not differentiable at 0). Sigmoid: 0.12, 0.5, 0.88.")
        self.pausa()
        self.wait(0.5)


# ======================================================================
# E6 · La capa oculta transforma el espacio (S01-V14) · diapositiva 32
# ======================================================================
class E6_TransformacionEspacio(EscenaBase):
    codigo = "S01-V14"
    titulo = "Hidden representation and decision boundary"

    def construct(self):
        red = entrenar_mlp_v(semilla=3)
        X, Y, W1, b1, W2, b2 = red["X"], red["Y"], red["W1"], red["b1"], red["W2"], red["b2"]
        LIM = 2.2  # dominio de entrada

        a = ValueTracker(0.0)   # 0: x · 1: z1 = W1 x + b1 · 2: h = ReLU(z1)
        k = ValueTracker(3.0 / 2.5)
        mx, my = ValueTracker(0.0), ValueTracker(0.0)
        C = LEFT * 2.75 + DOWN * 0.4
        R = 3.0  # semilado de la ventana, en unidades de pantalla

        def etapa(p):
            av = a.get_value()
            z1 = p @ W1.T + b1
            if av <= 1:
                return (1 - av) * p + av * z1
            t = av - 1
            return (1 - t) * z1 + t * np.maximum(z1, 0)

        def pant(q):
            q = np.asarray(q, dtype=float)
            out = np.zeros(q.shape[:-1] + (3,))
            out[..., 0] = C[0] + k.get_value() * (q[..., 0] - mx.get_value())
            out[..., 1] = C[1] + k.get_value() * (q[..., 1] - my.get_value())
            return out

        def a_datos(pp):
            return np.array([mx.get_value() + (pp[0] - C[0]) / k.get_value(),
                             my.get_value() + (pp[1] - C[1]) / k.get_value()])

        def dentro(P, marg=0.0):
            return (np.abs(P[..., 0] - C[0]) <= R - marg) & (np.abs(P[..., 1] - C[1]) <= R - marg)

        def recortar(P):
            Q = P.copy()
            Q[..., 0] = np.clip(Q[..., 0], C[0] - R, C[0] + R)
            Q[..., 1] = np.clip(Q[..., 1], C[1] - R, C[1] + R)
            return Q

        def vista(centro, semirango):
            return [mx.animate.set_value(centro[0]), my.animate.set_value(centro[1]),
                    k.animate.set_value(R / semirango)]

        # vistas: entrada, z1 (se calcula para que quepa todo) y h (esquina del cuadrante)
        esquinas = np.array([[-LIM, -LIM], [LIM, -LIM], [LIM, LIM], [-LIM, LIM]])
        Zc = esquinas @ W1.T + b1
        c1 = (Zc.max(0) + Zc.min(0)) / 2
        r1 = 1.05 * np.max(Zc.max(0) - Zc.min(0)) / 2
        i1, i2 = -b2 / W2[0], -b2 / W2[1]          # cortes de w2ᵀh + b2 = 0 con los ejes
        r2 = 2.2 * max(i1, i2)
        c2 = np.array([0.75 * r2, 0.75 * r2]) - 0.1 * r2
        VISTA_X, VISTA_Z, VISTA_H = ((0, 0), 2.5), (tuple(c1), r1), (tuple(c2), r2)

        marco = Square(side_length=2 * R, color=GRIS, stroke_width=1.5).move_to(C)

        # rejilla del espacio de entrada
        vals = np.linspace(-2, 2, 9)
        s = np.linspace(-LIM, LIM, 89)
        lineas = [np.column_stack([np.full_like(s, v), s]) for v in vals] + \
                 [np.column_stack([s, np.full_like(s, v)]) for v in vals]

        def rejilla():
            g = VGroup()
            for i, L_ in enumerate(lineas):
                central = abs(vals[i % 9]) < 1e-9
                P = pant(etapa(L_))
                ok = dentro(P)
                # partir la polilínea en tramos visibles
                tramo = []
                for p_, o_ in zip(P, ok):
                    if o_:
                        tramo.append(p_)
                    elif tramo:
                        if len(tramo) > 1:
                            g.add(VMobject(stroke_color=GRIS if central else GRIS_CLARO,
                                           stroke_width=2.2 if central else 1.3).set_points_as_corners(tramo))
                        tramo = []
                if len(tramo) > 1:
                    g.add(VMobject(stroke_color=GRIS if central else GRIS_CLARO,
                                   stroke_width=2.2 if central else 1.3).set_points_as_corners(tramo))
            return g

        def nube():
            P = pant(etapa(X))
            ok = dentro(P, 0.04)
            return VGroup(*[Dot(p, radius=0.055, color=NARANJA if yy else AZUL_CLASE)
                            for p, yy, o in zip(P, Y, ok) if o])

        def ejes_actuales():
            o = pant(np.zeros(2))
            g = VGroup()
            if abs(o[1] - C[1]) < R:
                g.add(Line([C[0] - R, o[1], 0], [C[0] + R, o[1], 0], color=TINTA, stroke_width=1.5))
            if abs(o[0] - C[0]) < R:
                g.add(Line([o[0], C[1] - R, 0], [o[0], C[1] + R, 0], color=TINTA, stroke_width=1.5))
            return g

        g_rej = always_redraw(rejilla)
        g_ejes = always_redraw(ejes_actuales)
        g_pts = always_redraw(nube)

        # panel de pasos
        pasos = VGroup(
            MathTex(r"1.\ \ x\in\mathbb{R}^2", font_size=28),
            MathTex(r"2.\ \ z_1=W_1x+b_1", font_size=28),
            MathTex(r"3.\ \ a_1=\mathrm{ReLU}(z_1)", font_size=28),
            MathTex(r"4.\ \ w_2^\top a_1+b_2=0", font_size=28),
            MathTex(r"5.\ \ \text{boundary in } x", font_size=28),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(RIGHT * 3.4 + UP * 1.0)
        pasos.set_opacity(0.3)
        info = VGroup(
            Text("Trained 2–2–1 MLP, not hand-picked weights", font_size=17, color=GRIS),
            Text(f"numpy · seed {red['semilla']} · {red['iteraciones']} GD steps · η = {red['lr']}",
                 font_size=17, color=GRIS),
            Text(f"training accuracy: {100 * red['acc']:.0f} %", font_size=17, color=GRIS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(pasos, DOWN, buff=0.5, aligned_edge=LEFT)
        if info.get_right()[0] > 6.9:
            info.scale_to_fit_width(6.9 - info.get_left()[0]).next_to(pasos, DOWN, buff=0.5, aligned_edge=LEFT)
        leyenda_cl = VGroup(
            VGroup(punto(ORIGIN, NARANJA, 0.06), Text("class 1", font_size=17)).arrange(RIGHT, buff=0.1),
            VGroup(punto(ORIGIN, AZUL_CLASE, 0.06), Text("class 0", font_size=17)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3).next_to(info, DOWN, buff=0.25, aligned_edge=LEFT)
        rotulos = ["input space x", "z₁ = W₁x + b₁", "representation a₁"]
        nombre = always_redraw(lambda: Text(
            rotulos[min(int(round(a.get_value())), 2)],
            font_size=20, color=AZUL, weight="BOLD").next_to(marco, UP, buff=0.08))

        def activar(i):
            return [pasos[j].animate.set_opacity(1.0 if j == i else 0.3) for j in range(len(pasos))]

        # ---------------------------------------------------------- 1 · entrada
        self.add(marco)
        self.play(FadeIn(g_rej), FadeIn(g_ejes), FadeIn(nombre), FadeIn(pasos), FadeIn(info),
                  FadeIn(leyenda_cl), *activar(0))
        self.play(FadeIn(g_pts))
        self.leyenda("Two classes separated by a V: no straight line in x separates them.")
        self.pausa()

        # ---------------------------------------------------------- 2 · afín
        self.leyenda("Question: what does W₁x + b₁ do to the grid? Do straight lines stay straight?")
        self.pausa()
        self.play(*activar(1))
        self.play(a.animate.set_value(1.0), *vista(*VISTA_Z), run_time=3.5)
        self.leyenda("W₁x + b₁ rotates, stretches and shifts: straight lines stay straight and the V is still there.")
        self.pausa()

        # ---------------------------------------------------------- 3 · ReLU
        self.leyenda("Question: what does ReLU do to points with a negative coordinate?")
        self.pausa()
        self.play(*activar(2))
        self.play(a.animate.set_value(2.0), run_time=3)
        self.play(*vista(*VISTA_H), run_time=1.5)
        e_h = VGroup(MathTex("a_{1,1}", font_size=26).move_to(C + np.array([R - 0.3, -R + 0.3, 0])),
                     MathTex("a_{1,2}", font_size=26).move_to(C + np.array([-R + 0.3, R - 0.3, 0])))
        self.play(FadeIn(e_h))
        self.leyenda("ReLU sends negative coordinates to 0: the plane folds onto the first quadrant.")
        self.pausa()

        # ---------------------------------------------------------- 4 · frontera lineal en h
        hmax = a_datos(C + np.array([R, R, 0]))
        reg_neg = Polygon(*pant(np.array([[0, 0], [i1, 0], [0, i2]])), stroke_width=0)
        reg_pos = Polygon(*pant(np.array([[i1, 0], [hmax[0], 0], [hmax[0], hmax[1]], [0, hmax[1]], [0, i2]])),
                          stroke_width=0)
        # z = w2ᵀh + b2: z > 0 (clase 1) en el triángulo junto al origen si b2 > 0
        col_neg, col_pos = (NARANJA, AZUL_CLASE) if b2 > 0 else (AZUL_CLASE, NARANJA)
        reg_neg.set_fill(col_neg, 0.16)
        reg_pos.set_fill(col_pos, 0.16)
        extremo_a = np.array([1.6 * i1, -0.6 * i2])
        extremo_b = np.array([-0.6 * i1, 1.6 * i2])
        linea = Line(*pant(np.array([extremo_a, extremo_b])), color=TINTA, stroke_width=4)
        self.play(*activar(3))
        self.play(FadeIn(reg_neg), FadeIn(reg_pos), Create(linea))
        self.bring_to_front(g_pts)
        self.leyenda("In a₁, the output applies a linear score: one straight line separates the classes.")
        self.pausa()

        # ---------------------------------------------------------- 5 · volver al espacio de entrada
        self.leyenda("Question: if we go back to the input space, what shape does that boundary have in x?")
        self.pausa()
        n = 30
        bordes = np.linspace(-LIM, LIM, n + 1)
        esq_celdas, col_celdas = [], []
        for i in range(n):
            for j in range(n):
                x0, x1_, y0, y1 = bordes[i], bordes[i + 1], bordes[j], bordes[j + 1]
                centro = np.array([(x0 + x1_) / 2, (y0 + y1) / 2])
                zc = W2 @ np.maximum(W1 @ centro + b1, 0) + b2
                esq_celdas.append([[x0, y0], [x1_, y0], [x1_, y1], [x0, y1]])
                col_celdas.append(NARANJA if zc >= 0 else AZUL_CLASE)
        esq_celdas = np.array(esq_celdas)

        def dibujar_celdas():
            P = recortar(pant(etapa(esq_celdas)))
            g = VGroup()
            for poly, col in zip(P, col_celdas):
                g.add(Polygon(*poly, stroke_width=0).set_fill(col, 0.22))
            return g

        g_cel = always_redraw(dibujar_celdas)
        self.play(FadeIn(g_cel), FadeOut(reg_neg), FadeOut(reg_pos), FadeOut(linea), FadeOut(e_h))
        self.bring_to_front(g_rej, g_ejes, g_pts)
        self.play(*activar(4))
        self.play(a.animate.set_value(1.0), *vista(*VISTA_Z), run_time=3)
        self.play(a.animate.set_value(0.0), *vista(*VISTA_X), run_time=3)

        def z_pantalla(px, py):
            q = a_datos(np.array([px, py]))
            return float(W2 @ np.maximum(W1 @ q + b1, 0) + b2)
        m_ = 0.35
        frontera = ImplicitFunction(z_pantalla, x_range=[C[0] - R + m_, C[0] + R - m_],
                                    y_range=[C[1] - R + m_, C[1] + R - m_], color=TINTA, stroke_width=4)
        self.play(Create(frontera), run_time=2)
        self.bring_to_front(g_pts)
        self.leyenda("A straight line in a₁ is a piecewise boundary in x: the learned V.")
        self.pausa()
        self.leyenda("With seeds 2 and 7 the same training stops at 91–93 %: capacity does not guarantee good training.")
        self.pausa()
        self.wait(0.5)
