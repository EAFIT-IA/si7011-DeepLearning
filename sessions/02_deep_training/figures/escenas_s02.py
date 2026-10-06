"""
Escenas Manim · SI7011 · Sesión 2: Deep Training

    A01_SignalPropagation   std(a_l) y std(δ_{a_l}) a través de 30 capas ReLU
                            para tres escalas de inicialización (c = 0.5, 1, 2).
    A02_MomentumValley      SGD frente a Momentum en un valle cuadrático estrecho.
    A03_ResidualGradient    Bloque residual: el camino identidad deja pasar el
                            gradiente; red plana frente a residual con los mismos pesos.

Todos los datos se calculan con numpy y semillas fijas al construir la escena.
Notación del curso: z_l = W_l a_{l-1} + b_l, a_l = φ(z_l), δ_v = ∂L/∂v, J objetivo.

Render:
    S01_VIDEO=1 manim -qh escenas_s02.py A01_SignalPropagation      # video lineal
    manim -qh escenas_s02.py A01_SignalPropagation                  # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo (como E10 / E11)
FONDO = WHITE
NAVY = "#14213d"
GRIS_TXT = "#5b6b80"
GRIS_EJE = "#5b6577"
GRIS_CLARO = "#c4cbd5"
AZUL = "#1f6fb4"
TEAL = "#167d78"
NARANJA = "#d9631e"
AZ_FILL, AZ_BORDE = "#e4eefc", "#7aa6dc"
VE_FILL, VE_BORDE = "#e3f5ea", "#83c9a0"
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


def ejes_log(x_range, dec_min, dec_max, x_length, y_length, x_label, y_label, paso_dec=1):
    """Ejes lineales en log10(y); etiquetas 10^k en el eje vertical."""
    ax = Axes(
        x_range=x_range, y_range=[0, dec_max - dec_min, paso_dec], x_length=x_length, y_length=y_length,
        tips=False,
        axis_config={"color": GRIS_EJE, "stroke_width": 1.8, "include_ticks": True, "tick_size": 0.05},
        x_axis_config={"include_numbers": True, "font_size": 20,
                       "decimal_number_config": {"num_decimal_places": 0, "color": GRIS_EJE}},
    )
    marcas = VGroup(*[
        M(rf"10^{{{k}}}", font_size=22, color=GRIS_EJE).next_to(ax.c2p(x_range[0], k - dec_min), LEFT, buff=0.12)
        for k in range(dec_min, dec_max + 1, paso_dec)
    ])
    lx = M(x_label, font_size=26, color=GRIS_TXT).next_to(ax.x_axis, DOWN, buff=0.32)
    ax.base = dec_min      # y en pantalla = log10(v) - base
    ly = M(y_label, font_size=28, color=GRIS_TXT).next_to(ax.y_axis, UP, buff=0.15)
    return VGroup(ax, marcas, lx, ly), ax


# ======================================================================
# A01 · Signal propagation through depth
# ======================================================================
N_ANCHO, D_PROF, N_LOTE = 256, 30, 1000
ESCALAS = [(0.5, AZUL, r"c=0.5", "too small"), (1.0, TEAL, r"c=1\ \text{(He)}", "He"),
           (2.0, NARANJA, r"c=2", "too large")]


def propagacion(c, semilla=1):
    """MLP ReLU sin sesgo, W_l ~ N(0, c·2/n). Devuelve std(a_l) y std(δ_{a_l}), l = 1..D."""
    X = np.random.default_rng(0).normal(size=(N_LOTE, N_ANCHO))
    r = np.random.default_rng(semilla)
    a, Ws, Zs, As = X, [], [], []
    for _ in range(D_PROF):
        W = r.normal(0, np.sqrt(c * 2 / N_ANCHO), (N_ANCHO, N_ANCHO))
        z = a @ W.T
        a = np.maximum(z, 0)
        Ws.append(W); Zs.append(z); As.append(a)
    d = np.random.default_rng(9).normal(size=As[-1].shape)  # δ_{a_D}
    std_d = [d.std()]
    for l in range(D_PROF - 1, 0, -1):
        d = (d * (Zs[l] > 0)) @ Ws[l]                        # δ_{a_{l}} = W_{l+1}^T(δ ⊙ φ')
        std_d.append(d.std())
    return np.array([x.std() for x in As]), np.array(std_d[::-1])


class A01_SignalPropagation(EscenaBase):
    def construct(self):
        datos = {c: propagacion(c) for c, *_ in ESCALAS}

        tit = titulo("Signal propagation through depth")
        setup = VGroup(
            T("30-layer ReLU MLP, width 256, random weights:", font_size=20, color=GRIS_TXT),
            M(r"(W_l)_{ij}\sim\mathcal N\!\left(0,\ c\cdot\tfrac{2}{n_\text{in}}\right)", font_size=30, color=GRIS_TXT),
        ).arrange(RIGHT, buff=0.2).next_to(tit, DOWN, buff=0.2, aligned_edge=LEFT)

        grupo_ejes, ax = ejes_log([0, 30, 5], -5, 5, 8.2, 4.4, r"\text{layer } l", r"\operatorname{std}(a_l)")
        grupo_ejes.move_to([-1.7, -0.25, 0])
        uno = DashedLine(ax.c2p(0, 5), ax.c2p(30, 5), color=GRIS_CLARO, stroke_width=2, dash_length=0.08)
        uno_lbl = T("stable scale (std = 1)", font_size=16, color=GRIS_TXT).next_to(ax.c2p(0.5, 5), UR, buff=0.06)

        # leyenda / lecturas a la derecha
        X_LEY = 5.0
        estado = {"fase": 0, "l": 1}          # fase 0: forward (std a_l), 1: backward (std δ)

        def fmt(v):
            if v < 0.01 or v > 999:
                e = int(np.floor(np.log10(v)))
                return rf"{v / 10 ** e:.1f}\times10^{{{e}}}"
            return rf"{v:.2f}"

        filas = VGroup()
        for c, col, tex, nombre in ESCALAS:
            lin = Line(ORIGIN, RIGHT * 0.45, color=col, stroke_width=6)
            et = M(tex, font_size=28, color=col)
            filas.add(VGroup(lin, et).arrange(RIGHT, buff=0.15))
        filas.arrange(DOWN, aligned_edge=LEFT, buff=0.75).move_to([X_LEY, 0.35, 0])

        def lectura(k, c, col):
            def dib():
                v = datos[c][estado["fase"]][int(np.clip(round(estado["l"]), 1, D_PROF)) - 1]
                return M(r"\operatorname{std}=" + fmt(v), font_size=24, color=col).next_to(
                    filas[k], DOWN, buff=0.12, aligned_edge=LEFT)
            return always_redraw(dib)

        lecturas = VGroup(*[lectura(k, c, col) for k, (c, col, *_) in enumerate(ESCALAS)])

        def curva(valores, l_hasta, col, desde_derecha=False):
            ls = np.arange(1, D_PROF + 1)
            if desde_derecha:
                sel = ls >= l_hasta
            else:
                sel = ls <= l_hasta
            pts = [ax.c2p(l, np.log10(np.clip(v, 1e-5, 1e5)) + 5) for l, v in zip(ls[sel], valores[sel])]
            m = VMobject(stroke_color=col, stroke_width=4.5)
            if len(pts) > 1:
                m.set_points_as_corners(pts)
            return m

        # ---------------- presentación
        self.play(FadeIn(tit), FadeIn(setup), run_time=0.8)
        self.play(Create(grupo_ejes), Create(uno), FadeIn(uno_lbl), run_time=1.2)
        self.play(FadeIn(filas))
        self.add(lecturas)
        preg = nota("Question: what happens to std(a_l) layer after layer for each c?")
        preg[0:9].set_color(NARANJA)
        self.play(FadeIn(preg))
        self.pausa(1.5)

        # ---------------- forward
        l_t = ValueTracker(1)
        curvas = VGroup(*[
            always_redraw(lambda c=c, col=col: curva(datos[c][0], l_t.get_value(), col))
            for c, col, *_ in ESCALAS
        ])
        marcador = always_redraw(lambda: DashedLine(ax.c2p(l_t.get_value(), 0), ax.c2p(l_t.get_value(), 10),
                                                    color=GRIS_CLARO, stroke_width=1.5, dash_length=0.06))

        lector = Mobject().add_updater(lambda m: estado.__setitem__("l", l_t.get_value()))
        self.add(curvas, marcador, lector)
        fwd = nota("Forward: each layer multiplies the scale by about √c.")
        self.play(FadeOut(preg), FadeIn(fwd), run_time=0.5)
        self.play(l_t.animate.set_value(D_PROF), run_time=6, rate_func=linear)
        self.pausa(1.0)

        res = nota("c = 0.5: std(a_30) ≈ 5·10⁻⁵ (vanishes).  c = 2: ≈ 6·10⁴ (explodes).  He: stays near 1.")
        self.play(FadeOut(fwd), FadeIn(res))
        self.pausa(2.0)

        # ---------------- backward
        lector.clear_updaters()
        fijas = VGroup(*[curva(datos[c][0], D_PROF, col).set_stroke(opacity=0.22) for c, col, *_ in ESCALAS])
        self.add(fijas)
        self.remove(curvas, marcador)
        estado["fase"], estado["l"] = 1, D_PROF

        ly_b = M(r"\operatorname{std}(\delta_{a_l})", font_size=28, color=GRIS_TXT).move_to(grupo_ejes[3])
        tit_b = titulo("Backward: the same factor, from output to input")
        preg_b = nota("Question: backward starts at layer 30. What reaches layer 1?")
        preg_b[0:9].set_color(NARANJA)
        self.play(FadeOut(res), FadeIn(preg_b), Transform(grupo_ejes[3], ly_b),
                  FadeOut(tit), FadeIn(tit_b), run_time=0.8)
        self.pausa(1.5)

        lb = ValueTracker(D_PROF)
        curvas_b = VGroup(*[
            always_redraw(lambda c=c, col=col: curva(datos[c][1], lb.get_value(), col, desde_derecha=True))
            for c, col, *_ in ESCALAS
        ])
        marcador_b = always_redraw(lambda: DashedLine(ax.c2p(lb.get_value(), 0), ax.c2p(lb.get_value(), 10),
                                                      color=GRIS_CLARO, stroke_width=1.5, dash_length=0.06))

        lector_b = Mobject().add_updater(lambda m: estado.__setitem__("l", lb.get_value()))
        self.add(curvas_b, marcador_b, lector_b)
        bwd = nota("Gradients pass through the same weights, so they shrink or grow in the same way.")
        self.play(FadeOut(preg_b), FadeIn(bwd), run_time=0.5)
        self.play(lb.animate.set_value(1), run_time=6, rate_func=linear)
        self.pausa(1.0)

        cierre = nota("With He initialization both directions stay near 1: early layers can still learn.")
        self.play(FadeOut(bwd), FadeIn(cierre))
        self.wait(3.0)


# ======================================================================
# A02 · Momentum in a narrow valley
# ======================================================================
L1, L2 = 1.0, 20.0          # curvaturas: J = ½(L1 u² + L2 v²)
P0 = (-9.0, 1.5)
N_PASOS = 40
OPT = [("SGD", AZUL, 0.09, 0.0), ("Momentum", NARANJA, 0.04, 0.7)]


def J_val(p):
    return 0.5 * (L1 * p[0] ** 2 + L2 * p[1] ** 2)


def trayectoria(eta, beta):
    p, v = np.array(P0), np.zeros(2)
    P = [p.copy()]
    for _ in range(N_PASOS):
        g = np.array([L1 * p[0], L2 * p[1]])
        v = beta * v + g
        p = p - eta * v
        P.append(p.copy())
    return np.array(P)


class A02_MomentumValley(EscenaBase):
    def construct(self):
        trays = {n: trayectoria(eta, beta) for n, _, eta, beta in OPT}

        tit = titulo("Momentum in a narrow valley")
        formula = M(r"J(u,v)=\tfrac12\left(u^2+20\,v^2\right)", font_size=30).to_corner(UR, buff=0.4)

        # ---------- plano de parámetros (misma escala en u y v)
        ESC = 0.62
        ax = Axes(x_range=[-10, 2, 2], y_range=[-2, 2, 1], x_length=12 * ESC, y_length=4 * ESC, tips=False,
                  axis_config={"color": GRIS_EJE, "stroke_width": 1.6, "include_ticks": False})
        ax.move_to([-2.75, 0.85, 0])
        lu = M("u", font_size=28, color=GRIS_TXT).next_to(ax.x_axis.get_end(), RIGHT, buff=0.12)
        lv = M("v", font_size=28, color=GRIS_TXT).next_to(ax.y_axis.get_end(), UP, buff=0.08)
        niveles = [0.5, 2, 6, 14, 26, 42]
        contornos = VGroup()
        for c in niveles:
            a_, b_ = np.sqrt(2 * c / L1), np.sqrt(2 * c / L2)
            t = np.linspace(0, 2 * np.pi, 200)
            pts = [ax.c2p(a_ * np.cos(s), b_ * np.sin(s)) for s in t]
            pts = [p for p, s in zip(pts, t) if -10 <= a_ * np.cos(s) <= 2 and -2 <= b_ * np.sin(s) <= 2]
            if len(pts) > 2:
                contornos.add(VMobject(stroke_color="#a9c6ea", stroke_width=1.8).set_points_smoothly(pts))
        minimo = Dot(ax.c2p(0, 0), radius=0.07, color=NAVY)
        inicio = Circle(radius=0.08, color=NAVY, stroke_width=3).move_to(ax.c2p(*P0))
        etq_v = T("steep: across the valley", font_size=16, color=GRIS_TXT).next_to(ax.c2p(-0.2, 2), LEFT, buff=0.1)
        etq_u = T("shallow: along the valley", font_size=16, color=GRIS_TXT).next_to(ax.c2p(-5, -2), UP, buff=0.08)

        # ---------- J frente a paso (escala log)
        grupo_c, axc = ejes_log([0, N_PASOS, 10], -4, 2, 4.0, 3.0, r"\text{step } t", r"J(\theta_t)", paso_dec=2)
        grupo_c.move_to([4.45, 0.55, 0])

        # ---------- leyenda
        ley = VGroup()
        for n, col, eta, beta in OPT:
            txt = rf"\eta={eta:g}" if beta == 0 else rf"\eta={eta:g},\ \beta={beta:g}"
            ley.add(VGroup(Line(ORIGIN, RIGHT * 0.45, color=col, stroke_width=6),
                           T(n, font_size=20, color=col), M(txt, font_size=24, color=GRIS_TXT)).arrange(RIGHT, buff=0.15))
        ley.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-3.3, -1.85, 0])

        k_t = ValueTracker(0)
        contador = always_redraw(lambda: VGroup(M("t=", font_size=28),
                                                Integer(int(round(k_t.get_value())), font_size=28, color=NAVY))
                                 .arrange(RIGHT, buff=0.08).next_to(ax, UP, buff=0.12).align_to(ax, RIGHT))

        def ruta(n, col):
            def dib():
                k = int(round(k_t.get_value()))
                P = trays[n][:k + 1]
                g = VGroup()
                if k >= 1:
                    g.add(VMobject(stroke_color=col, stroke_width=3.5).set_points_as_corners([ax.c2p(*p) for p in P]))
                g.add(Dot(ax.c2p(*P[-1]), radius=0.065, color=col))
                return g
            return always_redraw(dib)

        def costo(n, col):
            def dib():
                k = int(round(k_t.get_value()))
                Js = [np.log10(max(J_val(p), 1e-4)) for p in trays[n][:k + 1]]
                pts = [axc.c2p(i, j + 4) for i, j in enumerate(Js)]
                g = VGroup(Dot(pts[-1], radius=0.05, color=col))
                if len(pts) > 1:
                    g.add(VMobject(stroke_color=col, stroke_width=3).set_points_as_corners(pts))
                return g
            return always_redraw(dib)

        # =============================================================
        self.play(FadeIn(tit), FadeIn(formula), run_time=0.8)
        self.play(Create(ax), FadeIn(lu), FadeIn(lv), LaggedStart(*[Create(c) for c in contornos], lag_ratio=0.1),
                  run_time=1.6)
        self.play(FadeIn(minimo), FadeIn(inicio), FadeIn(etq_u), FadeIn(etq_v))

        # gradiente en el punto inicial
        g0 = np.array([L1 * P0[0], L2 * P0[1]])
        dirg = -g0 / np.linalg.norm(g0)
        flecha = Arrow(ax.c2p(*P0), ax.c2p(P0[0] + 1.6 * dirg[0], P0[1] + 1.6 * dirg[1]), buff=0,
                       color=NARANJA, stroke_width=5, max_tip_length_to_length_ratio=0.25)
        flecha_lbl = M(r"-\nabla J", font_size=26, color=NARANJA).next_to(flecha.get_end(), RIGHT, buff=0.08)
        preg = nota("Question: the negative gradient points mostly across the valley. Where will SGD go?")
        preg[0:9].set_color(NARANJA)
        self.play(GrowArrow(flecha), FadeIn(flecha_lbl), FadeIn(preg))
        self.pausa(1.5)

        rutas = VGroup(*[ruta(n, col) for n, col, *_ in OPT])
        costos = VGroup(*[costo(n, col) for n, col, *_ in OPT])
        self.play(FadeOut(flecha), FadeOut(flecha_lbl), Create(grupo_c), FadeIn(ley), FadeIn(contador))
        self.add(rutas, costos)

        # primeros pasos, uno por uno
        n1 = nota("SGD bounces from wall to wall; Momentum's velocity averages those bounces out.")
        self.play(FadeOut(preg), FadeIn(n1))
        for k in range(1, 7):
            self.play(k_t.animate.set_value(k), run_time=0.55, rate_func=smooth)
        self.pausa(1.0)

        # velocidad acumulada de Momentum en t = 6
        _, _, eta_m, beta_m = OPT[1]
        P = trays["Momentum"]
        vel = (P[6] - P[7]) / eta_m if len(P) > 7 else None
        if vel is not None:
            dirv = -vel / max(np.linalg.norm(vel), 1e-9)
            fv = Arrow(ax.c2p(*P[6]), ax.c2p(P[6][0] + 1.4 * dirv[0], P[6][1] + 1.4 * dirv[1]), buff=0,
                       color=NARANJA, stroke_width=4, max_tip_length_to_length_ratio=0.25)
            fv_lbl = M(r"-v_t", font_size=24, color=NARANJA).next_to(fv.get_end(), UP, buff=0.05)
            self.play(GrowArrow(fv), FadeIn(fv_lbl))
            self.pausa(1.0)
            self.play(FadeOut(fv), FadeOut(fv_lbl))

        self.play(k_t.animate.set_value(N_PASOS), run_time=5, rate_func=rate_functions.ease_in_out_sine)
        self.pausa(1.0)

        jf = {n: J_val(trays[n][20]) for n, *_ in OPT}
        cierre = nota(f"After 20 steps: SGD J ≈ {jf['SGD']:.2f},  Momentum J ≈ {jf['Momentum']:.4f}. "
                      "Momentum builds speed along the valley.")
        self.play(FadeOut(n1), FadeIn(cierre))
        self.wait(3.0)


# ======================================================================
# A03 · Residual blocks: a direct path for gradients
# ======================================================================
N2, D2 = 64, 20


def senal_backward(residual, escala=0.5):
    """||δ_{a_l}|| / ||δ_{a_D}|| para l = 0..D en una red tanh, con y sin conexiones residuales.
    Mismos pesos en ambos casos: W_l ~ N(0, escala²/n)."""
    X = np.random.default_rng(3).normal(size=(500, N2))
    r = np.random.default_rng(4)
    a, Ws, Zs = X, [], []
    for _ in range(D2):
        W = r.normal(0, escala / np.sqrt(N2), (N2, N2))
        z = a @ W.T
        f = np.tanh(z)
        a = a + f if residual else f
        Ws.append(W); Zs.append(z)
    d = np.random.default_rng(5).normal(size=a.shape)
    normas = [np.linalg.norm(d)]
    for l in range(D2 - 1, -1, -1):
        df = (d * (1 - np.tanh(Zs[l]) ** 2)) @ Ws[l]
        d = d + df if residual else df
        normas.append(np.linalg.norm(d))
    normas = np.array(normas[::-1])
    return normas / normas[-1]


class Caja(VGroup):
    def __init__(self, contenido, ancho, fill, borde, alto=0.8):
        super().__init__()
        self.caja = RoundedRectangle(width=ancho, height=alto, corner_radius=0.1, fill_color=fill,
                                     fill_opacity=1, stroke_color=borde, stroke_width=2)
        self.add(self.caja, contenido.move_to(self.caja))


class A03_ResidualGradient(EscenaBase):
    def construct(self):
        plano = senal_backward(False)
        resid = senal_backward(True)

        tit = titulo("Residual blocks: a direct path for gradients")

        # ---------------- bloque residual
        Y = 0.35
        x_in = M("a_{l-1}", font_size=36).move_to([-5.6, Y, 0])
        cW = Caja(T("Affine", font_size=22, weight=BOLD), 1.6, AZ_FILL, AZ_BORDE).move_to([-3.2, Y, 0])
        cP = Caja(M(r"\phi", font_size=36), 1.2, VE_FILL, VE_BORDE).move_to([-1.3, Y, 0])
        suma = VGroup(Circle(radius=0.3, color=NAVY, stroke_width=2.5, fill_color=WHITE, fill_opacity=1),
                      M("+", font_size=36)).move_to([0.6, Y, 0])
        x_out = M("a_l", font_size=36).move_to([2.6, Y, 0])
        fl = VGroup(
            Arrow(x_in.get_right(), cW.get_left(), buff=0.1, color=GRIS_EJE, stroke_width=2.5),
            Arrow(cW.get_right(), cP.get_left(), buff=0.05, color=GRIS_EJE, stroke_width=2.5),
            Arrow(cP.get_right(), suma.get_left(), buff=0.05, color=GRIS_EJE, stroke_width=2.5),
            Arrow(suma.get_right(), x_out.get_left(), buff=0.1, color=GRIS_EJE, stroke_width=2.5),
        )
        p_a = x_in.get_right() + RIGHT * 0.25
        atajo = VMobject(stroke_color=GRIS_EJE, stroke_width=2.5).set_points_as_corners(
            [p_a, p_a + UP * 1.35, suma.get_top() + UP * 1.0, suma.get_top()])
        atajo_tip = Arrow(suma.get_top() + UP * 0.45, suma.get_top(), buff=0.0, color=GRIS_EJE, stroke_width=2.5,
                          max_tip_length_to_length_ratio=0.5)
        lbl_id = T("identity", font_size=18, color=GRIS_TXT).next_to(atajo, UP, buff=0.08)
        br = Brace(VGroup(cW, cP), DOWN, buff=0.15, color=GRIS_EJE)
        lbl_F = M(r"F(a_{l-1})", font_size=28, color=GRIS_TXT).next_to(br, DOWN, buff=0.1)
        eq = M(r"a_l = a_{l-1} + F(a_{l-1})", font_size=32).move_to([4.6, Y + 1.2, 0])
        bloque = VGroup(x_in, cW, cP, suma, x_out, fl, atajo, atajo_tip, lbl_id, br, lbl_F)

        self.play(FadeIn(tit), run_time=0.6)
        self.play(LaggedStart(FadeIn(x_in), FadeIn(cW), FadeIn(cP), FadeIn(suma), FadeIn(x_out), lag_ratio=0.15),
                  *[GrowArrow(f) for f in fl], run_time=1.4)
        self.play(Create(atajo), GrowArrow(atajo_tip), FadeIn(lbl_id), FadeIn(br), FadeIn(lbl_F), Write(eq),
                  run_time=1.2)

        # ---------------- forward: dos caminos
        p1 = Dot(x_in.get_right(), radius=0.09, color=AZUL)
        p2 = p1.copy()
        camino_F = VMobject().set_points_as_corners([x_in.get_right(), cW.get_center(), cP.get_center(),
                                                     suma.get_center()])
        camino_I = VMobject().set_points_as_corners([x_in.get_right(), p_a, p_a + UP * 1.35,
                                                     suma.get_top() + UP * 1.0, suma.get_center()])
        n_f = nota("Forward: the input goes through F and also skips around it.")
        self.play(FadeIn(n_f), FadeIn(p1), FadeIn(p2))
        self.play(MoveAlongPath(p1, camino_F), MoveAlongPath(p2, camino_I), run_time=2.0, rate_func=linear)
        self.play(FadeOut(p2), p1.animate.move_to(x_out.get_left()), run_time=0.5)
        self.play(FadeOut(p1))
        self.pausa(1.0)

        # ---------------- backward: el gradiente se divide
        eq_b = M(r"\delta_{a_{l-1}}=\left(\frac{\partial F}{\partial a_{l-1}}\right)^{\!\top}\delta_{a_l}"
                 r"+\ \delta_{a_l}", font_size=30, color=NARANJA).move_to([4.4, Y - 1.25, 0])
        preg = nota("Question: if F is a poor path for gradients (small ∂F/∂a), what still reaches the block input?")
        preg[0:9].set_color(NARANJA)
        self.play(FadeOut(n_f), FadeIn(preg))
        self.pausa(1.5)

        g1 = Dot(x_out.get_left(), radius=0.11, color=NARANJA)
        self.play(FadeIn(g1, scale=0.5))
        self.play(g1.animate.move_to(suma.get_center()), run_time=0.6)
        g_F, g_I = g1.copy(), g1.copy()
        self.remove(g1)
        self.add(g_F, g_I)
        inv_F = VMobject().set_points_as_corners([suma.get_center(), cP.get_center(), cW.get_center(),
                                                  x_in.get_right()])
        inv_I = VMobject().set_points_as_corners([suma.get_center(), suma.get_top() + UP * 1.0,
                                                  p_a + UP * 1.35, p_a, x_in.get_right()])
        self.play(MoveAlongPath(g_F, inv_F), g_F.animate.scale(0.35).set_opacity(0.45),
                  MoveAlongPath(g_I, inv_I), run_time=2.2, rate_func=linear)
        lbl_gF = M(r"\times\,(\partial F/\partial a)^{\top}", font_size=24, color=NARANJA).next_to(cW, UP, buff=0.12)
        lbl_gI = M(r"\times\,I", font_size=26, color=NARANJA).next_to(lbl_id, RIGHT, buff=0.15)
        self.play(Write(eq_b), FadeIn(lbl_gF), FadeIn(lbl_gI))
        n_b = nota("The identity path passes δ unchanged, even if the path through F shrinks it.")
        self.play(FadeOut(preg), FadeIn(n_b))
        self.pausa(2.0)

        # ---------------- red plana vs residual
        self.play(FadeOut(VGroup(bloque, eq, eq_b, lbl_gF, lbl_gI, g_F, g_I, n_b)), run_time=0.8)
        grupo_e, ax = ejes_log([0, 20, 5], -6, 1, 7.8, 4.3, r"\text{layer } l",
                               r"\lVert\delta_{a_l}\rVert\,/\,\lVert\delta_{a_{20}}\rVert")
        grupo_e.move_to([-1.9, -0.2, 0])
        sub = T("20 tanh layers, width 64, the same random weights in both networks",
                font_size=19, color=GRIS_TXT).next_to(tit, DOWN, buff=0.2, aligned_edge=LEFT)
        ley = VGroup(
            VGroup(Square(0.25, fill_color=AZUL, fill_opacity=1, stroke_width=0),
                   M(r"\text{plain: } a_l=\phi(z_l)", font_size=26, color=AZUL)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.25, fill_color=NARANJA, fill_opacity=1, stroke_width=0),
                   M(r"\text{residual: } a_l=a_{l-1}+\phi(z_l)", font_size=26, color=NARANJA)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([4.95, 0.9, 0])
        self.play(FadeIn(sub), Create(grupo_e), FadeIn(ley), run_time=1.2)

        ancho = ax.x_axis.unit_size * 0.38
        barras_p, barras_r = VGroup(), VGroup()
        for l in range(D2, -1, -1):
            for vals, col, dx, grupo in ((plano, AZUL, -0.2, barras_p), (resid, NARANJA, 0.2, barras_r)):
                y = np.log10(max(vals[l], 1e-6))
                base, tope = ax.c2p(l + dx, 0), ax.c2p(l + dx, y + 6)
                b = Rectangle(width=ancho, height=max(tope[1] - base[1], 0.01), fill_color=col, fill_opacity=0.9,
                              stroke_width=0).move_to((base + tope) / 2)
                grupo.add(b)
        n_c = nota("Read from right to left: the backward signal travels from layer 20 to layer 0.")
        self.play(FadeIn(n_c))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in barras_p], lag_ratio=0.08), run_time=3.0)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in barras_r], lag_ratio=0.08), run_time=3.0)
        self.pausa(1.0)
        e = int(np.floor(np.log10(plano[0])))
        sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
        cierre = nota(f"Signal reaching layer 0, relative to the output:  plain ≈ {plano[0] / 10 ** e:.0f}·10{sup},"
                      f"  residual ≈ {resid[0]:.1f}.")
        self.play(FadeOut(n_c), FadeIn(cierre))
        self.wait(3.0)
