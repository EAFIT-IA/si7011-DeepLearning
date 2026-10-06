"""
E11_PerillasGradiente · SI7011
Cada parámetro es una perilla. El gradiente indica hacia dónde girar cada una
(y qué tanto importa) para bajar el costo.

Modelo (7 perillas):
    f(x) = v₁ tanh(w₁x + b₁) + v₂ tanh(w₂x + b₂) + c,   costo MSE sobre 14 puntos.
Las flechas naranjas son −∂J/∂θ (J = costo del dataset) calculadas en vivo; su arco es proporcional a
|∂𝓛/∂θ| (en escala del gradiente inicial más grande). Los pasos son descenso de
gradiente real con η = 0.2.

Convención: girar a la derecha (horario) aumenta θ.

Render:
    S01_VIDEO=1 manim -qh escena_perillas.py E11_PerillasGradiente
    manim -qh escena_perillas.py E11_PerillasGradiente      # con pausas (manim-slides)
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo (como E10)
FONDO = WHITE
NAVY = "#14213d"
GRIS_TXT = "#5b6b80"
GRIS_EJE = "#5b6577"
AZ_FILL, AZ_BORDE = "#e4eefc", "#7aa6dc"
VE_FILL, VE_BORDE = "#e3f5ea", "#83c9a0"
AZUL = "#1f6fb4"
NARANJA = "#d9631e"
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


# ---------------------------------------------------------------- modelo y datos
NOMBRES = ["w_1", "b_1", "w_2", "b_2", "v_1", "v_2", "c"]
ETA = 0.2
N_PASOS = 120


def modelo(p, x):
    w1, b1, w2, b2, v1, v2, c = p
    return v1 * np.tanh(w1 * x + b1) + v2 * np.tanh(w2 * x + b2) + c


def costo_y_gradiente(p, x, y):
    w1, b1, w2, b2, v1, v2, c = p
    t1, t2 = np.tanh(w1 * x + b1), np.tanh(w2 * x + b2)
    r = v1 * t1 + v2 * t2 + c - y
    d = 2 * r / len(x)
    s1, s2 = d * v1 * (1 - t1 ** 2), d * v2 * (1 - t2 ** 2)
    g = np.array([np.sum(s1 * x), np.sum(s1), np.sum(s2 * x), np.sum(s2),
                  np.sum(d * t1), np.sum(d * t2), np.sum(d)])
    return float(np.mean(r ** 2)), g


def datos():
    maestro = np.array([2.0, -1.0, -1.5, -1.2, 1.2, 0.9, -0.3])
    x = np.linspace(-2.5, 2.5, 14)
    y = modelo(maestro, x) + np.random.default_rng(1).normal(0, 0.06, len(x))
    return x, y


P0 = np.array([0.5, 0.3, -0.4, 0.2, 0.4, -0.3, 0.1])

# ---------------------------------------------------------------- perillas
RANGO = 2.5                # θ ∈ [−2.5, 2.5] ocupa 270° de la perilla
R_PERILLA = 0.46
R_ANILLO = 0.58
R_FLECHA = 0.74
ARCO_MAX = 1.9             # rad para el gradiente inicial más grande


def angulo(theta):
    """θ = 0 apunta arriba; θ > 0 gira en sentido horario."""
    return PI / 2 - np.clip(theta, -RANGO, RANGO) * (3 * PI / 4) / RANGO


class E11_PerillasGradiente(EscenaBase):
    def construct(self):
        X, Y = datos()
        tray = [P0.copy()]
        for _ in range(N_PASOS):
            tray.append(tray[-1] - ETA * costo_y_gradiente(tray[-1], X, Y)[1])
        tray = np.array(tray)
        costos = np.array([costo_y_gradiente(p, X, Y)[0] for p in tray])
        g_max0 = np.abs(costo_y_gradiente(P0, X, Y)[1]).max()

        t_paso = ValueTracker(0.0)   # posición (fraccional) en la trayectoria
        mueve = ValueTracker(0.0)    # giro manual de una perilla (demostración)
        K_DEMO = 6                   # la perilla c

        def params():
            t = np.clip(t_paso.get_value(), 0, N_PASOS)
            i = int(min(np.floor(t), N_PASOS - 1))
            a = t - i
            p = (1 - a) * tray[i] + a * tray[i + 1]
            p = p.copy()
            p[K_DEMO] += mueve.get_value()
            return p

        # ----- encabezado
        titulo = Text("Each parameter is a knob", font=FUENTE, weight=BOLD, color=NAVY,
                      font_size=30).to_corner(UL, buff=0.35)
        formula = MathTex(r"f(x) = v_1 \tanh(w_1 x + b_1) + v_2 \tanh(w_2 x + b_2) + c",
                          color=NAVY, font_size=30).to_corner(UR, buff=0.4)

        def subtitulo(*piezas):
            g = VGroup()
            for tipo, s in piezas:
                g.add(Text(s, font=FUENTE, font_size=21, color=GRIS_TXT) if tipo == "t"
                      else MathTex(s, font_size=30, color=GRIS_TXT))
            g.arrange(RIGHT, buff=0.12)
            return g.next_to(titulo, DOWN, buff=0.22, aligned_edge=LEFT)

        # ----- ajuste del modelo
        ejes_f = Axes(x_range=[-2.8, 2.8, 1], y_range=[-2.4, 2.0, 1], x_length=5.6, y_length=2.85,
                      tips=False, axis_config={"color": GRIS_EJE, "stroke_width": 1.6,
                                               "include_ticks": False}).move_to([-3.45, 0.72, 0])
        rot_f = Text("data and model", font=FUENTE, font_size=19, color=GRIS_TXT)
        rot_f.next_to(ejes_f, UP, buff=0.06).align_to(ejes_f, LEFT)
        puntos = VGroup(*[Dot(ejes_f.c2p(a, b), radius=0.055, color=NAVY) for a, b in zip(X, Y)])
        xs = np.linspace(-2.8, 2.8, 200)

        def curva():
            yy = np.clip(modelo(params(), xs), -2.4, 2.0)
            return VMobject(stroke_color=AZUL, stroke_width=4.5).set_points_smoothly(
                [ejes_f.c2p(a, b) for a, b in zip(xs, yy)])

        curva_m = always_redraw(curva)

        # ----- historia del costo
        ejes_c = Axes(x_range=[0, N_PASOS, 20], y_range=[0, 2.0, 0.5], x_length=5.0, y_length=2.85,
                      tips=False, axis_config={"color": GRIS_EJE, "stroke_width": 1.6,
                                               "include_ticks": False}).move_to([3.75, 0.72, 0])
        rot_cx = Text("step", font=FUENTE, font_size=18, color=GRIS_TXT).next_to(ejes_c.x_axis, DOWN, buff=0.1)
        rot_cx.align_to(ejes_c, RIGHT)

        def costo_actual():
            return costo_y_gradiente(params(), X, Y)[0]

        def historia():
            t = t_paso.get_value()
            n = int(np.floor(t))
            pts = [ejes_c.c2p(i, costos[i]) for i in range(n + 1)]
            if t > n:
                pts.append(ejes_c.c2p(t, costo_actual()))
            linea = VMobject(stroke_color=NARANJA, stroke_width=3.5)
            if len(pts) > 1:
                linea.set_points_as_corners(pts)
            punto = Dot(ejes_c.c2p(t, min(costo_actual(), 2.0)), radius=0.08, color=NARANJA)
            return VGroup(linea, punto)

        hist = always_redraw(historia)

        lectura_L = MathTex(r"J =", color=NAVY, font_size=34)
        lectura_n = DecimalNumber(costos[0], num_decimal_places=3, color=NAVY, font_size=34)
        lectura = VGroup(lectura_L, lectura_n).arrange(RIGHT, buff=0.15)
        lectura.next_to(ejes_c, UP, buff=0.06).align_to(ejes_c, LEFT)
        lectura_n.add_updater(lambda m: m.set_value(costo_actual()))

        # ----- perillas
        xs_p = [-5.75, -4.15, -2.55, -0.95, 1.95, 3.55, 5.15]
        Y_P = -2.15

        def perilla_estatica(k):
            c = np.array([xs_p[k], Y_P, 0])
            oculta = k < 4
            anillo = Arc(radius=R_ANILLO, start_angle=angulo(-RANGO), angle=-3 * PI / 2,
                         arc_center=c, color="#c4cbd5", stroke_width=2)
            marcas = VGroup()
            for th in np.linspace(-RANGO, RANGO, 11):
                u = np.array([np.cos(angulo(th)), np.sin(angulo(th)), 0])
                largo = 0.13 if abs(th) < 1e-9 else 0.07
                marcas.add(Line(c + u * (R_ANILLO - largo / 2), c + u * (R_ANILLO + largo / 2),
                                color=GRIS_EJE if abs(th) < 1e-9 else "#aab3c0", stroke_width=2))
            cuerpo = Circle(radius=R_PERILLA, fill_color=AZ_FILL if oculta else VE_FILL, fill_opacity=1,
                            stroke_color=AZ_BORDE if oculta else VE_BORDE, stroke_width=2.5).move_to(c)
            nombre = MathTex(NOMBRES[k], color=NAVY, font_size=34).move_to(c + UP * (R_FLECHA + 0.3))
            return VGroup(anillo, marcas, cuerpo, nombre)

        def perilla_dinamica(k):
            c = np.array([xs_p[k], Y_P, 0])

            def dib():
                p = params()
                th = p[k]
                u = np.array([np.cos(angulo(th)), np.sin(angulo(th)), 0])
                aguja = Line(c + u * 0.08, c + u * (R_PERILLA - 0.06), color=NAVY, stroke_width=5)
                eje = Dot(c, radius=0.05, color=NAVY)
                valor = DecimalNumber(th, num_decimal_places=2, include_sign=True, color=NAVY,
                                      font_size=24).move_to(c + DOWN * (R_FLECHA + 0.22))
                return VGroup(aguja, eje, valor)
            return always_redraw(dib)

        def flecha_gradiente(k):
            c = np.array([xs_p[k], Y_P, 0])

            def dib():
                p = params()
                g = costo_y_gradiente(p, X, Y)[1][k]
                barrido = np.sign(g) * min(abs(g) / g_max0, 1.15) * ARCO_MAX  # −g: g>0 → antihorario
                if abs(barrido) < 0.07:
                    return VGroup()
                arco = Arc(radius=R_FLECHA, start_angle=angulo(p[k]), angle=barrido, arc_center=c,
                           color=NARANJA, stroke_width=5)
                arco.add_tip(tip_length=0.16, tip_width=0.16)
                return arco
            return always_redraw(dib)

        estaticas = VGroup(*[perilla_estatica(k) for k in range(7)])
        dinamicas = VGroup(*[perilla_dinamica(k) for k in range(7)])
        flechas = VGroup(*[flecha_gradiente(k) for k in range(7)])

        def grupo(desde, hasta, texto):
            a = estaticas[desde].get_left() + DOWN * 0
            b = estaticas[hasta].get_right()
            y = Y_P - R_FLECHA - 0.55
            linea = Line([a[0] + 0.1, y, 0], [b[0] - 0.1, y, 0], color="#c4cbd5", stroke_width=2)
            rot = Text(texto, font=FUENTE, font_size=18, color=GRIS_TXT).next_to(linea, DOWN, buff=0.08)
            return VGroup(linea, rot)

        grupos = VGroup(grupo(0, 3, "hidden layer"), grupo(4, 6, "output"))

        def destello(k):
            anillo = Circle(radius=R_FLECHA + 0.1, color=NARANJA, stroke_width=3).move_to([xs_p[k], Y_P, 0])
            return Succession(Create(anillo, run_time=0.5), FadeOut(anillo, run_time=0.5))

        # =============================================================
        # 1 · Presentación
        # =============================================================
        sub = subtitulo(("t", "Each knob controls one parameter; the cost"), ("m", r"J(\theta)"),
                        ("t", "measures how badly the model fits."))
        self.play(FadeIn(titulo), FadeIn(formula), run_time=0.8)
        self.play(FadeIn(sub))
        self.play(LaggedStart(*[FadeIn(VGroup(e, d)) for e, d in zip(estaticas, dinamicas)], lag_ratio=0.1),
                  FadeIn(grupos), run_time=1.6)
        self.play(Create(ejes_f), FadeIn(rot_f), FadeIn(puntos), Create(ejes_c), FadeIn(rot_cx), run_time=1.0)
        self.play(Create(curva_m), FadeIn(hist), FadeIn(lectura), run_time=1.0)
        self.pausa(1.0)

        # =============================================================
        # 2 · Girar una perilla a mano
        # =============================================================
        sub2 = subtitulo(("t", "Turning"), ("m", "c"), ("t", "right raises the cost; turning it left lowers it."))
        marco = Circle(radius=R_FLECHA + 0.12, color=NARANJA, stroke_width=3).move_to([xs_p[K_DEMO], Y_P, 0])
        self.play(FadeOut(sub), FadeIn(sub2), Create(marco))
        self.play(mueve.animate.set_value(0.35), run_time=1.6)
        self.wait(0.6)
        self.play(mueve.animate.set_value(-0.35), run_time=2.4)
        self.wait(0.6)
        self.play(mueve.animate.set_value(0.0), run_time=1.2)
        self.pausa(1.0)

        # =============================================================
        # 3 · El gradiente: dirección y tamaño para cada perilla
        # =============================================================
        sub3 = subtitulo(("t", "Each arrow is"), ("m", r"-\partial J/\partial\theta_k"),
                         ("t", ": which way to turn, and its size says how much it matters."))
        self.play(FadeOut(sub2), FadeIn(sub3), FadeOut(marco))
        self.add(flechas[K_DEMO])
        self.play(destello(K_DEMO))
        self.pausa(0.8)
        self.add(*[flechas[k] for k in range(7) if k != K_DEMO])
        self.play(*[destello(k) for k in range(7) if k != K_DEMO],
                  run_time=1.0)
        self.pausa(1.5)

        # =============================================================
        # 4 · Pasos de descenso de gradiente
        # =============================================================
        sub4 = subtitulo(("t", "One step: every knob turns at once,"),
                         ("m", r"\theta \leftarrow \theta - \eta\,\nabla_\theta J"),
                         ("t", f"(η = {ETA})."))
        self.play(FadeOut(sub3), FadeIn(sub4))
        for k in range(3):
            self.play(t_paso.animate.set_value(k + 1), run_time=1.6, rate_func=smooth)
            self.pausa(0.8)

        sub5 = subtitulo(("t", "Repeat: the arrows shrink as we approach the minimum."))
        self.play(FadeOut(sub4), FadeIn(sub5))
        self.play(t_paso.animate.set_value(N_PASOS), run_time=10, rate_func=rate_functions.ease_in_out_sine)
        self.pausa(1.0)

        sub6 = subtitulo(("t", "At the minimum"), ("m", r"\nabla_\theta J \approx 0"),
                         ("t", ": there is nowhere left to turn."))
        self.play(FadeOut(sub5), FadeIn(sub6))
        self.wait(3.0)
