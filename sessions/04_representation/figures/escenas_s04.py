"""
Escenas Manim · SI7011 · Sesión 4: Representation Learning

    A01_LatentSpace         Código h ∈ ℝ² de un autoencoder de Fashion-MNIST en 9 momentos
                            del entrenamiento: la nube se separa en grupos sin etiquetas.
                            Datos: s04_a01_latent_snapshots.npz (make_s04_a01_latent_snapshots.py).
    A02_DenoisingManifold   Puntos sobre una curva, ruido, y el mejor denoiser posible
                            E[x | x̃] que los devuelve hacia los datos; luego el campo
                            d(g(x̃)) − x̃ para dos niveles de ruido.
    A03_Masking             Imagen → 16 patches → 75 % enmascarado → el encoder ve 4 tokens →
                            el decoder recibe 16 (4 + 12 [M]) → pérdida solo en los enmascarados.
    A04_ContrastiveSphere   6 imágenes × 2 vistas sobre el círculo unitario; descenso de
                            gradiente exacto de InfoNCE; después, el peso de cada negativo
                            cuando la temperatura baja de 1.0 a 0.1.

Todos los números se calculan con numpy al construir la escena (semillas fijas), salvo A01,
que lee el entrenamiento real guardado en el .npz.
Notación del curso: g encoder, d decoder, h = g(x), x̃ entrada corrompida, u = p(h),
s_ij similitud, τ temperatura, L_i pérdida por ejemplo.

Render:
    S01_VIDEO=1 manim -qh escenas_s04.py A02_DenoisingManifold   # video lineal
    manim -qh escenas_s04.py A02_DenoisingManifold               # con pausas (manim-slides)
Poster (último cuadro): añadir -s.
"""
import os
from pathlib import Path

import numpy as np
from manim import *

AQUI = Path(__file__).resolve().parent

# ---------------------------------------------------------------- estilo (como S02, S03)
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


def caja(ancho, alto, fill=AZ_FILL, borde=AZUL, r=0.14):
    return RoundedRectangle(corner_radius=r, width=ancho, height=alto, stroke_color=borde,
                            stroke_width=2, fill_color=fill, fill_opacity=1)


def pixel_art(arr, alto):
    m = ImageMobject(arr)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.set(height=alto)


# ======================================================================
# A01 · The latent space organizes itself
# ======================================================================
CLASES_FMNIST = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
PALETA_10 = ["#1f6fb4", "#d9631e", "#2e8b57", "#c0392b", "#7b4fa0",
             "#8c564b", "#d4579e", "#5b6577", "#b8a019", "#17a2b8"]


class A01_LatentSpace(EscenaBase):
    def construct(self):
        datos = np.load(AQUI / "s04_a01_latent_snapshots.npz")
        codigos, pasos, mse = datos["codigos"], datos["pasos"], datos["mse"]
        y, probe = datos["etiquetas"], float(datos["probe"])

        # cada foto se centra y escala para llenar el panel: importa la forma, no la escala
        CENTRO = np.array([-2.3, -0.45])
        RADIO = 2.85

        def a_pantalla(c):
            c = c - np.median(c, axis=0)
            r = np.percentile(np.linalg.norm(c, axis=1), 97)
            c = c / r * RADIO
            c = np.clip(c, [-3.5, -2.95], [3.5, 2.95])          # los atípicos quedan en el borde del marco
            return np.c_[c + CENTRO, np.zeros(len(c))]

        P = [a_pantalla(c) for c in codigos]

        tit = titulo("The latent space organizes itself")
        self.add(tit)
        marco = RoundedRectangle(corner_radius=0.15, width=7.4, height=6.25, stroke_color=GRIS_CLARO,
                                 stroke_width=1.5).move_to([*CENTRO, 0])
        ejes = VGroup(M(r"h_1", font_size=26, color=GRIS_TXT).next_to(marco, DOWN, buff=0.08),
                      M(r"h_2", font_size=26, color=GRIS_TXT).next_to(marco, LEFT, buff=0.1))
        arq = VGroup(M(r"784 \to 256 \to 2 \to 256 \to 784", font_size=28),
                     M(r"\mathbf h = g(\mathbf x) \in \mathbb R^{2}", font_size=30),
                     T("1 000 test images, one dot each", font_size=19, color=GRIS_TXT)
                     ).arrange(DOWN, buff=0.22, aligned_edge=LEFT).move_to([4.15, 2.0, 0])

        puntos = VGroup(*[Dot(p, radius=0.03, color=GRIS_EJE, fill_opacity=0.75) for p in P[0]])
        self.play(Create(marco), FadeIn(ejes), FadeIn(arq))
        self.play(FadeIn(puntos, lag_ratio=0.002), run_time=1.2)

        paso_t = T("training step", font_size=22, color=GRIS_TXT)
        paso_n = Integer(0, color=NAVY, font_size=40, group_with_commas=False)
        mse_t = T("test reconstruction MSE", font_size=22, color=GRIS_TXT)
        mse_n = DecimalNumber(mse[0], num_decimal_places=4, color=NAVY, font_size=40)
        panel = VGroup(VGroup(paso_t, paso_n).arrange(DOWN, buff=0.12, aligned_edge=LEFT),
                       VGroup(mse_t, mse_n).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
                       ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([4.0, -0.75, 0], aligned_edge=LEFT)
        panel.align_to(arq, LEFT)
        sin_y = T("no labels: only  L = ‖x − x̂‖²", font_size=21, color=TEAL).next_to(panel, DOWN, buff=0.45)
        sin_y.align_to(arq, LEFT)
        self.play(FadeIn(panel), FadeIn(sin_y))
        self.pausa()

        for k in range(1, len(P)):
            rt = 1.5 if k < 4 else 1.0
            self.play(*[d.animate.move_to(p) for d, p in zip(puntos, P[k])],
                      ChangeDecimalToValue(mse_n, mse[k]),
                      ChangeDecimalToValue(paso_n, int(pasos[k])),
                      run_time=rt, rate_func=smooth)
            if k == 3:
                self.pausa()
        self.pausa()

        # solo ahora se colorea por clase
        self.play(*[d.animate.set_color(PALETA_10[c]) for d, c in zip(puntos, y)], run_time=1.6)
        leyenda = VGroup(*[VGroup(Dot(radius=0.07, color=PALETA_10[i]), T(n, font_size=17))
                           .arrange(RIGHT, buff=0.12) for i, n in enumerate(CLASES_FMNIST)])
        leyenda.arrange_in_grid(rows=5, cols=2, buff=(0.35, 0.14), col_alignments="ll", flow_order="dr")
        leyenda.move_to([4.15, -0.55, 0]).align_to(arq, LEFT)
        self.play(FadeOut(panel), FadeOut(sin_y), FadeIn(leyenda))
        clave = VGroup(T("Colors added only now:", font_size=22, color=NARANJA),
                       T("labels were never used.", font_size=22, color=NARANJA)
                       ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
        clave.next_to(leyenda, DOWN, buff=0.35).align_to(arq, LEFT)
        self.play(FadeIn(clave))
        fin = VGroup(T(f"Linear probe on h (k = 2): {probe:.0%}", font_size=19, color=GRIS_TXT),
                     T("chance: 10 %", font_size=19, color=GRIS_TXT)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        fin.next_to(clave, DOWN, buff=0.3).align_to(arq, LEFT)
        self.play(FadeIn(fin))
        self.pausa(2.0)


# ======================================================================
# A02 · Denoising pulls points back to the data
# ======================================================================
def curva(t):
    return np.c_[1.7 * t, 1.15 * np.sin(1.25 * t) - 0.35]


def mejor_denoiser(xt, sigma, ref):
    """E[x | x̃] cuando los datos están uniformes sobre la curva y x̃ = x + N(0, σ²I)."""
    d2 = ((xt[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    w = np.exp(-(d2 - d2.min(1, keepdims=True)) / (2 * sigma ** 2))
    w /= w.sum(1, keepdims=True)
    return w @ ref


class A02_DenoisingManifold(EscenaBase):
    def construct(self):
        rng = np.random.default_rng(4)
        ref = curva(np.linspace(-3.1, 3.1, 3000))                    # la curva, densa
        t_lim = (-3.0, 3.0)
        N = 26
        x = curva(np.sort(rng.uniform(*t_lim, N)))                   # datos limpios
        S1, S2 = 0.45, 1.3
        xt = x + rng.normal(0, S1, x.shape)                          # x̃ = x + ε
        xh = mejor_denoiser(xt, S1, ref)                             # d(g(x̃)) ideal

        P3 = lambda a: np.c_[a, np.zeros(len(a))]
        tit = titulo("Denoising pulls points back to the data")
        self.add(tit)

        linea = VMobject(stroke_color=GRIS_CLARO, stroke_width=3).set_points_smoothly(
            [*map(np.array, P3(curva(np.linspace(*t_lim, 80))))])
        limpios = VGroup(*[Dot(p, radius=0.07, color=AZUL) for p in P3(x)])
        lab = VGroup(T("data", font_size=24, color=AZUL), M(r"\mathbf x", font_size=30, color=AZUL)) \
            .arrange(RIGHT, buff=0.15).move_to([4.9, 2.35, 0])
        lab2 = T("lie near a curve", font_size=20, color=GRIS_TXT).next_to(lab, DOWN, buff=0.1)
        self.play(Create(linea), FadeIn(limpios, lag_ratio=0.05), FadeIn(lab), FadeIn(lab2))
        self.pausa()

        # ruido
        ruidosos = VGroup(*[Circle(radius=0.075, color=NARANJA, stroke_width=3).move_to(p) for p in P3(xt)])
        saltos = VGroup(*[DashedLine(a, b, color=NARANJA, stroke_width=1.6, dash_length=0.06)
                          for a, b in zip(P3(x), P3(xt))])
        lab_r = M(r"\tilde{\mathbf x} = \mathbf x + \boldsymbol\varepsilon", font_size=32, color=NARANJA) \
            .move_to([-4.6, 2.35, 0])
        self.play(*[TransformFromCopy(d, c) for d, c in zip(limpios, ruidosos)], Create(saltos),
                  FadeIn(lab_r), run_time=1.4)
        self.play(limpios.animate.set_opacity(0.18), FadeOut(saltos))
        q = nota("The denoiser sees only the orange points. Where should each one go?", font_size=22)
        self.play(FadeIn(q))
        self.pausa()

        # el mejor denoiser posible
        flechas = VGroup(*[Arrow(a, b, buff=0.07, color=TEAL, stroke_width=3.5, max_tip_length_to_length_ratio=0.3,
                                 tip_length=0.16) for a, b in zip(P3(xt), P3(xh))])
        salidas = VGroup(*[Dot(p, radius=0.06, color=TEAL) for p in P3(xh)])
        lab_d = M(r"d\big(g(\tilde{\mathbf x})\big)", font_size=32, color=TEAL).next_to(lab_r, DOWN, buff=0.25)
        lab_d.align_to(lab_r, LEFT)
        self.play(FadeIn(lab_d), FadeOut(q))
        self.play(*[GrowArrow(f) for f in flechas], run_time=1.3)
        self.play(FadeIn(salidas))
        r1 = nota("It cannot recover the exact x, only move toward where the data are.", font_size=22)
        self.play(FadeIn(r1))
        self.pausa()

        # el campo d(g(x̃)) − x̃ sobre toda la región
        self.play(FadeOut(VGroup(ruidosos, flechas, salidas, r1)), limpios.animate.set_opacity(1))
        xs, ys = np.arange(-5.95, 6.0, 0.62), np.arange(-2.95, 1.45, 0.58)
        grid = np.array([[a, b] for b in ys for a in xs])

        def campo(sigma, color):
            dest = mejor_denoiser(grid, sigma, ref)
            v = dest - grid
            largo = np.linalg.norm(v, axis=1, keepdims=True)
            v = v / np.maximum(largo, 1e-9) * np.minimum(largo, 0.5)          # recorte para que se lea
            return VGroup(*[Arrow([*a, 0], [*(a + b), 0], buff=0, color=color, stroke_width=2.2,
                                  tip_length=0.1, max_tip_length_to_length_ratio=0.45)
                            for a, b, l in zip(grid, v, largo[:, 0]) if l > 0.04])

        c1 = campo(S1, TEAL)
        lab_c = M(r"d\big(g(\tilde{\mathbf x})\big) - \tilde{\mathbf x}", font_size=32, color=TEAL)
        lab_c.move_to(lab_r, aligned_edge=LEFT)
        s_lab = M(r"\text{noise } \sigma = 0.45", font_size=28, color=GRIS_TXT).next_to(lab_c, RIGHT, buff=0.5)
        self.play(FadeOut(lab_d), FadeOut(lab_r), FadeIn(lab_c), FadeIn(s_lab))
        self.play(LaggedStart(*[GrowArrow(f) for f in c1], lag_ratio=0.004), run_time=1.6)
        self.bring_to_front(limpios)
        r2 = nota("A field that points toward the data: the denoiser has learned where the data live.",
                  font_size=21)
        self.play(FadeIn(r2))
        self.pausa()

        c2 = campo(S2, AZUL)
        s_lab2 = M(r"\text{noise } \sigma = 1.3", font_size=28, color=GRIS_TXT).move_to(s_lab, aligned_edge=LEFT)
        r3 = nota("More noise: a coarser picture, everything points to the bulk of the curve.", font_size=21)
        self.play(ReplacementTransform(c1, c2), Transform(s_lab, s_lab2), FadeOut(r2), FadeIn(r3), run_time=1.6)
        self.bring_to_front(limpios)
        self.pausa(2.0)


# ======================================================================
# A03 · Masking in motion
# ======================================================================
LADO_A03, P_A03 = 64, 16
VISIBLES = [(0, 1), (1, 3), (2, 0), (3, 2)]          # (fila, columna) desde 0, como en S04-F04


def imagen_a03():
    from skimage import data, transform
    im = data.chelsea()                                  # la misma imagen de S03-A04
    h, w, _ = im.shape
    c = min(h, w)
    im = im[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2 + 30:(w - c) // 2 + 30 + c]
    im = transform.resize(im, (LADO_A03, LADO_A03), anti_aliasing=True)
    return (im * 255).astype(np.uint8)


def reconstruccion_a03(im):
    """Ilustración de la salida del decoder: inpainting biarmónico desde los 4 patches visibles."""
    from skimage.restoration import inpaint_biharmonic
    mask = np.ones(im.shape[:2], bool)
    for i, j in VISIBLES:
        mask[i * P_A03:(i + 1) * P_A03, j * P_A03:(j + 1) * P_A03] = False
    rec = inpaint_biharmonic(im.astype(float) / 255, mask, channel_axis=-1)
    return (np.clip(rec, 0, 1) * 255).astype(np.uint8)


class A03_Masking(EscenaBase):
    def construct(self):
        im = imagen_a03()
        rec = reconstruccion_a03(im)
        n = LADO_A03 // P_A03
        tit = titulo("Masking in motion")
        self.add(tit)

        ALTO = 3.0
        lado = ALTO / n
        X0, Y0 = -5.0, 0.15
        full = pixel_art(im, ALTO).move_to([X0, Y0, 0])
        ul = full.get_corner(UL)

        def centro(i, j):
            return ul + RIGHT * (j + 0.5) * lado + DOWN * (i + 0.5) * lado

        lineas = VGroup()
        for k in range(1, n):
            lineas.add(Line(ul + RIGHT * k * lado, ul + RIGHT * k * lado + DOWN * ALTO, color=WHITE, stroke_width=3))
            lineas.add(Line(ul + DOWN * k * lado, ul + DOWN * k * lado + RIGHT * ALTO, color=WHITE, stroke_width=3))
        lab = M(r"\mathbf x:\ 16 \text{ patches}", font_size=28).next_to(full, UP, buff=0.2)
        self.play(FadeIn(full), Create(lineas), FadeIn(lab))
        self.pausa()

        # enmascarar 12 de 16
        ocultos = [(i, j) for i in range(n) for j in range(n) if (i, j) not in VISIBLES]
        tapas = VGroup(*[Square(lado, stroke_width=0, fill_color="#9aa3b0", fill_opacity=1).move_to(centro(i, j))
                         for i, j in ocultos])
        lab_m = VGroup(T("mask 75 %", font_size=24, color=NAVY),
                       T("keep 4 of 16", font_size=20, color=GRIS_TXT)).arrange(DOWN, buff=0.08)
        lab_m.next_to(full, DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(t, scale=0.8) for t in tapas], lag_ratio=0.06), FadeIn(lab_m), run_time=1.4)
        self.add(lineas)
        self.pausa()

        # el encoder (grande) recibe solo los 4 visibles
        enc = caja(1.7, 4.0).move_to([-1.55, Y0, 0])
        enc_t = VGroup(M(r"g", font_size=40, color=AZUL), T("encoder", font_size=18, color=AZUL)) \
            .arrange(DOWN, buff=0.1).move_to(enc)
        self.play(FadeIn(enc), FadeIn(enc_t))
        ys_in = np.linspace(1.15, -0.85, 4) + Y0
        copias = Group()
        for (i, j), yy in zip(VISIBLES, ys_in):
            p = pixel_art(im[i * P_A03:(i + 1) * P_A03, j * P_A03:(j + 1) * P_A03], lado).move_to(centro(i, j))
            copias.add(p)
        self.add(copias)
        self.play(*[c.animate.set(height=0.5).move_to([-2.95, yy, 0]) for c, yy in zip(copias, ys_in)],
                  run_time=1.2)
        tok_h = VGroup(*[RoundedRectangle(corner_radius=0.05, width=0.5, height=0.32, stroke_color=AZUL,
                                          stroke_width=1.6, fill_color=AZUL, fill_opacity=0.85).move_to([-0.2, yy, 0])
                         for yy in ys_in])
        self.play(*[c.animate.move_to(enc.get_center()).set(height=0.1).set_opacity(0) for c in copias], run_time=0.8)
        self.remove(copias)
        self.play(LaggedStart(*[FadeIn(t, shift=RIGHT * 0.3) for t in tok_h], lag_ratio=0.15))
        lab_h = M(r"\mathbf h_i", font_size=26, color=AZUL).next_to(tok_h, UP, buff=0.12)
        costo = VGroup(T("The encoder processes", font_size=19, color=GRIS_TXT),
                       T("4 tokens, not 16.", font_size=19, color=GRIS_TXT)).arrange(DOWN, buff=0.06)
        costo.next_to(enc, DOWN, buff=0.2)
        self.play(FadeIn(lab_h), FadeIn(costo))
        self.pausa()

        # el decoder (pequeño) recibe 16: 4 codificados + 12 [M], cada uno en su posición
        ALTO_S = 0.215
        ys_s = Y0 + (np.arange(16)[::-1] - 7.5) * ALTO_S
        X_S = 1.15
        idx_vis = [i * n + j for i, j in VISIBLES]
        slots = VGroup()
        for k in range(16):
            r = Rectangle(width=0.62, height=ALTO_S, stroke_color=WHITE, stroke_width=1.2,
                          fill_color=AZUL if k in idx_vis else "#c9cfd8", fill_opacity=1).move_to([X_S, ys_s[k], 0])
            if k not in idx_vis:
                r = VGroup(r, T("[M]", font_size=11, color=NAVY).move_to(r))
            slots.add(r)
        mvis = VGroup(*[slots[k] for k in idx_vis])
        mmask = VGroup(*[slots[k] for k in range(16) if k not in idx_vis])
        self.play(*[ReplacementTransform(t, slots[k]) for t, k in zip(tok_h, idx_vis)], FadeOut(lab_h), run_time=1.2)
        self.play(LaggedStart(*[FadeIn(s) for s in mmask], lag_ratio=0.05), run_time=1.0)
        lab_s = VGroup(T("4 encoded + 12 mask tokens", font_size=17, color=NAVY),
                       T("each at its position", font_size=17, color=GRIS_TXT)).arrange(DOWN, buff=0.06)
        lab_s.next_to(slots, UP, buff=0.18)
        self.play(FadeIn(lab_s))

        dec = caja(0.95, 1.6, fill=VE_FILL, borde=TEAL).move_to([2.55, Y0, 0])
        dec_t = VGroup(M(r"d", font_size=36, color=TEAL), T("decoder", font_size=14, color=TEAL)) \
            .arrange(DOWN, buff=0.06).move_to(dec)
        f1 = Arrow([X_S + 0.35, Y0, 0], dec.get_left(), buff=0.05, color=GRIS_EJE, stroke_width=3)
        self.play(FadeIn(dec), FadeIn(dec_t), GrowArrow(f1))
        tam = T("small", font_size=17, color=GRIS_TXT).next_to(dec, DOWN, buff=0.12)
        self.play(FadeIn(tam))
        self.pausa()

        # salida: 16 parches predichos; la pérdida solo cuenta los 12 enmascarados
        salida = pixel_art(rec, ALTO).move_to([5.25, Y0, 0])
        ul2 = salida.get_corner(UL)
        f2 = Arrow(dec.get_right(), salida.get_left(), buff=0.08, color=GRIS_EJE, stroke_width=3)
        lineas2 = VGroup()
        for k in range(1, n):
            lineas2.add(Line(ul2 + RIGHT * k * lado, ul2 + RIGHT * k * lado + DOWN * ALTO, color=WHITE, stroke_width=3))
            lineas2.add(Line(ul2 + DOWN * k * lado, ul2 + DOWN * k * lado + RIGHT * ALTO, color=WHITE, stroke_width=3))
        lab_o = M(r"\hat{\mathbf x}", font_size=32).next_to(salida, UP, buff=0.2)
        ilus = T("(illustration: a blurry guess)", font_size=15, color=GRIS_TXT).next_to(salida, DOWN, buff=0.12)
        self.play(GrowArrow(f2), FadeIn(salida), Create(lineas2), FadeIn(lab_o), FadeIn(ilus))
        self.pausa()

        marcos = VGroup(*[Square(lado * 0.86, color=NARANJA, stroke_width=4)
                          .move_to(ul2 + RIGHT * (j + 0.5) * lado + DOWN * (i + 0.5) * lado) for i, j in ocultos])
        formula = M(r"L_i = \frac{1}{|M|}\sum_{m \in M} \lVert \mathbf x_m - \hat{\mathbf x}_m \rVert^2",
                    font_size=34, color=NARANJA).to_edge(DOWN, buff=0.45).shift(RIGHT * 1.6)
        fl = T("loss only on the 12 masked patches", font_size=19, color=NARANJA).next_to(formula, LEFT, buff=0.4)
        self.play(FadeOut(costo), LaggedStart(*[Create(m) for m in marcos], lag_ratio=0.05),
                  FadeIn(formula), FadeIn(fl), run_time=1.6)
        self.pausa()

        # después del preentrenamiento
        tacha = Line(dec.get_corner(DL), dec.get_corner(UR), color=GRIS_EJE, stroke_width=4)
        self.play(VGroup(dec, dec_t).animate.set_opacity(0.3), Create(tacha),
                  enc.animate.set_stroke(width=5), FadeOut(VGroup(formula, fl)))
        fin = nota("After pretraining: keep the encoder g, discard the decoder d.", font_size=22)
        self.play(FadeIn(fin))
        self.pausa(2.0)


# ======================================================================
# A04 · Pull and push on the sphere
# ======================================================================
def infonce_circulo(th, tau, N):
    """InfoNCE para 2N embeddings unitarios u_i = (cos θ_i, sen θ_i); i e i+N son pareja.
    Devuelve J (promedio de L_i) y dJ/dθ (exacto, verificado contra diferencias finitas)."""
    n = 2 * N
    D = th[:, None] - th[None, :]
    S = np.cos(D) / tau                                   # s_ij = u_i·u_j / τ
    np.fill_diagonal(S, -np.inf)
    pos = (np.arange(n) + N) % n
    m = S.max(1, keepdims=True)
    lse = (m + np.log(np.exp(S - m).sum(1, keepdims=True)))[:, 0]
    J = np.mean(-S[np.arange(n), pos] + lse)              # L_i = −s_i,pos + log Σ_{k≠i} exp s_ik
    P = np.exp(S - lse[:, None])
    W = P.copy()
    W[np.arange(n), pos] -= 1
    W /= n
    dS = -np.sin(D) / tau
    np.fill_diagonal(dS, 0)
    return J, (W * dS).sum(1) - (W * dS).sum(0)


COLORES_6 = [AZUL, NARANJA, TEAL, ROJO, "#7b4fa0", "#b8a019"]


class A04_ContrastiveSphere(EscenaBase):
    def construct(self):
        N, TAU, PASOS, LR = 6, 0.2, 200, 0.6
        rng = np.random.default_rng(3)
        th0 = rng.uniform(0, 2 * np.pi, 2 * N)
        tray, Js = [th0.copy()], []
        th = th0.copy()
        for _ in range(PASOS):
            J, g = infonce_circulo(th, TAU, N)
            Js.append(J)
            th = th - LR * g
            tray.append(th.copy())
        Js.append(infonce_circulo(th, TAU, N)[0])
        tray = np.array(tray)

        tit = titulo("Pull and push on the sphere")
        self.add(tit)
        C, R = np.array([-3.3, -0.05, 0]), 2.4
        circ = Circle(radius=R, color=GRIS_CLARO, stroke_width=2.5).move_to(C)
        pt = lambda a: C + R * np.array([np.cos(a), np.sin(a), 0])

        def marcador(k, a):
            col = COLORES_6[k % N]
            if k < N:
                return Dot(pt(a), radius=0.12, color=col)
            return Circle(radius=0.12, color=col, stroke_width=4).move_to(pt(a))

        marcas = VGroup(*[marcador(k, a) for k, a in enumerate(th0)])
        ley = VGroup(VGroup(Dot(radius=0.1, color=NAVY), T("view 1", font_size=19)).arrange(RIGHT, buff=0.12),
                     VGroup(Circle(radius=0.1, color=NAVY, stroke_width=4), T("view 2", font_size=19))
                     .arrange(RIGHT, buff=0.12),
                     T("same color = two views of one image", font_size=19, color=GRIS_TXT)
                     ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to([0.9, 2.1, 0], aligned_edge=LEFT)
        ulab = M(r"\lVert \mathbf u \rVert = 1", font_size=28, color=GRIS_TXT).next_to(circ, DOWN, buff=0.15)
        self.play(Create(circ), FadeIn(ulab), FadeIn(marcas, lag_ratio=0.08), FadeIn(ley))
        self.pausa()

        # fuerzas en el punto 0: tira hacia su pareja, empujan los demás
        a0 = th0[0]
        D = th0 - a0
        S = np.cos(D) / TAU
        S[0] = -np.inf
        p = np.exp(S - S.max())
        p /= p.sum()
        tang = np.array([-np.sin(a0), np.cos(a0), 0])
        lineas = VGroup()
        for k in range(1, 2 * N):
            if k == N:
                lineas.add(Line(pt(a0), pt(th0[k]), color=NARANJA, stroke_width=5))
            else:
                lineas.add(Line(pt(a0), pt(th0[k]), color=GRIS_EJE, stroke_width=1 + 10 * p[k],
                                stroke_opacity=0.25 + 0.75 * min(1, p[k] / p[1:].max())))
        txt_f = VGroup(T("orange: pulled toward its other view", font_size=19, color=NARANJA),
                       T("gray: pushed away by the negatives,", font_size=19, color=GRIS_EJE),
                       T("harder for the closer ones", font_size=19, color=GRIS_EJE)
                       ).arrange(DOWN, buff=0.08, aligned_edge=LEFT).next_to(ley, DOWN, buff=0.5)
        txt_f.align_to(ley, LEFT)
        self.play(Create(lineas), FadeIn(txt_f), marcas[0].animate.scale(1.5))
        self.pausa()

        # descenso de gradiente sobre InfoNCE
        self.play(FadeOut(lineas), FadeOut(txt_f), marcas[0].animate.scale(1 / 1.5))
        formula = M(r"L_i = -\log \frac{e^{s_{i,j(i)}}}{\sum_{k\ne i} e^{s_{ik}}}", font_size=32)
        formula.next_to(ley, DOWN, buff=0.45).align_to(ley, LEFT)
        tau_t = M(rf"\tau = {TAU}", font_size=28, color=GRIS_TXT).next_to(formula, DOWN, buff=0.2)
        tau_t.align_to(ley, LEFT)
        jt = M(r"J =", font_size=30).next_to(tau_t, DOWN, buff=0.3).align_to(ley, LEFT)
        jn = DecimalNumber(Js[0], num_decimal_places=2, color=NAVY, font_size=34).next_to(jt, RIGHT, buff=0.15)
        self.play(FadeIn(formula), FadeIn(tau_t), FadeIn(jt), FadeIn(jn))

        prog = ValueTracker(0)

        def actualizar(mob):
            s = prog.get_value()
            i0 = int(np.floor(s))
            i1 = min(i0 + 1, PASOS)
            f = s - i0
            a = tray[i0] * (1 - f) + tray[i1] * f
            for k, m in enumerate(mob):
                m.move_to(pt(a[k]))

        marcas.add_updater(actualizar)
        jn.add_updater(lambda m: m.set_value(Js[min(int(prog.get_value()), PASOS)]))
        self.play(prog.animate.set_value(PASOS), run_time=6, rate_func=linear)
        marcas.clear_updaters()
        jn.clear_updaters()
        r1 = nota("Views of the same image merge; different images spread around the sphere.", font_size=21)
        self.play(FadeIn(r1))
        self.pausa()

        # temperatura: peso de cada negativo para un ancla fija
        self.play(FadeOut(VGroup(marcas, formula, tau_t, jt, jn, r1, ley, ulab)))
        ancla, pareja = np.pi / 2, np.pi / 2 - 0.3
        negs = np.array([np.pi / 2 + 0.55, np.pi / 2 + 1.0, np.pi / 2 - 1.05, np.pi + 0.9, -0.2,
                         np.pi + 0.2, -0.95, 0.35, np.pi / 2 + 1.6, -np.pi / 2])
        mk_a = Dot(pt(ancla), radius=0.14, color=NAVY)
        mk_p = Circle(radius=0.13, color=NAVY, stroke_width=4).move_to(pt(pareja))
        mk_n = VGroup(*[Dot(pt(a), radius=0.1, color=GRIS_EJE) for a in negs])
        la = T("anchor and its other view", font_size=17, color=NAVY).next_to(mk_a, UP, buff=0.15)
        self.play(FadeIn(mk_a), FadeIn(mk_p), FadeIn(mk_n), FadeIn(la))

        tau = ValueTracker(1.0)

        def pesos():
            s = np.cos(negs - ancla) / tau.get_value()
            w = np.exp(s - s.max())
            return w / w.sum()                                # peso relativo de cada negativo en el empuje

        orden = np.argsort(-np.cos(negs - ancla))             # del más cercano al más lejano
        X0, ANCHO_MAX, ALTO_B = 2.6, 3.4, 0.36
        nombres = VGroup(*[T(f"n{r + 1}", font_size=16, color=GRIS_TXT) for r in range(len(negs))])
        for r, nm in enumerate(nombres):
            nm.move_to([X0 - 0.35, 2.0 - r * (ALTO_B + 0.08), 0])
        etiq_n = VGroup(*[T(f"n{list(orden).index(k) + 1}", font_size=15, color=GRIS_TXT)
                          .move_to(C + (R + 0.32) * np.array([np.cos(a), np.sin(a), 0]))
                          for k, a in enumerate(negs)])

        def barras():
            w = pesos()
            g = VGroup()
            for r, k in enumerate(orden):
                ancho = max(ANCHO_MAX * w[k], 0.02)
                g.add(Rectangle(width=ancho, height=ALTO_B, stroke_width=0, fill_color=GRIS_EJE, fill_opacity=0.85)
                      .move_to([X0 + ancho / 2, 2.0 - r * (ALTO_B + 0.08), 0]))
            return g

        def rayos():
            w = pesos()
            return VGroup(*[Line(pt(ancla), pt(a), color=GRIS_EJE, stroke_width=0.5 + 12 * w[k],
                                 stroke_opacity=0.15 + 0.85 * w[k] / w.max()) for k, a in enumerate(negs)])

        bs = always_redraw(barras)
        ry = always_redraw(rayos)
        tit_b = T("share of the push, nearest negative first", font_size=19, color=NAVY)
        tit_b.move_to([X0 + 1.4, 2.65, 0])
        tval = VGroup(M(r"\tau =", font_size=34), DecimalNumber(1.0, num_decimal_places=2, font_size=34, color=NAVY))
        tval.arrange(RIGHT, buff=0.15).move_to([X0 + 1.2, -2.35, 0])
        tval[1].add_updater(lambda m: m.set_value(tau.get_value()))
        self.add(ry)
        self.play(FadeIn(bs), FadeIn(nombres), FadeIn(etiq_n), FadeIn(tit_b), FadeIn(tval))
        self.pausa()
        self.play(tau.animate.set_value(0.1), run_time=4, rate_func=smooth)
        self.pausa()
        fin = nota("Small τ: the nearest negatives do almost all the pushing. Large τ: every negative pushes a little.",
                   font_size=19)
        aviso = T("Here the embeddings move directly; in SimCLR the gradient flows into g and p.",
                  font_size=16, color=GRIS_TXT).next_to(fin, UP, buff=0.12)
        self.play(FadeIn(fin), FadeIn(aviso))
        self.pausa(2.0)
