"""
Escenas Manim · SI7011 · Sesión 3: Architectures for Vision

    A01_DenseToConv       Denso → localmente conectado → compartido: las conexiones
                          desaparecen y los pesos colapsan a un solo kernel.
    A02_KernelSliding     Un kernel 3×3 se desliza sobre una imagen 6×6 de enteros;
                          productos y suma en la primera posición, mapa 4×4 completo.
    A03_ReceptiveField    Campo receptivo de una unidad, capa por capa, con stride 1
                          (3, 5, 7) y con stride 2 (3, 7, 15).
    A04_PatchesToTokens   Imagen → 16 patches → aplanar → la misma W_E → tokens t_i
                          + posición p_i, y el token [CLS].

Todos los números se calculan con numpy al construir la escena (semillas fijas).
Notación del curso: K tamaño del kernel, s stride, r_l campo receptivo, t_i tokens de
tamaño d, W_E embedding de patches, p_i embedding posicional.

Render:
    S01_VIDEO=1 manim -qh escenas_s03.py A02_KernelSliding      # video lineal
    manim -qh escenas_s03.py A02_KernelSliding                  # con pausas (manim-slides)
Poster (último cuadro): añadir -s.
"""
import os

import numpy as np
from manim import *

# ---------------------------------------------------------------- estilo (como S02)
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


def celda(lado, valor=None, fill=WHITE, opac=1.0, fs=24, color_txt=NAVY, borde=GRIS_CLARO):
    sq = Square(lado, stroke_color=borde, stroke_width=1.5, fill_color=fill, fill_opacity=opac)
    if valor is None:
        return VGroup(sq)
    return VGroup(sq, T(str(valor), font_size=fs, color=color_txt).move_to(sq))


def rejilla(valores, lado, fills=None, fs=24):
    """VGroup de filas × columnas de celdas con números; g[i][j] es la celda (i, j)."""
    filas = VGroup()
    for i, fila in enumerate(valores):
        f = VGroup()
        for j, v in enumerate(fila):
            fill = fills[i][j] if fills is not None else WHITE
            f.add(celda(lado, v, fill=fill, fs=fs))
        f.arrange(RIGHT, buff=0)
        filas.add(f)
    return filas.arrange(DOWN, buff=0)


def gris(v, vmax):
    """Relleno gris claro → oscuro para un valor de píxel; texto siempre legible."""
    return interpolate_color(ManimColor("#ffffff"), ManimColor("#9aa3b0"), float(v) / vmax)


def con_signo(v, vmax):
    base = ManimColor(NARANJA) if v > 0 else ManimColor(AZUL)
    return interpolate_color(ManimColor("#ffffff"), base, min(abs(v) / vmax, 1.0) * 0.75)


# ======================================================================
# A01 · From dense to convolution
# ======================================================================
N_A01 = 8


class A01_DenseToConv(EscenaBase):
    def construct(self):
        tit = titulo("From dense to convolution")
        sub = T("8 inputs → 8 outputs, drawn as 1-D rows", font_size=20, color=GRIS_TXT)
        sub.next_to(tit, DOWN, buff=0.15, aligned_edge=LEFT)

        xs = np.linspace(-5.9, -0.5, N_A01)
        ent = VGroup(*[Dot([x, -2.0, 0], radius=0.11, color=NAVY) for x in xs])
        sal = VGroup(*[Dot([x, 1.4, 0], radius=0.11, color=NARANJA) for x in xs])
        lbl_x = M("x", font_size=34).next_to(ent, LEFT, buff=0.35)
        lbl_y = M("y", font_size=34).next_to(sal, LEFT, buff=0.35)

        def linea(i, j, col=GRIS_CLARO, w=1.6):
            # i: salida, j: entrada
            return Line(sal[i].get_center(), ent[j].get_center(), color=col, stroke_width=w)

        densas = VGroup(*[linea(i, j) for i in range(N_A01) for j in range(N_A01)])

        # panel derecho: conteo de parámetros
        X0 = 2.9
        cab = VGroup(VectorizedPoint(), T("drawn", font_size=20, color=GRIS_TXT),
                     T("32 × 32 image", font_size=20, color=GRIS_TXT))
        filas_txt = [
            ("dense", "8·8 + 8 = 72", "1 049 600"),
            ("local, 3 wide", "8·3 + 8 = 32", "10 240"),
            ("shared kernel", "3 + 1 = 4", "10"),
        ]
        tabla = VGroup()
        for nombre, a, b in filas_txt:
            tabla.add(VGroup(T(nombre, font_size=22), T(a, font_size=22), T(b, font_size=22, weight=BOLD)))
        cols_x = [X0 - 1.25, X0 + 0.85, X0 + 2.75]
        for k, g in enumerate([cab] + list(tabla)):
            for c, m in enumerate(g):
                m.move_to([cols_x[c], 1.6 - 0.75 * k, 0])
        tabla_t = T("parameters (with bias)", font_size=20, color=GRIS_TXT).move_to([X0 + 0.9, 2.35, 0])

        self.add(tit, sub)
        self.play(FadeIn(ent), FadeIn(sal), FadeIn(lbl_x), FadeIn(lbl_y))
        self.play(LaggedStart(*[Create(l) for l in densas], lag_ratio=0.01, run_time=1.6))
        self.play(FadeIn(tabla_t), FadeIn(cab), FadeIn(tabla[0]))
        n1 = nota("Dense: every output sees every input, each with its own weight.")
        self.play(FadeIn(n1))
        self.pausa()

        # --- locally connected: sólo vecinos |i-j| ≤ 1, cada uno con su propio color (pesos propios)
        locales = [(i, j) for i in range(N_A01) for j in range(N_A01) if abs(i - j) <= 1]
        rng = np.random.default_rng(4)
        paleta = [AZUL, TEAL, NARANJA, "#7b4fa0", "#c0392b", "#2e8b57", "#b8860b", "#5b6b80"]
        propios = {}
        for (i, j) in locales:
            propios[(i, j)] = paleta[int(rng.integers(len(paleta)))]
        quitar = VGroup(*[densas[i * N_A01 + j] for i in range(N_A01) for j in range(N_A01) if abs(i - j) > 1])
        quedan = {(i, j): densas[i * N_A01 + j] for (i, j) in locales}
        n2 = nota("Locality: each output sees 3 neighbors, still with its own weights.")
        self.play(FadeOut(quitar), FadeOut(n1), run_time=1.0)
        self.play(*[quedan[k].animate.set_color(propios[k]).set_stroke(width=3) for k in quedan],
                  FadeIn(tabla[1]), FadeIn(n2))
        self.pausa()

        # --- sharing: el color sólo depende del desplazamiento j - i
        por_desp = {-1: AZUL, 0: TEAL, 1: NARANJA}
        kern = VGroup(
            M(r"k_{-1}", font_size=30, color=AZUL), M(r"k_{0}", font_size=30, color=TEAL),
            M(r"k_{+1}", font_size=30, color=NARANJA),
        ).arrange(RIGHT, buff=0.5)
        kern_box = SurroundingRectangle(kern, color=NAVY, buff=0.18, stroke_width=1.5)
        kern_g = VGroup(kern_box, kern, T("one kernel, reused at every position", font_size=18,
                                          color=GRIS_TXT).next_to(kern_box, DOWN, buff=0.12))
        kern_g.move_to([-3.2, 2.55, 0])
        n3 = nota("Sharing: the same 3 weights at every position. That is a convolution.")
        self.play(*[quedan[k].animate.set_color(por_desp[k[1] - k[0]]) for k in quedan],
                  FadeOut(sub), FadeIn(kern_g), FadeIn(tabla[2]), FadeOut(n2))
        self.play(FadeIn(n3), Indicate(tabla[2][2], color=NARANJA, scale_factor=1.3))
        self.pausa(2.0)


# ======================================================================
# A02 · A kernel sliding over an image
# ======================================================================
IMG_A02 = np.array([
    [1, 1, 7, 7, 1, 1],
    [1, 2, 8, 7, 2, 1],
    [0, 1, 7, 8, 1, 0],
    [1, 1, 8, 7, 1, 1],
    [2, 1, 7, 7, 1, 2],
    [1, 1, 8, 8, 1, 1],
])
KER_A02 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])


def correlacion_valida(x, k):
    K = k.shape[0]
    n = x.shape[0] - K + 1
    return np.array([[int((x[i:i + K, j:j + K] * k).sum()) for j in range(n)] for i in range(n)])


class A02_KernelSliding(EscenaBase):
    def construct(self):
        Y = correlacion_valida(IMG_A02, KER_A02)          # 4×4
        L = 0.6
        tit = titulo("A kernel sliding over an image")
        form = M(r"y_{i,j}=\sum_{u=0}^{2}\sum_{v=0}^{2} k_{u,v}\,x_{i+u,\;j+v}", font_size=34)
        form.next_to(tit, DOWN, buff=0.2, aligned_edge=LEFT)

        imgg = rejilla(IMG_A02, L, fills=[[gris(v, 8) for v in f] for f in IMG_A02])
        imgg.move_to([-4.0, -0.75, 0])
        kerg = rejilla(KER_A02, L, fills=[[con_signo(v, 1.4) for v in f] for f in KER_A02])
        kerg.move_to([0.25, 1.15, 0])
        outg = VGroup(*[VGroup(*[celda(L) for _ in range(4)]).arrange(RIGHT, buff=0) for _ in range(4)])
        outg.arrange(DOWN, buff=0).move_to([4.6, -0.75, 0])

        lab_x = T("image x  (6 × 6)", font_size=22).next_to(imgg, UP, buff=0.18)
        lab_k = T("kernel k  (3 × 3)", font_size=22).next_to(kerg, UP, buff=0.18)
        lab_y = T("feature map y  (4 × 4)", font_size=22).next_to(outg, UP, buff=0.18)

        self.add(tit, form)
        self.play(FadeIn(imgg), FadeIn(lab_x))
        self.play(FadeIn(kerg), FadeIn(lab_k))
        self.play(FadeIn(outg), FadeIn(lab_y))
        self.pausa()

        def ventana(i, j):
            sq = Square(3 * L, stroke_color=NARANJA, stroke_width=6)
            return sq.move_to(imgg[i + 1][j + 1].get_center())

        # --- primera posición, en detalle
        win = ventana(0, 0)
        self.play(Create(win))
        prod = IMG_A02[0:3, 0:3] * KER_A02
        prodg = rejilla(prod, L, fills=[[con_signo(v, 9) for v in f] for f in prod], fs=22)
        prodg.move_to([0.25, -1.25, 0])
        lab_p = T("k ⊙ window", font_size=20, color=GRIS_TXT).next_to(prodg, UP, buff=0.12)
        copias = VGroup(*[imgg[u][v].copy() for u in range(3) for v in range(3)])
        self.play(LaggedStart(*[copias[n].animate.move_to(prodg[n // 3][n % 3]) for n in range(9)],
                              lag_ratio=0.06, run_time=1.2), FadeIn(lab_p))
        self.play(FadeOut(copias), FadeIn(prodg))
        suma = M(r"\textstyle\sum = " + str(Y[0, 0]), font_size=34, color=NARANJA).next_to(prodg, DOWN, buff=0.2)
        self.play(Write(suma))
        self.pausa()
        val = T(str(Y[0, 0]), font_size=24).move_to(outg[0][0])
        self.play(outg[0][0][0].animate.set_fill(con_signo(Y[0, 0], 24), 1),
                  ReplacementTransform(suma.copy(), val))
        self.play(FadeOut(prodg), FadeOut(lab_p), FadeOut(suma))
        self.pausa()

        # --- el resto, rápido: el mismo kernel en cada posición
        for n in range(1, 16):
            i, j = divmod(n, 4)
            v = T(str(Y[i, j]), font_size=24).move_to(outg[i][j])
            self.play(win.animate.move_to(ventana(i, j)), run_time=0.28)
            self.play(outg[i][j][0].animate.set_fill(con_signo(Y[i, j], 24), 1), FadeIn(v), run_time=0.22)
        self.play(FadeOut(win))
        msg = nota("Large where the patch looks like the kernel: + on the left edge of the bright stripe, − on the right.",
                   font_size=20)
        self.play(FadeIn(msg))
        self.pausa(2.0)


# ======================================================================
# A03 · The receptive field grows with depth
# ======================================================================
N_A03 = 15


def capas_rf(s, K=3, n0=N_A03, D=3):
    """Longitudes de capa sin padding y los índices del campo receptivo de la unidad central de arriba."""
    lens = [n0]
    for _ in range(D):
        lens.append((lens[-1] - K) // s + 1)
    top = lens[-1] // 2
    rf = [[top]]
    for l in range(D, 0, -1):
        hijos = sorted({i * s + u for i in rf[0] for u in range(K)})
        rf.insert(0, hijos)
    return lens, rf          # rf[l] = índices en la capa l (0 = entrada)


class A03_ReceptiveField(EscenaBase):
    def construct(self):
        tit = titulo("The receptive field grows with depth")
        form = M(r"r_l = r_{l-1} + (K_l-1)\prod_{i<l} s_i,\qquad r_0 = 1", font_size=32)
        form.next_to(tit, DOWN, buff=0.2, aligned_edge=LEFT)
        self.add(tit, form)

        YS = [-2.75, -1.35, 0.05, 1.45]
        XS0 = np.linspace(-5.65, -0.55, N_A03)
        nombres = VGroup(*[T(t, font_size=18, color=GRIS_TXT).move_to([-6.45, y, 0])
                           for t, y in zip(["input", "layer 1", "layer 2", "layer 3"], YS)])

        # rejilla 2-D de la entrada a la derecha
        CL = 0.24
        grid = VGroup(*[Square(CL, stroke_color=GRIS_CLARO, stroke_width=1) for _ in range(N_A03 ** 2)])
        grid.arrange_in_grid(N_A03, N_A03, buff=0).move_to([3.7, -0.3, 0])
        grid_lbl = T("input, 15 × 15", font_size=20, color=GRIS_TXT).next_to(grid, UP, buff=0.15)
        self.play(FadeIn(nombres), FadeIn(grid), FadeIn(grid_lbl))

        def construir(s):
            lens, rf = capas_rf(s)
            xs = [XS0]
            for l in range(1, 4):
                prev = xs[-1]
                xs.append(np.array([prev[i * s:i * s + 3].mean() for i in range(lens[l])]))
            puntos = [VGroup(*[Dot([x, YS[l], 0], radius=0.085, color=GRIS_CLARO) for x in xs[l]])
                      for l in range(4)]
            return lens, rf, xs, puntos

        def fase(s, etiqueta, color_reg):
            lens, rf, xs, puntos = construir(s)
            cab = T(etiqueta, font_size=24, weight=BOLD, color=color_reg).move_to([3.7, 2.55, 0])
            self.play(FadeIn(cab), *[FadeIn(p) for p in puntos])
            top = rf[3][0]
            self.play(puntos[3][top].animate.set_color(NARANJA).scale(1.6))
            lecturas = VGroup()
            todas = VGroup()
            region = None
            r = 1
            prod_s = 1
            for l in range(3, 0, -1):
                lineas = VGroup()
                for i in rf[l]:
                    for u in range(3):
                        j = i * s + u
                        lineas.add(Line(puntos[l][i].get_center(), puntos[l - 1][j].get_center(),
                                        color=color_reg, stroke_width=2.2, stroke_opacity=0.8))
                todas.add(lineas)
                self.play(Create(lineas), *[puntos[l - 1][j].animate.set_color(color_reg) for j in rf[l - 1]],
                          run_time=0.9)
            # lectura capa por capa (de abajo hacia arriba): r_1, r_2, r_3
            for l in range(1, 4):
                r = r + (3 - 1) * prod_s
                prod_s *= s
                lec = M(rf"r_{l} = {r}", font_size=30, color=color_reg)
                lecturas.add(lec)
            lecturas.arrange(RIGHT, buff=0.6).next_to(grid, DOWN, buff=0.3)
            for l, lec in enumerate(lecturas):
                rr = int(lec.get_tex_string().split("=")[1])
                nueva = Square(rr * CL, stroke_color=color_reg, stroke_width=4,
                               fill_color=color_reg, fill_opacity=0.18).move_to(grid.get_center())
                if region is None:
                    self.play(FadeIn(nueva), FadeIn(lec), run_time=0.7)
                else:
                    self.play(ReplacementTransform(region, nueva), FadeIn(lec), run_time=0.7)
                region = nueva
            return VGroup(todas, cab, *puntos), lecturas, region

        g1, lec1, reg1 = fase(1, "K = 3, stride 1", TEAL)
        n1 = nota("Stride 1: each layer adds K − 1 = 2. Three layers see 7 × 7.")
        self.play(FadeIn(n1))
        self.pausa()

        self.play(FadeOut(g1), FadeOut(reg1), FadeOut(n1),
                  lec1.animate.scale(0.8).set_opacity(0.55).next_to(grid, DOWN, buff=0.85))
        g2, lec2, reg2 = fase(2, "K = 3, stride 2", NARANJA)
        n2 = nota("Stride 2: each layer adds 2 × (product of earlier strides). Three layers see all 15 × 15.")
        self.play(FadeIn(n2))
        self.pausa(2.0)


# ======================================================================
# A04 · From image to tokens
# ======================================================================
P_A04, LADO_A04, D_TOK = 16, 64, 8


def imagen_a04():
    from skimage import data, transform
    im = data.chelsea()                                  # foto de un gato, sin restricciones de copyright
    h, w, _ = im.shape
    c = min(h, w)
    im = im[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2 + 30:(w - c) // 2 + 30 + c]
    im = transform.resize(im, (LADO_A04, LADO_A04), anti_aliasing=True)
    return (im * 255).astype(np.uint8)


def pixel_art(arr, alto):
    m = ImageMobject(arr)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.set(height=alto)


class A04_PatchesToTokens(EscenaBase):
    def construct(self):
        im = imagen_a04()
        n = LADO_A04 // P_A04                              # 4 × 4 patches
        patches = [im[i * P_A04:(i + 1) * P_A04, j * P_A04:(j + 1) * P_A04] for i in range(n) for j in range(n)]
        planos = [p.reshape(-1, 3).astype(float) / 255 for p in patches]          # 256 píxeles × 3
        rng = np.random.default_rng(7)
        W_E = rng.normal(0, 1 / np.sqrt(P_A04 * P_A04 * 3), (D_TOK, P_A04 * P_A04 * 3))
        tokens = np.array([W_E @ (p.reshape(-1) - 0.5) for p in planos])          # 16 × d
        pos = rng.normal(0, 0.6 * tokens.std(), (n * n, D_TOK))
        vmax = np.abs(tokens + pos).max()

        tit = titulo("From image to tokens")
        self.add(tit)

        ALTO = 3.6
        full = pixel_art(im, ALTO).move_to([-3.8, -0.4, 0])
        lab = T("image  3 × 64 × 64", font_size=22).next_to(full, UP, buff=0.18)
        self.play(FadeIn(full), FadeIn(lab))
        lineas = VGroup()
        for k in range(1, n):
            d = ALTO * k / n
            lineas.add(Line(full.get_corner(UL) + RIGHT * d, full.get_corner(DL) + RIGHT * d,
                            color=WHITE, stroke_width=3))
            lineas.add(Line(full.get_corner(UL) + DOWN * d, full.get_corner(UR) + DOWN * d,
                            color=WHITE, stroke_width=3))
        txt = VGroup(T("patches of 16 × 16", font_size=24),
                     M(r"\tfrac{64}{16}\times\tfrac{64}{16} = 16 \text{ patches}", font_size=32)
                     ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([2.6, 0.3, 0])
        self.play(Create(lineas), FadeIn(txt))
        self.pausa()

        # patches separados en su posición
        lado_p = ALTO / n
        piezas = Group()
        for k, p in enumerate(patches):
            i, j = divmod(k, n)
            pm = pixel_art(p, lado_p)
            pm.move_to(full.get_corner(UL) + RIGHT * (j + 0.5) * lado_p + DOWN * (i + 0.5) * lado_p)
            piezas.add(pm)
        self.add(piezas)
        self.remove(full)
        self.play(FadeOut(lineas), FadeOut(lab), FadeOut(txt),
                  *[piezas[k].animate.shift(0.12 * ((k % n) - 1.5) * RIGHT + 0.12 * (1.5 - k // n) * UP)
                    for k in range(n * n)])
        self.pausa()

        # en fila, en orden de lectura
        XS = np.linspace(-5.7, 6.0, n * n)
        Y_FILA = 2.3
        self.play(*[piezas[k].animate.set(height=0.62).move_to([XS[k], Y_FILA, 0]) for k in range(n * n)],
                  run_time=1.6)
        orden = T("in reading order: patch 1, 2, …, 16", font_size=20, color=GRIS_TXT).move_to([0.15, 2.95, 0])
        self.play(FadeIn(orden))

        # aplanar: cada patch → columna de 256 píxeles (× 3 canales) = 768 números
        tiras = Group()
        for k, p in enumerate(planos):
            col = (p.reshape(P_A04 * P_A04, 1, 3) * 255).astype(np.uint8)
            t = ImageMobject(col)
            t.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            t.stretch_to_fit_height(1.5).stretch_to_fit_width(0.16).move_to([XS[k], 1.75, 0])
            tiras.add(t)
        flat = M(r"\text{flatten: } 16\cdot16\cdot3 = 768 \text{ numbers}", font_size=28).move_to([0.15, 2.95, 0])
        self.play(*[FadeTransform(piezas[k], tiras[k]) for k in range(n * n)], FadeOut(orden), FadeIn(flat),
                  run_time=1.4)
        self.pausa()

        # la misma W_E para todos
        caja = RoundedRectangle(corner_radius=0.12, width=12.6, height=0.55, stroke_color=NAVY,
                                stroke_width=1.6, fill_color=AZ_FILL, fill_opacity=1).move_to([0.15, 0.45, 0])
        caja_t = M(r"\text{the same linear map } W_E \in \mathbb{R}^{d\times 768}\text{ for every patch}",
                   font_size=28).move_to(caja)
        self.play(FadeIn(caja), FadeIn(caja_t))

        CEL = 0.17

        def columna(vals, borde=NAVY):
            g = VGroup(*[Rectangle(width=0.36, height=CEL, stroke_color=borde, stroke_width=0.8,
                                   fill_color=con_signo(v, vmax), fill_opacity=1) for v in vals])
            return g.arrange(DOWN, buff=0)

        Y_TOK = -0.6
        toks = VGroup(*[columna(tokens[k]).move_to([XS[k], Y_TOK, 0]) for k in range(n * n)])
        self.play(*[FadeTransform(tiras[k], toks[k]) for k in range(n * n)], run_time=1.6)
        etiq = VGroup(*[M(rf"t_{{{k + 1}}}", font_size=24).next_to(toks[k], DOWN, buff=0.1) for k in (0, 1, 15)],
                      M(r"\cdots", font_size=28).next_to(toks[8], DOWN, buff=0.16))
        dlab = M(r"d", font_size=28, color=GRIS_TXT).next_to(toks[15], RIGHT, buff=0.12)
        self.play(FadeIn(etiq), FadeIn(dlab), FadeOut(flat))
        self.pausa()

        # + posición
        mas = M(r"+\,p_i", font_size=30, color=GRIS_TXT).move_to([-6.45, Y_TOK, 0])
        posg = VGroup(*[columna(pos[k], borde=GRIS_EJE).set_opacity(0.85).move_to([XS[k], Y_TOK - 1.85, 0])
                        for k in range(n * n)])
        plab = T("positional embedding: where the patch was", font_size=20, color=GRIS_TXT)
        plab.next_to(posg, DOWN, buff=0.1)
        self.play(FadeIn(posg), FadeIn(plab), FadeIn(mas))
        nuevos = VGroup(*[columna(tokens[k] + pos[k]).move_to(toks[k]) for k in range(n * n)])
        self.play(*[posg[k].animate.move_to(toks[k]).set_opacity(0) for k in range(n * n)],
                  *[Transform(toks[k], nuevos[k]) for k in range(n * n)], FadeOut(plab), FadeOut(mas),
                  run_time=1.4)
        self.remove(posg)

        # token [CLS]
        cls_v = rng.normal(0, tokens.std(), D_TOK)
        cls = VGroup(*[Rectangle(width=0.36, height=CEL, stroke_color=NARANJA, stroke_width=1.6,
                                 fill_color=NA_FILL, fill_opacity=1) for _ in cls_v]).arrange(DOWN, buff=0)
        cls.move_to([-6.45, Y_TOK, 0])
        cls_l = M(r"[\texttt{CLS}]", font_size=24, color=NARANJA).next_to(cls, DOWN, buff=0.1)
        self.play(FadeIn(cls, shift=RIGHT * 0.3), FadeIn(cls_l))
        fin = nota("nn.Conv2d(3, d, kernel_size=16, stride=16) computes exactly these tokens.", font_size=20)
        resumen = T("17 tokens of size d enter the Transformer", font_size=22, color=NAVY)
        resumen.move_to([0.15, -2.45, 0])
        self.play(FadeIn(resumen), FadeIn(fin))
        self.pausa(2.0)
