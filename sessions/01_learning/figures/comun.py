"""Estilo, datos y utilidades compartidas por las escenas de la sesión 1 (SI7011).

- Paleta tomada de las diapositivas Marp de la sesión.
- Los datos sintéticos usan semillas fijas: el notebook de la práctica 1 puede
  importar `datos_regresion()` para trabajar exactamente con los mismos puntos.
- Pausas: si `manim-slides` está instalado, cada `self.pausa()` crea un corte
  para avanzar con el teclado. Con la variable de entorno S01_VIDEO=1 las
  escenas heredan de `Scene` y las pausas se convierten en esperas (video lineal).
"""
import os

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, UL, Axes, DecimalNumber, FadeIn, FadeOut, MathTex,
    Scene, Tex, Text, VGroup, config,
)

# ---------------------------------------------------------------- paleta
FONDO = "#ffffff"
TINTA = "#15263b"
AZUL = "#183c65"       # títulos, clase 0
TEAL = "#166b73"       # modelo / predicciones
NARANJA = "#d78b32"    # énfasis, clase 1
GRIS = "#617186"
GRIS_CLARO = "#c9d3df"
ROJO = "#b5443b"       # residuos, pérdida
AZUL_CLASE = "#2f6cab"  # clase 0 en dispersión (más claro que AZUL para puntos)

import manimpango  # noqa: E402

_disponibles = set(manimpango.list_fonts())
# Arial es la fuente de las diapositivas; si no está instalada se usa una equivalente.
FUENTE = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _disponibles), "")

config.background_color = FONDO
Text.set_default(color=TINTA, font=FUENTE)
MathTex.set_default(color=TINTA)
Tex.set_default(color=TINTA)
DecimalNumber.set_default(color=TINTA)

# ---------------------------------------------------------------- escena base
USAR_SLIDES = os.environ.get("S01_VIDEO", "0") != "1"
try:
    if not USAR_SLIDES:
        raise ImportError
    from manim_slides import Slide as _Base
except ImportError:  # pragma: no cover
    _Base = Scene
    USAR_SLIDES = False


class EscenaBase(_Base):
    """Escena con encabezado, leyenda inferior y pausas docentes."""

    codigo = ""
    titulo = ""

    def setup(self):
        super().setup()
        self._leyenda = None
        if self.codigo or self.titulo:
            # Production codes (S01-Vxx) are not shown to students.
            cab = VGroup(
                Text(self.titulo, font_size=24, color=AZUL, weight="BOLD"),
            )
            cab.to_corner(UL, buff=0.35)
            self.add(cab)
            self.encabezado = cab

    def pausa(self, espera=1.2):
        """Punto donde el docente pide una predicción antes de continuar."""
        if USAR_SLIDES:
            self.wait(0.2)
            self.next_slide()
        else:
            self.wait(espera)

    def leyenda(self, texto, color=TINTA, resaltar=None, run_time=0.5):
        """Reemplaza el texto explicativo de la parte inferior."""
        t2c = {"Question:": NARANJA}
        if resaltar:
            t2c.update(resaltar)
        nueva = Text(texto, font_size=25, color=color, t2c=t2c, line_spacing=0.8)
        if nueva.width > 12.8:
            nueva.scale_to_fit_width(12.8)
        nueva.to_edge(DOWN, buff=0.3)
        anims = [FadeIn(nueva, shift=0.1 * UP)]
        if self._leyenda is not None:
            anims.append(FadeOut(self._leyenda))
        self.play(*anims, run_time=run_time)
        self._leyenda = nueva
        return nueva

    def quitar_leyenda(self):
        if self._leyenda is not None:
            self.play(FadeOut(self._leyenda), run_time=0.3)
            self._leyenda = None


def ejes(x_range, y_range, x_length, y_length, x_label=None, y_label=None,
         numeros=True, decimales=0, fs=20):
    """Axes con el estilo de las diapositivas."""
    ax = Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        tips=False,
        axis_config={
            "color": TINTA, "stroke_width": 2, "include_numbers": numeros,
            "font_size": fs, "tick_size": 0.05,
            "decimal_number_config": {"num_decimal_places": decimales, "color": TINTA},
        },
    )
    etiquetas = VGroup()
    if x_label:
        lx = MathTex(x_label, font_size=30).next_to(ax.x_axis.get_end(), DOWN + RIGHT * 0.2, buff=0.15)
        etiquetas.add(lx)
    if y_label:
        ly = MathTex(y_label, font_size=30).next_to(ax.y_axis.get_end(), UP, buff=0.12)
        etiquetas.add(ly)
    # Las etiquetas viajan con los ejes (move_to/scale); `ax.etiquetas` queda
    # como grupo vacío para que las escenas puedan seguir usando FadeIn(ax.etiquetas).
    ax.add(etiquetas)
    ax.etiquetas = VGroup()
    return ax


def lectura(simbolo, valor, decimales=2, fs=32, color=TINTA):
    """Grupo 'símbolo = valor' con número actualizable (devuelve grupo, número)."""
    s = MathTex(simbolo + "=", font_size=fs, color=color)
    n = DecimalNumber(valor, num_decimal_places=decimales, font_size=fs, color=color,
                      include_sign=False)
    g = VGroup(s, n).arrange(RIGHT, buff=0.12)
    return g, n


def fijar_numero(num, anterior, getter):
    """Actualizador que mantiene un DecimalNumber junto a su símbolo."""
    def _upd(m):
        m.set_value(getter())
        m.next_to(anterior, RIGHT, buff=0.12)
    return _upd


# ---------------------------------------------------------------- datos
def datos_regresion():
    """8 puntos sintéticos (semilla 11). y ≈ 1.2 x + 2 + ruido. Unidades abstractas.

    Incluye entradas negativas a propósito: al aumentar w, las predicciones con
    x < 0 bajan (Δŷ = x·Δw), como señala la nota de la diapositiva 07.
    """
    rng = np.random.default_rng(11)
    x = np.array([-2.6, -1.9, -1.1, -0.4, 0.3, 1.0, 1.8, 2.5])
    y = np.round(1.2 * x + 2.0 + rng.normal(0.0, 0.55, x.size), 2)
    return x, y


def mse(w, b, x, y):
    return float(np.mean((y - (w * x + b)) ** 2))


def optimo_mse(x, y):
    """Mínimos cuadrados cerrados: (w*, b*, L*) y hessiana/2 H = E[[x²,x],[x,1]]."""
    A = np.column_stack([x, np.ones_like(x)])
    w, b = np.linalg.lstsq(A, y, rcond=None)[0]
    H = np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
    return float(w), float(b), mse(w, b, x, y), H


def elipse_nivel(nivel, w0, b0, L0, H, n=160):
    """Puntos (w, b) con L(w, b) = nivel para la MSE lineal (forma cuadrática)."""
    c = nivel - L0
    if c <= 0:
        return None
    vals, vecs = np.linalg.eigh(H)
    Hm12 = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    t = np.linspace(0, 2 * np.pi, n)
    u = np.vstack([np.cos(t), np.sin(t)])
    pts = np.sqrt(c) * (Hm12 @ u)
    return np.column_stack([w0 + pts[0], b0 + pts[1]])


def grad_mse(w, b, x, y):
    r = y - (w * x + b)
    return np.array([-2 * np.mean(x * r), -2 * np.mean(r)])


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def datos_clasificacion():
    """Dos clases separables aproximadamente por x1 + x2 = 0 (semilla 5)."""
    rng = np.random.default_rng(5)
    n = 12
    c1 = rng.normal([1.2, 1.0], 0.55, (n, 2))
    c0 = rng.normal([-1.2, -1.0], 0.55, (n, 2))
    return np.clip(c0, -2.7, 2.7), np.clip(c1, -2.7, 2.7)


def datos_v(seed=0, n=200, margen=0.25):
    """Puntos uniformes en [-2.2, 2.2]²; clase 1 por encima de la V  x₂ = |x₁| − 0.5.

    Se descartan los puntos a menos de `margen` de la frontera (en unidades de
    x₂ − |x₁| + 0.5) para que las clases queden visualmente separadas.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.2, 2.2, (n, 2))
    f = X[:, 1] - np.abs(X[:, 0]) + 0.5
    keep = np.abs(f) > margen
    return X[keep], (f[keep] > 0).astype(float)


def entrenar_mlp_v(semilla=3, lr=0.1, iteraciones=6000):
    """Entrena una MLP 2–2–1 (ReLU oculta, salida logit) con descenso de gradiente
    de lote completo y BCE desde logits sobre `datos_v()`.

    Nota honesta para clase: con estos datos e hiperparámetros, las semillas 2 y 7
    se quedan en 91–93 % de accuracy de entrenamiento; las demás semillas 0–9
    llegan a 99–100 %. La escena usa la semilla 3.
    """
    X, Y = datos_v()
    rng = np.random.default_rng(semilla)
    W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros(2)
    W2 = rng.normal(0, 1, (1, 2)); b2 = np.zeros(1)
    for _ in range(iteraciones):
        Z1 = X @ W1.T + b1
        Hh = np.maximum(Z1, 0)
        z = (Hh @ W2.T + b2)[:, 0]
        p = sigmoide(z)
        g = (p - Y) / len(Y)
        gW2 = g[None, :] @ Hh; gb2 = g.sum(keepdims=True)
        gZ1 = (g[:, None] * W2) * (Z1 > 0)
        W1 -= lr * (gZ1.T @ X); b1 -= lr * gZ1.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2
    z = (np.maximum(X @ W1.T + b1, 0) @ W2.T + b2)[:, 0]
    acc = float(np.mean((z >= 0) == (Y == 1)))
    bce = float(np.mean(np.logaddexp(0, z) - Y * z))
    return dict(X=X, Y=Y, W1=W1, b1=b1, W2=W2[0], b2=float(b2[0]),
                acc=acc, bce=bce, semilla=semilla, iteraciones=iteraciones, lr=lr)
