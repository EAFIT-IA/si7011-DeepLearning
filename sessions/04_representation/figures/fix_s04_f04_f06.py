"""
Correcciones sobre las imágenes del generador de S04-F04 y S04-F06.

S04-F04 (MAE): en la salida faltaban los bordes naranja de dos parches enmascarados,
(fila 1, col 1) y (fila 2, col 3), así que parecía que la pérdida se calculaba sobre 10
parches y no sobre 12; y la entrada del decoder mostraba los parches crudos en lugar de
los tokens codificados h_i. Se dibujan los dos bordes y se reemplazan los 4 parches por
tokens h_i.

S04-F06 (SimCLR): la vista en gris era otro pájaro. Las dos vistas se reconstruyen a
partir de x: recorte + cambio de color (vista i) y la imagen completa volteada en gris
(vista j).

Entradas: s04_f04_mae_pipeline_generated.png, s04_f06_simclr_generated.png (las imágenes
tal como salieron del generador). Salidas: s04_f04_mae_pipeline.png, s04_f06_simclr.png.
Las coordenadas son las de esas dos imágenes (1942 × 809 y 1855 × 848).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

AQUI = Path(__file__).resolve().parent
FUENTE = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
NARANJA = (227, 109, 31)
AZUL_TOKEN, AZUL_BORDE, NAVY = (148, 212, 254), (31, 111, 180), (20, 33, 61)


def corregir_f04():
    im = Image.open(AQUI / "s04_f04_mae_pipeline_generated.png").convert("RGB")
    d = ImageDraw.Draw(im)
    X = [(1626, 1694), (1699, 1767), (1772, 1840), (1845, 1913)]       # celdas de la salida
    Y = [(153, 221), (226, 295), (300, 369), (374, 443)]
    for i, j in [(0, 0), (1, 2)]:                                       # enmascarados sin borde
        d.rectangle([X[j][0], Y[i][0], X[j][1], Y[i][1]], outline=NARANJA, width=2)
    XC = [(1152, 1206), (1215, 1268), (1276, 1331), (1340, 1394)]      # entrada del decoder
    YC = [(150, 214), (222, 286), (295, 360), (370, 435)]
    f, fs = ImageFont.truetype(FUENTE, 30), ImageFont.truetype(FUENTE, 18)
    for i, j in [(0, 1), (1, 3), (2, 0), (3, 2)]:                       # los 4 visibles → h_i
        x0, x1 = XC[j]
        y0, y1 = YC[i]
        d.rectangle([x0 - 6, y0 - 12, x1 + 6, y1 + 4], fill=(255, 255, 255))
        d.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=AZUL_TOKEN, outline=AZUL_BORDE, width=2)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        d.text((cx - 4, cy + 2), "h", font=f, fill=NAVY, anchor="mm")
        d.text((cx + 10, cy + 12), "i", font=fs, fill=NAVY, anchor="mm")
    im.save(AQUI / "s04_f04_mae_pipeline.png")


def corregir_f06():
    im = Image.open(AQUI / "s04_f06_simclr_generated.png").convert("RGB")
    x = im.crop((38, 309, 249, 518))                                    # interior de x
    W, H = x.size
    vi = x.crop((int(0.30 * W), int(0.18 * H), int(0.95 * W), int(0.83 * H)))
    vi = ImageEnhance.Brightness(ImageEnhance.Color(vi).enhance(1.6)).enhance(1.08)
    r, g, b = vi.split()
    vi = Image.merge("RGB", (b.point(lambda v: v * 0.9), g, r))         # cambio de tono
    vj = ImageOps.grayscale(ImageOps.mirror(x)).convert("RGB")
    for v, (x0, y0, x1, y1) in [(vi, (378, 136, 590, 350)), (vj, (378, 499, 590, 702))]:
        im.paste(v.resize((x1 - x0 + 1, y1 - y0 + 1), Image.LANCZOS), (x0, y0))
    im.save(AQUI / "s04_f06_simclr.png")


if __name__ == "__main__":
    corregir_f04()
    corregir_f06()
