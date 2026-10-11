"""
Datos para la animación S04-A01 (el espacio latente se organiza solo).

Entrena un autoencoder MLP 784 → 256 → 2 → 256 → 784 en Fashion-MNIST, con MSE y
sin usar etiquetas, y guarda el código h = g(x) de 1 000 imágenes de prueba en
varios momentos del entrenamiento. Las etiquetas se guardan solo para colorear
el último cuadro de la animación.

Es el mismo modelo de la sección 4 del notebook 4.1 (k = 2).

Uso:
    python make_s04_a01_latent_snapshots.py               # descarga con torchvision
    python make_s04_a01_latent_snapshots.py --data DIR    # DIR con los .gz originales
Salida: s04_a01_latent_snapshots.npz junto a este archivo.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import torch
from torch import nn

AQUI = Path(__file__).resolve().parent
PASOS_SNAPSHOT = [0, 25, 75, 150, 300, 600, 1200, 2400, 4800]   # pasos de optimización
N_VIS = 1000


def leer_idx(ruta):
    with gzip.open(ruta, "rb") as f:
        datos = f.read()
    if datos[2] == 8 and datos[3] == 3:                          # imágenes
        n, h, w = (int.from_bytes(datos[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(datos, np.uint8, offset=16).reshape(n, h * w)
    return np.frombuffer(datos, np.uint8, offset=8)               # etiquetas


def cargar(data_dir):
    if data_dir:
        d = Path(data_dir)
        xtr = leer_idx(d / "train-images-idx3-ubyte.gz")
        xte = leer_idx(d / "t10k-images-idx3-ubyte.gz")
        yte = leer_idx(d / "t10k-labels-idx1-ubyte.gz")
    else:
        from torchvision import datasets
        tr = datasets.FashionMNIST("data", train=True, download=True)
        te = datasets.FashionMNIST("data", train=False, download=True)
        xtr, xte, yte = tr.data.reshape(-1, 784).numpy(), te.data.reshape(-1, 784).numpy(), te.targets.numpy()
    f = lambda a: torch.tensor(a, dtype=torch.float32) / 255
    return f(xtr), f(xte), torch.tensor(yte.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    x_tr, x_te, y_te = cargar(args.data)
    idx = torch.randperm(len(x_te), generator=torch.Generator().manual_seed(0))[:N_VIS]
    x_vis, y_vis = x_te[idx], y_te[idx]

    k = 2
    g = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, k))           # encoder g
    d = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 784))           # decoder d
    opt = torch.optim.Adam([*g.parameters(), *d.parameters()], lr=1e-3)

    codigos, perdidas = [], []

    def foto():
        with torch.no_grad():
            h = g(x_vis)                                                           # h = g(x)
            mse = ((d(g(x_te)) - x_te) ** 2).mean().item()                         # L sobre todo el test
        codigos.append(h.numpy())
        perdidas.append(mse)
        print(f"paso {paso:5d}  MSE test {mse:.4f}")

    paso, lote = 0, 128
    foto()
    while paso < PASOS_SNAPSHOT[-1]:
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr) - lote + 1, lote):
            xb = x_tr[perm[i:i + lote]]
            loss = ((d(g(xb)) - xb) ** 2).mean()                                   # L_i = ||x - x̂||², promedio
            opt.zero_grad()
            loss.backward()
            opt.step()
            paso += 1
            if paso in PASOS_SNAPSHOT:
                foto()
            if paso >= PASOS_SNAPSHOT[-1]:
                break

    # probe lineal sobre h (con etiquetas, solo para reportar cuánta estructura hay)
    with torch.no_grad():
        h_tr, h_te = g(x_tr), g(x_te)
    mu, sd = h_tr.mean(0), h_tr.std(0)
    h_tr, h_te = (h_tr - mu) / sd, (h_te - mu) / sd
    y_tr = torch.tensor(leer_idx(Path(args.data) / "train-labels-idx1-ubyte.gz").astype(np.int64)) \
        if args.data else None
    if y_tr is None:
        from torchvision import datasets
        y_tr = datasets.FashionMNIST("data", train=True).targets
    probe = nn.Linear(k, 10)
    po = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        l = nn.functional.cross_entropy(probe(h_tr), y_tr)
        po.zero_grad()
        l.backward()
        po.step()
    acc = (probe(h_te).argmax(1) == y_te).float().mean().item()
    print(f"probe lineal sobre h (k=2): {acc:.3f}")

    np.savez_compressed(AQUI / "s04_a01_latent_snapshots.npz", codigos=np.stack(codigos),
                        pasos=np.array(PASOS_SNAPSHOT), mse=np.array(perdidas),
                        etiquetas=y_vis.numpy(), probe=acc)


if __name__ == "__main__":
    main()
