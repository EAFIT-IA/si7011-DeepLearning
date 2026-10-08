"""Insert (or refresh) an "open in Colab / Kaggle / Lightning" badge cell at the top of every course notebook.

Run from the repository root after adding or renaming a notebook: python tools/add_open_in_badges.py
"""
import glob, sys, urllib.parse, nbformat as nbf
REPO = 'EAFIT-IA/si7011-DeepLearning'
MARK = '<!-- open-in-badges -->'

# Published Lightning Studios (one Studio per session). Others import from GitHub.
LIGHTNING_STUDIO = {
    'sessions/01_learning/': 'https://lightning.ai/jdmartinev-org/vision-model/studios/si7011-sesion01/code',
}

# Published Kaggle notebooks (students click "Copy & Edit"). Others import from GitHub.
KAGGLE = {
    'sessions/01_learning/notebooks/sesion_01_pytorch_60min.ipynb':
        'https://www.kaggle.com/code/juanmartinezv4399/si7011-sesion-01-pytorch-60min-ipynb',
}


def lightning_url(path, blob):
    for prefix, url in LIGHTNING_STUDIO.items():
        if path.startswith(prefix):
            return url
    return 'https://lightning.ai/new?repo_url=' + urllib.parse.quote(blob, safe='')


def badges(path):
    blob = f'https://github.com/{REPO}/blob/main/{path}'
    return (f"{MARK}\n"
            f"[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/{path}) "
            f"[![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
            f"({KAGGLE.get(path, 'https://kaggle.com/kernels/welcome?src=' + blob)}) "
            f"[![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)]"
            f"({lightning_url(path, blob)})")


if __name__ == '__main__':
    for path in sorted(glob.glob('sessions/*/notebooks/*.ipynb')):
        nb = nbf.read(path, as_version=4)
        cell = nbf.v4.new_markdown_cell(badges(path))
        if nb.cells and nb.cells[0].cell_type == 'markdown' and MARK in nb.cells[0].source:
            nb.cells[0] = cell
        else:
            nb.cells.insert(0, cell)
        nbf.write(nb, path)
        print('ok', path)
