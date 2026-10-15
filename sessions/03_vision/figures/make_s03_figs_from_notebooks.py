# Builds S03-F02, S03-F06 and S03-F10 from the outputs of the executed notebooks 3.1 and 3.4.
# Usage (from this folder): python make_s03_figs_from_notebooks.py ../notebooks .
# Numbers and images come from the notebooks' saved outputs, so the slides show the same run the students see.
import ast, base64, io, json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

NAVY, BLUE, TEAL, ORANGE, GRAY = '#14213d', '#1f6fb4', '#167d78', '#d9631e', '#8a94a3'
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'DejaVu Sans'})
W, H, DPI = 1672, 941, 100
NB, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def cell(nb_name, needle):
    nb = json.load(open(NB / nb_name))
    hits = [c for c in nb['cells'] if c['cell_type'] == 'code' and needle in ''.join(c['source'])]
    assert len(hits) == 1 and hits[0].get('outputs'), f'{nb_name}: run the notebook first ({needle!r})'
    return hits[0]['outputs']


def image(outputs):
    png = next(o['data']['image/png'] for o in outputs if 'image/png' in o.get('data', {}))
    return Image.open(io.BytesIO(base64.b64decode(png))).convert('RGB')


def text(outputs):
    return ''.join(''.join(o.get('text', '')) + ''.join(o.get('data', {}).get('text/plain', '')) for o in outputs)


def banner(fig, msg):
    fig.patches.append(FancyBboxPatch((0.0375, 0.045), 0.925, 0.09, boxstyle='round,pad=0,rounding_size=0.012',
                                      transform=fig.transFigure, fc='#eef0f8', ec='#b9bfd6', lw=1.2, zorder=-5))
    fig.text(0.5, 0.09, msg, ha='center', va='center', fontsize=17, color=NAVY, weight='bold')


# ------------------------------------------------------------------ S03-F02 · permutation experiment (notebook 3.1)
res = ast.literal_eval(text(cell('sesion_03_1_imagenes_estructura.ipynb', 'resultados = {name: max(')).strip())
grid = image(cell('sesion_03_1_imagenes_estructura.ipynb', 'permute = lambda X'))

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
split = int(grid.height * 0.52)                                           # row 1: originals with class names
for k, (part, label, col) in enumerate([(grid.crop((0, 0, grid.width, split)), 'original', NAVY),
                                         (grid.crop((0, split, grid.width, grid.height)), 'permuted\n(one fixed π)', ORANGE)]):
    ax = fig.add_axes([0.15, 0.55 - 0.31 * k, 0.44, 0.32]); ax.imshow(part); ax.axis('off')
    fig.text(0.14, 0.71 - 0.31 * k, label, fontsize=16, color=col, ha='right', va='center')
bx = fig.add_axes([0.68, 0.27, 0.29, 0.6])
groups, x = ['MLP', 'CNN'], [0, 1]
orig = [res['mlp'], res['cnn']]; perm = [res['mlp_permutado'], res['cnn_permutado']]
b1 = bx.bar([i - 0.19 for i in x], orig, 0.36, color=BLUE, label='original')
b2 = bx.bar([i + 0.19 for i in x], perm, 0.36, color=ORANGE, label='pixels permuted')
for bars in (b1, b2):
    for r in bars:
        bx.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f'{r.get_height():.2f}', ha='center', fontsize=15, color=NAVY)
bx.axhline(0.1, color=GRAY, ls=':', lw=1.5); bx.text(-0.42, 0.11, 'chance', color=GRAY, fontsize=12, ha='left')
bx.set_xticks(x, groups, fontsize=17); bx.set_ylim(0, 0.75); bx.set_ylabel('validation accuracy', fontsize=14)
bx.spines[['top', 'right']].set_visible(False); bx.legend(frameon=False, fontsize=13, loc='upper left')
banner(fig, 'The MLP does not notice the permutation. The CNN loses its advantage: it assumed that neighbors matter.')
fig.savefig(OUT / 's03_f02_permutation_experiment.png', dpi=DPI, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ S03-F10 · transfer vs labeled data (notebook 3.4)
rows = re.findall(r'k = (\S+)\s+· probe ([\d.]+) · fine-tuning ([\d.]+) · desde cero ([\d.]+)',
                  text(cell('sesion_03_4_transfer_learning.ipynb', "for name in KS:\n    print(f'k = ")))
sizes = text(cell('sesion_03_4_transfer_learning.ipynb', "{name: len(subset(k)[1])"))
n_all = ast.literal_eval(sizes.strip())['todas'] / 37
xs = [5, 20, n_all]
probe, ft, scratch = ([float(r[i]) for r in rows] for i in (1, 2, 3))

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0.1, 0.24, 0.62, 0.66])
for ys, name, col, mk in [(probe, 'linear probe (backbone frozen)', TEAL, 'o'), (ft, 'fine-tuning', BLUE, 's'),
                          (scratch, 'from scratch', ORANGE, '^')]:
    ax.plot(xs, ys, marker=mk, ms=10, lw=3, color=col, label=name)
    for i, (xv, yv) in enumerate(zip(xs, ys)):
        other = probe[i] if ys is ft else ft[i]                              # probe and fine-tuning are close:
        dy = -0.06 if (ys is not scratch and yv < other) else 0.035         # the lower one is labelled below
        ax.text(xv, yv + dy, f'{yv:.2f}', ha='center', fontsize=13, color=col)
ax.axhline(1 / 37, color=GRAY, ls=':', lw=1.5); ax.text(xs[-1], 1 / 37 + 0.015, 'chance (1/37)', color=GRAY, fontsize=12, ha='right')
ax.set_xscale('log'); ax.set_xticks(xs, ['5', '20', f'{n_all:.0f}\n(all)'], fontsize=15); ax.minorticks_off()
ax.set_ylim(0, 1); ax.set_xlabel('training images per breed', fontsize=15); ax.set_ylabel('validation accuracy', fontsize=15)
ax.tick_params(labelsize=13); ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=14, loc='center left', bbox_to_anchor=(1.02, 0.55))
fig.text(0.75, 0.3, 'Oxford-IIIT Pets, 37 breeds\nResNet-18 pretrained on ImageNet', fontsize=12, color=GRAY)
banner(fig, 'Few labels: the frozen backbone wins. More labels: fine-tuning pulls ahead. From scratch stays far behind.')
fig.savefig(OUT / 's03_f10_transfer_vs_data.png', dpi=DPI, facecolor='white'); plt.close(fig)

# ------------------------------------------------------------------ S03-F06 · feature hierarchy (notebook 3.4)
kern = image(cell('sesion_03_4_transfer_learning.ipynb', 'backbone.conv1.weight'))
kern = kern.crop((0, 38, kern.width, kern.height))                       # drop the notebook's Spanish title
patches = image(cell('sesion_03_4_transfer_learning.ipynb', 'stages = {'))

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.text(0.04, 0.925, 'conv1 · the 64 kernels (7 × 7 × 3) learned on ImageNet', fontsize=16, color=NAVY, weight='bold')
ax = fig.add_axes([0.04, 0.585, 0.56, 0.325]); ax.imshow(kern); ax.axis('off')
fig.text(0.62, 0.78, 'edges at every orientation,\ncolor blobs and color contrasts', fontsize=15, color=NAVY, va='center')
fig.text(0.04, 0.555, 'patches that most activate 4 channels per stage (3 patches each), Oxford-IIIT Pets',
         fontsize=16, color=NAVY, weight='bold')
ax = fig.add_axes([0.04, 0.16, 0.66, 0.38]); ax.imshow(patches); ax.axis('off')
for y, msg in [(0.465, 'small parts: eyes, noses, ears'), (0.345, 'faces'), (0.225, 'whole animals')]:
    fig.text(0.72, y, '← ' + msg, fontsize=15, color=TEAL, va='center')
banner(fig, 'Each stage combines the one below: the receptive field grows, and so does the pattern.')
fig.savefig(OUT / 's03_f06_feature_hierarchy.png', dpi=DPI, facecolor='white'); plt.close(fig)
print('F02:', res, '· F10:', list(zip(xs, probe, ft, scratch)))
