# SI7011 — Session 04

## Figure and animation inventory

Session 04 focuses on the question:

> **How can a network learn useful representations without explicit labels?**

The deck does not exist yet. Slide numbers below are those of the slide-by-slide plan, [`../S04_plan.md`](../S04_plan.md), where every figure and animation already has its own slide. When the deck is written, pending figures become hidden HTML comments in the slide source (`<!-- Pending figure S04-Fnn ... -->`, `<!-- Pending animation S04-Ann ... -->`), as in Session 03, so nothing unfinished is shown to students. To integrate a figure, replace the comment with:

```markdown
---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain <alt text>](../figures/s04_fnn_<name>.png)
```

Animations use the `media` class with a poster image, as in Sessions 02 and 03.

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper or another course (in particular, not the SimCLR augmentation figure or the self-supervised diagrams of the previous course PDF).

## Deck structure

66 slides in five parts, each opened by a divider slide that recaps the previous part: 51 content slides, 11 figures and 4 animations.

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · What is a representation? | V05–V13 | |
| 2 · Autoencoders | V14–V27 | Practice A (V27): notebooks 4.1–4.2 |
| 3 · Masked prediction | V28–V36 | |
| 4 · Contrastive learning | V37–V51 | |
| 5 · Evaluating representations | V52–V61 | Practice B (V61): notebooks 4.3–4.5 |
| Closure | V62–V66 | Integrating exercise (V62), exit question (V66) |

## Production routes

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md), same workflow as Sessions 02 and 03.
- **Script** — figures that show real data or a trained model: a `make_s04_fnn_*.py` script in this folder (matplotlib, fixed seeds). Numbers come from the notebook named in the table and are read from the JSON or checkpoint that notebook saves, so the slide and the notebook always agree. Specifications below.
- **Manim** — animations in `escenas_s04.py`, with the conventions of `escenas_s02.py` and `escenas_s03.py`.

## Planned figures

| ID | Slide | Purpose | Planned filename | Route | Status |
|---|---:|---|---|---|---|
| **S04-F01** | V09 | Supervised vs self-supervised pipeline: label from an annotator vs target generated from the input; pretext head discarded, $g$ kept | `s04_f01_supervised_vs_selfsupervised.png` | Generator | Pending |
| **S04-F02** | V16 | Autoencoder with shapes: 784 → 256 → $k$ → 256 → 784, bottleneck highlighted, loss | `s04_f02_autoencoder.png` | Generator | Pending |
| **S04-F03** | V21 | Reconstructions of the same 8 test images for $k$ = 2, 8, 32 and PCA with $k$ = 32, with test MSE and probe accuracy per row | `s04_f03_reconstructions_k.png` | Script (notebook 4.1, section 5: `resultados_ae.json`, `recon_k*.pt`) | Pending |
| **S04-F04** | V32 | Masked autoencoder: 16 patches, 75 % masked, encoder on visible tokens only, mask tokens, light decoder, loss on masked patches | `s04_f04_mae_pipeline.png` | Generator | Pending |
| **S04-F05** | V40 | Eight CIFAR-10 images, each with two SimCLR views | `s04_f05_augmented_views.png` | Script (notebook 4.4, section 2: same transform object) | Pending |
| **S04-F06** | V42 | SimCLR: two views, shared $g$, $\mathbf h$, projection head $p$, $\mathbf u$, similarity; downstream path uses $\mathbf h$ | `s04_f06_simclr.png` | Generator | Pending |
| **S04-F07** | V45 | $2N\times 2N$ similarity matrix for $N=8$ from the trained checkpoint; diagonal masked, positives marked, one row read as softmax | `s04_f07_similarity_matrix.png` | Script (checkpoint of notebook 4.4) | Pending |
| **S04-F08** | V55 | Retrieval: 5 query images and their 6 nearest neighbors with SimCLR $\mathbf h$ | `s04_f08_knn_retrieval.png` | Script (notebook 4.5, section 3) | Pending |
| **S04-F09** | V57 | Probe accuracy vs labels per class (1, 10, 100, 5000) for random, autoencoder, SimCLR and ImageNet encoders | `s04_f09_label_efficiency.png` | Script (notebook 4.5, section 5: `resultados_eficiencia.json`) | Pending |
| **S04-F10** | V59 | t-SNE of CIFAR-10 test embeddings: random $g$, autoencoder, SimCLR | `s04_f10_tsne_embeddings.png` | Script (notebook 4.5, section 6: `embeddings_test.npz`) | Pending |
| **S04-F11** | V65 | Summary: four objectives around one encoder $g$ and what each one keeps | `s04_f11_objectives_summary.png` | Generator | Pending |

## Script specifications

Common rules: white background, 16:9 at 1672 × 941 px (`figsize=(16.72, 9.41), dpi=100`), Arial/Helvetica, the slide palette (blue #1f6fb4, teal #167d78, orange #d9631e, red #c0392b, gray #8a94a3, navy #14213d), no title (the slide provides it), English labels, `np.random.seed(0)` and `torch.manual_seed(0)`. Every number printed on the figure is read from the notebook output, never typed by hand.

**S04-F03 · Reconstructions vs bottleneck size.** Grid of 5 rows × 8 columns of 28×28 grayscale images, `cmap='gray_r'`. Row labels on the left: "original", "AE, $k=2$", "AE, $k=8$", "AE, $k=32$", "PCA, $k=32$". The 8 test images are the first test example of 8 different classes (fixed indices saved by the notebook). On the right of each reconstruction row, two numbers in a small table column: "MSE" and "probe acc." (linear probe on $\mathbf h$, from `resultados_ae.json`). The point the figure must make: $k=2$ reconstructs worst but its probe is not proportionally worse; PCA and the AE at $k=32$ reconstruct similarly.

**S04-F05 · Two views per image.** 8 columns; each column has the original on top (gray frame) and the two views below it (teal frame for view 1, orange frame for view 2). Images upsampled with `interpolation='nearest'` to stay crisp. One image per class for 8 classes. Uses exactly the `transform` of notebook 4.4, with a fixed generator so the figure is reproducible. Left labels: "$\mathbf x$", "$\tilde{\mathbf x}_i$", "$\tilde{\mathbf x}_j$".

**S04-F07 · Similarity matrix.** Left two thirds: heatmap of $s_{ij}$ (cosine similarity divided by τ, before softmax) for $2N = 16$ views from the trained checkpoint, ordered as view 1 of images 1–8, then view 2 of images 1–8 (the order of `torch.cat([view1, view2])` in the notebook). Diagonal drawn as gray hatched cells labeled "masked". The 16 positive cells (the two off-diagonals at distance $N$) outlined in orange. Thumbnails of the 16 views along both axes. Diverging colormap centered at 0 (blue negative, red positive), colorbar labeled "$s_{ij}$". Right third: one row ($i=3$) shown as a bar chart of softmax probabilities over the 15 candidates, the positive bar in orange, with the caption "$L_i = -\log p_{i,j(i)}$" and the actual value of $L_i$.

**S04-F08 · Retrieval.** 5 rows; first column the query (navy frame), then the 6 nearest training images by cosine similarity of $\mathbf h$ (not $\mathbf u$). Frame of each neighbor teal if same class as the query, red if not; similarity printed under each one with two decimals. Queries chosen by fixed index, at least one where the neighbors share color or background but not class, so the figure also shows a failure.

**S04-F09 · Label efficiency.** Line plot, x axis "labels per class" on log scale with ticks 1, 10, 100, 5000; y axis "test accuracy (linear probe)" from 0 to 1. Four lines with markers: random $g$ (gray), autoencoder (blue), SimCLR (teal), ImageNet ResNet-18 (orange, dashed, labeled "ImageNet (1.2 M labels)"). Shaded band = min–max over 3 label samplings (seeds saved in the JSON). Horizontal dotted line at 0.10 labeled "chance". Direct labels at the right end of each line instead of a legend.

**S04-F10 · t-SNE.** Three panels side by side: "random $g$", "autoencoder", "SimCLR". 2 000 test images (same indices in the three panels), t-SNE with perplexity 30 and `random_state=0`, PCA to 50 dimensions first. Points colored by class with a 10-color qualitative palette, small markers, no axes or ticks. Shared legend at the bottom with the 10 class names. Small gray note at bottom right: "t-SNE keeps neighbors, not distances".

## Planned animations (Manim)

Only concepts where motion changes understanding.

| ID | Slide | Media slide title | What it shows | Rendered file | Status |
|---|---:|---|---|---|---|
| **S04-A01** | V20 | The latent space organizes itself | 2-D code of a Fashion-MNIST autoencoder (784 → 256 → 2 → 256 → 784, MSE, no labels) for 1 000 test images at 9 training steps (0 to 4 800): a shapeless cloud separates into groups. Points stay gray until the end; then colored by class, with "labels were never used" and the linear-probe accuracy on h (52 %). Data from `make_s04_a01_latent_snapshots.py` (same model as notebook 4.1, section 4) | `s04_a01_latent_space.mp4` | Included |
| **S04-A02** | V25 | Denoising pulls points back to the data | Points on a 1-D curve in 2-D; noise σ = 0.45 pushes them off; arrows send each corrupted point to the best possible denoiser output $E[\mathbf x\mid\tilde{\mathbf x}]$, which lands near the curve but not on its own clean point; then the field $d(g(\tilde{\mathbf x}))-\tilde{\mathbf x}$ on a grid for σ = 0.45 and σ = 1.3. The field is the exact optimum that a trained denoising autoencoder approximates, computed in the scene | `s04_a02_denoising_manifold.mp4` | Included |
| **S04-A03** | V33 | Masking in motion | The image of S03-A04 → 16 patches → 12 masked (visible patches as in S04-F04) → the large encoder takes only the 4 visible tokens → 16 slots (4 encoded + 12 [M]) → small decoder → output; loss outlined on the 12 masked patches only; decoder struck out at the end. The output is labeled as an illustration: biharmonic inpainting from the 4 visible patches stands in for a trained decoder | `s04_a03_masking.mp4` | Included |
| **S04-A04** | V46 | Pull and push on the sphere | Unit circle; 6 images, 2 views each. Forces on one anchor (pull toward its other view, push from the negatives), then 200 steps of exact gradient descent on InfoNCE with τ = 0.2: J goes from 5.08 to 0.29, views merge and images spread out. Second half: one anchor and 10 negatives; bars with each negative's share of the push while τ goes from 1.0 to 0.1. On screen: the embeddings move directly, in SimCLR the gradient goes into g and p | `s04_a04_contrastive_sphere.mp4` | Included |

Render from this folder (posters are the last frame):

```bash
python make_s04_a01_latent_snapshots.py                 # only for A01; writes s04_a01_latent_snapshots.npz
S01_VIDEO=1 manim -qh escenas_s04.py A01_LatentSpace    # linear video; without S01_VIDEO, manim-slides pauses
ffmpeg -sseof -0.1 -i s04_a01_latent_space.mp4 -frames:v 1 -update 1 s04_a01_latent_space_poster.png
```

## Production order

1. Generator figures, which need nothing else: F01, F02, F04, F06, F11.
2. Notebook 4.1, then F03 and A01 (they use its outputs).
3. Notebook 4.4 and the long SimCLR run for the checkpoint, then F05 and F07.
4. Notebook 4.5, then F08, F09 and F10.
5. Animations: done. A01 uses `make_s04_a01_latent_snapshots.py` (≈30 s on CPU); when notebook 4.1 exists, its section 4 must use the same model, seed and steps.
