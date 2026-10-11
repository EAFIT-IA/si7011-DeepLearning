# S04 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. For each figure, build the full prompt as: **common style block** + the figure's section.
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section. Do not deliver JPG.
4. Before delivering, check the image against the **review checklist** and the figure's own "Check" line. If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the slide-by-slide plan, `../S04_plan.md`. Figures that show real data or a trained model (S04-F03, F05, F07, F08, F09, F10) are **not** in this file: they are produced by scripts, see `figures.md`.

## Status

| Figure | Concept | Slide | Status |
|---|---|---:|---|
| S04-F01 | Supervised vs self-supervised pipeline | V09 | done (see review note in figures.md) |
| S04-F02 | Autoencoder with a bottleneck | V16 | done |
| S04-F04 | Masked autoencoder pipeline | V32 | done, patched by script |
| S04-F06 | SimCLR: two views, shared encoder, projection head | V42 | done, patched by script |
| S04-F11 | Four objectives, one encoder | V65 | done |

## Common style block (prepend to every prompt)

> Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. **No title at the top: the slide provides the title.** Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only. Images inside the figure are simple flat illustrations, never photographs.
>
> Notation (use it exactly): input $\mathbf x$; corrupted input $\tilde{\mathbf x}$; encoder $g$; decoder $d$; representation $\mathbf h = g(\mathbf x)$; reconstruction $\hat{\mathbf x}$; projection head $p$; projected embedding $\mathbf u = p(\mathbf h)$; similarity $s_{ij}$; temperature $\tau$; per-example loss $L_i$; batch of $N$ images. Never use $z$ for the representation, and never use $\sigma$ for anything except the sigmoid. Write the loss as a plain italic $L$ ($L_i$), never calligraphic $\mathcal L$.
>
> Palette: blue #1f6fb4 (encoder $g$), teal #167d78 (decoder $d$, projection head $p$), orange #d9631e (bottleneck, loss, positives), red #c0392b (errors), gray #8a94a3 (discarded or inactive parts), navy text #14213d. The encoder $g$ is **always blue** in every figure of this session.

## Review checklist (every image)

- Opens correctly and is 16:9 PNG.
- No title inside the image; no production codes such as "S04-F02".
- The encoder $g$ is blue; the representation is written $\mathbf h$, never $z$.
- Arrows point in the direction stated in the prompt.
- No logos, no photographs, no reproduction of a figure from a paper (in particular, not the SimCLR or MAE paper figures).

---

## S04-F01 · Supervised vs self-supervised pipeline (done, slide 9)

File: `s04_f01_supervised_vs_selfsupervised.png`

Two horizontal rows, each inside a rounded panel, left to right. Same encoder box in both rows.

**Top row, gray tag "supervised":**
1. Three small flat-illustration images stacked (a boat, a dog, a leaf), labeled $\mathbf x$.
2. A simple generic person icon (no face details) with a speech label "dog", labeled "annotator" and below it, in gray, "labels: costly".
3. Blue box "$g$" → small gray box "head" → $\hat y$. The images $\mathbf x$ go straight into $g$; the annotator is drawn above this path and only sends $y$ to the loss.
4. Orange circle "$L_i(\hat y, y)$", with the label $y$ coming from the annotator by a thin arrow.

**Bottom row, teal tag "self-supervised":**
1. The same dog image, labeled $\mathbf x$.
2. Arrow labeled "hide a part" to the same image with its right half covered by a gray block, labeled $\tilde{\mathbf x}$.
3. Blue box "$g$" → small gray box "pretext head" → the predicted hidden half.
4. Orange circle "$L_i$". A thin orange curved arrow goes from the original $\mathbf x$ (the hidden right half, outlined) to the loss, labeled "target comes from $\mathbf x$". Below the row, in gray: "labels: free".

**Right side, spanning both rows, a narrow panel "after pretraining":** the blue "$g$" box with an arrow to $\mathbf h$, labeled "kept"; next to it the "pretext head" box in gray with a single diagonal strike, labeled "discarded".

Check: there is no annotator in the bottom row; the encoder box is the same blue in both rows; the bottom-row target is a part of $\mathbf x$, not a word label; only the pretext head is struck out.

---

## S04-F02 · Autoencoder with a bottleneck (done, slide 16)

File: `s04_f02_autoencoder.png`

A left-to-right "hourglass" lying on its side.

1. **Input:** a 28 × 28 grayscale flat illustration of a T-shirt (Fashion-MNIST style, simple shapes), labeled $\mathbf x$ and, below, $\mathbf x \in \mathbb R^{784}$.
2. **Encoder:** a blue trapezoid narrowing from left to right, with three vertical layer bars inside labeled $784$, $256$ and $k$. Brace below: "encoder $g$".
3. **Bottleneck:** a narrow orange vertical bar labeled $\mathbf h \in \mathbb R^{k}$, with "bottleneck" above it and "$k = 32 \ll 784$" below it.
4. **Decoder:** a teal trapezoid widening from left to right, with layer bars labeled $k$, $256$ and $784$. Brace below: "decoder $d$".
5. **Output:** the same T-shirt, slightly blurred, labeled $\hat{\mathbf x} = d(g(\mathbf x))$.

Under the whole hourglass, centered: $L_i = \lVert \mathbf x_i - \hat{\mathbf x}_i \rVert^2$, with a thin orange arc joining $\mathbf x$ and $\hat{\mathbf x}$.

**Bottom-right corner, small gray inset** with a dashed border titled "the identity trap": the input box connected straight to the output box by a gray arrow labeled "$g = d = \text{identity}$", and the text "$L = 0$, nothing learned (if $k \ge 784$)".

Check: layer sizes are exactly 784 → 256 → $k$ → 256 → 784; the bottleneck is the narrowest point; the reconstruction is visibly blurrier than the input; the inset is clearly secondary.

---

## S04-F04 · Masked autoencoder pipeline (done, slide 32)

File: `s04_f04_mae_pipeline.png`

Left to right, five stages connected by arrows.

1. **Input:** a flat-illustration landscape (sun, hill, one tree, blue sky) divided by thin white lines into a **4 × 4 grid of 16 patches**, labeled $\mathbf x$, "16 patches".
2. **Masking:** the same grid with **12 patches gray** and **4 patches visible**, at (row, column) = (1, 2), (2, 4), (3, 1), (4, 3). Label above: "mask 75 %". Label below: "keep 4 of 16".
3. **Encoder:** the 4 visible patches as 4 small token squares in a column entering a **large** blue box "$g$ (ViT encoder)", which outputs 4 blue tokens $\mathbf h_i$. Small gray note: "sees only visible patches".
4. **Decoder:** a column of 16 token squares: the 4 blue ones (encoded tokens labeled $\mathbf h_i$, not image patches) at their original positions and 12 light-gray squares labeled "[M]", with a brace "mask tokens + positions". They enter a **small** teal box "$d$ (light decoder)". The size difference between $g$ (large) and $d$ (small) must be obvious.
5. **Output:** a 4 × 4 grid of predicted patches forming the reconstructed landscape, slightly blurred. The 12 previously masked patches have an orange outline; the 4 visible ones have none. Count them: exactly 12 outlined, exactly 4 without outline, at the visible positions.

Below stage 5: "loss only on masked patches: $L_i = \frac{1}{|M|}\sum_{m \in M} \lVert \mathbf x_m - \hat{\mathbf x}_m \rVert^2$".

Bottom strip, gray, full width: "after pretraining: keep $g$, discard $d$".

Check: exactly 16 patches, exactly 4 visible at the stated positions; the encoder receives 4 tokens, not 16; the decoder receives 16; orange outlines only on the 12 masked patches.

---

## S04-F06 · SimCLR: two views, shared encoder, projection head (done, slide 42)

File: `s04_f06_simclr.png`

A symmetric diagram, two parallel horizontal branches (upper and lower) that start from one image and meet at a similarity on the right.

1. **Left:** one flat-illustration image of a small bird on a branch, labeled $\mathbf x$.
2. Two arrows out of it: upper arrow labeled $t \sim \mathcal T$, lower arrow labeled $t' \sim \mathcal T$.
3. **Views:** upper view $\tilde{\mathbf x}_i$ = a zoomed-in crop of the bird with shifted colors; lower view $\tilde{\mathbf x}_j$ = the whole image flipped horizontally and in grayscale. **Both views must show exactly the same bird as $\mathbf x$** (same pose, same branch); only crop, color and flip change.
4. **Encoder:** a blue box "$g$" on each branch. A dashed vertical line between the two boxes labeled "shared weights".
5. **Representation:** a short vertical vector on each branch, $\mathbf h_i$ and $\mathbf h_j$, with "$\in \mathbb R^{512}$" under the upper one.
6. **Projection head:** a small teal box "$p$" on each branch, with "Linear – ReLU – Linear" written small inside or under it.
7. **Projected embedding:** shorter vectors $\mathbf u_i$ and $\mathbf u_j$, with "$\in \mathbb R^{128}$" under the upper one.
8. **Right:** both branches meet at an orange circle "$s_{ij}$" with the label "maximize agreement".

**Below the lower branch, a gray rounded box** under the similarity: "the other $2N - 2$ views in the batch are negatives".

**From $\mathbf h_i$, a teal arrow going upward** out of the upper branch to a small label "downstream tasks use $\mathbf h$". Above the upper $p$ box, small gray text: "$p$: used only for training".

Check: the two encoder boxes are identical and blue, joined by "shared weights"; $\mathbf h$ comes before $p$ and $\mathbf u$ after it; dimensions 512 and 128; the downstream arrow leaves from $\mathbf h$, not from $\mathbf u$; the views differ by crop/color and by flip/grayscale.

---

## S04-F11 · Four objectives, one encoder (done, slide 65)

File: `s04_f11_objectives_summary.png`

**Center:** a blue rounded box "$g$" with $\mathbf x$ entering from the left and $\mathbf h$ leaving to the right, a little larger than the other elements.

**Four panels in the four corners**, each connected to the central box by a thin gray line, each with a colored tag, a one-line sketch, a formula and a "keeps:" line:

1. **Top left, tag "reconstruct" (autoencoder):** sketch $\mathbf x \to g \to d \to \hat{\mathbf x}$. Formula $\lVert \mathbf x - \hat{\mathbf x} \rVert^2$. "keeps: everything needed to redraw the pixels, texture included".
2. **Top right, tag "denoise":** sketch $\tilde{\mathbf x} = \mathbf x + \boldsymbol\varepsilon \to g \to d \to \hat{\mathbf x}$. Formula $\lVert \mathbf x - d(g(\tilde{\mathbf x})) \rVert^2$. "keeps: the structure of the data, not the noise".
3. **Bottom left, tag "fill in" (masked):** sketch of a 4 × 4 patch grid with most patches gray → predicted patches. Formula $\sum_{m\in M}\lVert \mathbf x_m - \hat{\mathbf x}_m\rVert^2$. "keeps: how the parts of an input relate".
4. **Bottom right, tag "contrast":** sketch of two views → two arrows → one point on a small circle. Formula "InfoNCE over $s_{ij}$". "keeps: what the augmentations do not change".

**Bottom strip, full width, gray:** "the pretext task is discarded · $g$ is kept · measure it with $k$-NN, linear probe, fine-tuning".

Check: one single encoder in the center, blue; four panels, with the labels and formulas exactly as given; the word "keeps:" appears in all four panels.
