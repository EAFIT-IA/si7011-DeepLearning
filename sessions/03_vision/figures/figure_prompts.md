# S03 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. For each figure, build the full prompt as: **common style block** + the figure's section.
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section. Do not deliver JPG.
4. Before delivering, check the image against the **review checklist** and the figure's own "Check" line. If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the current `../slides/S03_slides.marp.md`. Figures that show real data or a real model (S03-F02, F03, F06, F10, F11) are **not** in this file: they are produced by scripts, see `figures.md`.

## Status

| Figure | Concept | After slide | Status |
|---|---|---:|---|
| S03-F01 | Image as a C×H×W tensor | V07 | done (optional correction below) |
| S03-F04 | Multichannel convolution | V20 | done |
| S03-F05 | CNN anatomy as tensor blocks | V31 | done |
| S03-F15 | VGG-style block and stacked 3×3 | V34 | done (optional redraw of right panel) |
| S03-F07 | ResNet basic and downsampling blocks | V39 | done |
| S03-F08 | Pretrained backbone and a new head | V46 | done |
| S03-F09 | From scratch, linear probe, fine-tuning | V48 | done |
| S03-F12 | Vision Transformer architecture | V61 | pending |
| S03-F13 | Inductive bias versus data | V62 | pending |
| S03-F14 | What one unit can see: MLP, CNN, ViT | V67 | pending |

## Common style block (prepend to every prompt)

> Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. **No title at the top: the slide provides the title.** Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only. Tensor shapes are always written channels first, as $C \times H \times W$.
>
> Notation (use it exactly): activations $a_l$; channels $C$, height $H$, width $W$; kernel size $K$; depth (number of blocks) $D$; tokens $t_i$ of size $d$; positional embedding $p_i$; patch embedding matrix $W_E$; a pretrained backbone $g$. $\sigma$ means **sigmoid only**; never use $\sigma$ for a standard deviation.
>
> Palette: blue #1f6fb4, teal #167d78, orange #d9631e, red #c0392b, gray #8a94a3, navy text #14213d. Image channels: R #c0392b, G #2e8b57, B #1f6fb4.

## Review checklist (every image)

- Opens correctly and is 16:9 PNG.
- No title inside the image; no production codes such as "S03-F05".
- Every shape is written $C \times H \times W$ and matches the numbers in the prompt.
- Arrows point in the direction stated in the prompt.
- No logos, no photographs of real people, no reproduction of a figure from a paper.

---

## S03-F01 · Image as a C×H×W tensor (done, slide 7 — optional correction: shading and callout values)

File: `s03_f01_image_tensor.png`

**Left two thirds:** a small color image of a simple flat-illustration leaf (no photograph) shown as an 8 × 8 pixel grid on the far left, labeled "image". An arrow to the right labeled "split into channels" leads to **three stacked, slightly offset 8 × 8 grids** in perspective, front to back: red grid labeled "R", green grid labeled "G", blue grid labeled "B". Cell shading in each grid follows the leaf's channel intensity. Brace along the stack labeled $C = 3$; brace along the vertical edge labeled $H$; brace along the horizontal edge labeled $W$.

One position is highlighted in all three grids with a thin orange outline at row 3, column 5, joined by an orange line that pierces the stack. The highlighted position is inside the leaf. A callout from that line shows a column vector $\begin{bmatrix}0.18\\0.62\\0.21\end{bmatrix}$ (low R, high G, low B: a green pixel) with the label "one position, all channels".

**Shading rule (important):** in each channel grid, the cell color intensity equals that channel's value. White background pixels have value 1 in every channel, so the background is the **most saturated** color in all three grids. The leaf is light in R and B and saturated in G.

Under the stack, centered: $x \in \mathbb{R}^{C \times H \times W}$.

**Right third:** four such stacks drawn small, side by side in a row, with a brace labeled $N$ and the caption $N \times C \times H \times W$ and the small text "a batch".

Check: exactly three channel grids, in the order R, G, B from front to back; the shape text reads $C \times H \times W$, not $H \times W \times C$; background saturated in all three grids, leaf saturated only in G; the callout's largest value is the G entry.

---

## S03-F04 · Multichannel convolution (done, slide 20)

File: `s03_f04_multichannel_conv.png`

Three stages from left to right, connected by arrows.

1. **Input:** a block of three stacked channels (R, G, B colors), labeled $C_\text{in} \times H \times W$ with the example "$3 \times 32 \times 32$" below in gray.
2. **One kernel:** a small cube of three stacked $3 \times 3$ slices, in the same R, G, B colors, labeled "one kernel: $C_\text{in} \times K \times K$" and "$3 \times 3 \times 3$" in gray. A dashed teal outline marks the $3 \times 3$ window on the input block where the kernel currently sits, spanning all three channels. Next to it, the text: "multiply and add over all $C_\text{in}$ channels → one number".
3. **Output:** first, a single teal feature map labeled "one map"; then, below or after it, a stack of $C_\text{out}$ maps in alternating teal and blue, labeled $C_\text{out} \times H \times W$ with the example "$64 \times 32 \times 32$" in gray, and a brace labeled "$C_\text{out}$ kernels → $C_\text{out}$ maps".

**Bottom banner:** "$\#\text{params} = C_\text{out}\,C_\text{in}\,K^2 + C_\text{out} = 64 \cdot 3 \cdot 9 + 64 = 1792$".

Check: the kernel has as many slices as the input has channels (3); the output has 64 maps; padding keeps $32 \times 32$.

---

## S03-F05 · CNN anatomy as tensor blocks (done, slide 31)

File: `s03_f05_cnn_anatomy.png`

A left-to-right pipeline of 3-D blocks whose **face size shrinks** (spatial size) and **depth grows** (channels). Each block is labeled with its shape below it:

1. Input image block (R, G, B colors): $3 \times 32 \times 32$.
2. After the stem (blue): $32 \times 32 \times 32$. Arrow label above: "stem: conv $3\times3$".
3. After stage 1 (blue, deeper, smaller face): $64 \times 16 \times 16$. Arrow label: "stage 1: conv–BN–ReLU blocks, ↓2".
4. After stage 2 (blue, deeper still, smaller face): $128 \times 8 \times 8$. Arrow label: "stage 2: conv–BN–ReLU blocks, ↓2".
5. A thin vertical bar (teal) labeled $128$. Arrow label: "global average pool".
6. A short orange bar labeled $10$, then the word "logits". Arrow label: "linear".

Above the whole pipeline, two thin trend arrows spanning stages: a gray arrow going down labeled "resolution: 32 → 16 → 8" and a blue arrow going up labeled "channels: 32 → 64 → 128".

Check: shapes are exactly $3\times32\times32$, $32\times32\times32$, $64\times16\times16$, $128\times8\times8$, $128$, $10$; block faces visibly shrink while depth grows.

---

## S03-F15 · VGG-style block and stacked 3×3 (done, slide 34 — optional redraw of the receptive-field sketch)

File: `s03_f15_vgg_block.png`

Two panels side by side, same visual style as the ResNet figure (S03-F07): rounded panels, light blue operation boxes, a vertical flow from top to bottom.

**Left panel, blue header "VGG-style block":** input $a_l$ with shape $64 \times 32 \times 32$. Boxes, top to bottom: "conv $3\times3$, 64" → "ReLU" → "conv $3\times3$, 64" → "ReLU" → "max pool $2\times2$". Output $a_{l+1}$ with shape $64 \times 16 \times 16$. **No shortcut line**: a small gray note on the right side of the boxes: "plain: no shortcut".

**Right panel, teal header "why stack two 3 × 3?":** three small square grids drawn left to right with arrows between them, as a 2-D receptive-field view:

1. a $5 \times 5$ gray input grid with a highlighted $5 \times 5$ light-teal region covering all of it, labeled "input";
2. a $3 \times 3$ grid labeled "after conv 1": one cell outlined in teal, with thin teal lines fanning down to a $3 \times 3$ patch of the input grid;
3. a single orange cell labeled "after conv 2": thin orange lines fanning down to all 9 cells of the middle grid.

Under the grids, the caption "two $3\times3$ layers see $5\times5$".

Below that, two parameter boxes side by side:

- left box (teal border): "two $3\times3$:" and $2 \cdot 9C^2 = 18C^2$, plus "+ one extra ReLU";
- right box (gray border): "one $5\times5$:" and $25C^2$.

A "<" sign between the two boxes.

**Optional redraw, 2-D (fixes the delivered version):** the dashed teal lines go from the four corners of the $5 \times 5$ input grid to the four corners of the **whole** $3 \times 3$ "after conv 1" grid, never to a single cell: one conv-1 cell sees only $3 \times 3$; all nine conv-1 cells together see $5 \times 5$. No cell of the conv-1 grid is highlighted. The orange cell of "after conv 2" keeps its 9 lines to the conv-1 grid. (A second attempt drew the lines to the centre conv-1 cell, which says one conv-1 unit sees $5\times5$: wrong, rejected.)

**Optional redraw of the receptive-field sketch (1-D, clearer):** replace the three grids with three horizontal rows of square cells, bottom to top: "input" with 5 cells, "after conv 1" with 3 cells, "after conv 2" with 1 orange cell. Each conv-1 cell has 3 thin teal lines to the 3 input cells below it (cells 1–3, 2–4, 3–5). The orange cell has 3 thin orange lines to all 3 conv-1 cells. All 5 input cells are shaded light teal. Caption: "two $3\times3$ layers see $5\times5$" (in 2-D the same holds per axis).

Check: the left block has no line going around the boxes; the output is $64 \times 16 \times 16$ (only the pool halves the size); the receptive field of the second layer is $5\times5$; the numbers are exactly $18C^2$ and $25C^2$.

---

## S03-F07 · ResNet basic and downsampling blocks (done, slide 39)

File: `s03_f07_resnet_blocks.png`

Two panels side by side, each a vertical flow from input (top) to output (bottom).

**Left panel, teal tag "basic block":** input $a_l$ with shape $64 \times 56 \times 56$. Main path (blue boxes, top to bottom): "conv $3\times3$, 64" → "BN" → "ReLU" → "conv $3\times3$, 64" → "BN". A teal curved line on the right side goes from the input directly to a circled "+" at the bottom of the main path, labeled "identity". After the "+": "ReLU", then output $a_{l+1}$ with shape $64 \times 56 \times 56$.

**Right panel, orange tag "downsampling block":** input $a_l$ with shape $64 \times 56 \times 56$. Main path: "conv $3\times3$, 128, stride 2" → "BN" → "ReLU" → "conv $3\times3$, 128" → "BN". The shortcut on the right side is an orange box "conv $1\times1$, 128, stride 2" → "BN", joining the circled "+". After the "+": "ReLU", then output $a_{l+1}$ with shape $128 \times 28 \times 28$. Small note next to the shortcut: "shapes must match to add".

Check: left output shape equals input shape; right output is $128 \times 28 \times 28$; both shortcuts end in the same "+" as the main path, before the final ReLU.

---

## S03-F08 · Pretrained backbone and a new head (done, slide 46)

File: `s03_f08_transfer_backbone.png`

**Top row, label "pretraining on ImageNet":** a horizontal chain of five blocks: "stem", "stage 1", "stage 2", "stage 3", "stage 4", then a gray head box "linear → 1000 classes". The stage blocks are filled with a gradient from teal (stem) to blue (stage 4).

Below the chain, a horizontal arrow spanning the stages, from left to right, labeled at the left end "general: edges, colors, textures" and at the right end "specific: object parts of the 1000 classes".

**Bottom row, label "new task: 37 pet breeds":** the same five blocks (same colors), with a dashed arrow from the top row labeled "copy weights". The gray 1000-class head is shown crossed out to the side, and a new orange head box "linear → 37 classes" is attached, with the tag "new, random init".

Check: the backbone blocks are identical in both rows; only the head changes; the head output counts are 1000 and 37.

---

## S03-F09 · From scratch, linear probe, fine-tuning (done, slide 48)

File: `s03_f09_transfer_strategies.png`

Three horizontal rows, each the same chain "stem → stage 1 → stage 2 → stage 3 → stage 4 → head". A legend at the bottom: orange = "trained, random init"; gray with a small lock icon = "frozen, pretrained"; blue = "trained, starts from pretrained, smaller $\eta$".

1. **Row tag "from scratch":** all six boxes orange. Right-side note: "all weights from random".
2. **Row tag "linear probe":** five backbone boxes gray with lock icons; head orange. Right-side note: "only the head learns; features $g(x)$ can be cached".
3. **Row tag "fine-tuning":** stem and stage 1 gray with locks; stages 2–4 blue; head orange. Right-side note: "head with larger $\eta$, backbone with smaller $\eta$".

Check: exactly three rows in this order; in "linear probe" no backbone box is colored; in "fine-tuning" the late stages are blue, not orange.

---

## S03-F12 · Vision Transformer architecture (pending, after slide 61)

File: `s03_f12_vit_architecture.png`

**Left:** a simple flat illustration of a dog (not a photograph) as a square image, labeled $3 \times 224 \times 224$, overlaid with a $14 \times 14$ grid of thin gray lines. Arrow labeled "196 patches of $16 \times 16$".

**Center-left:** a row of small squares (patches) flattened into a row of vertical bars, then an arrow through a box "$W_E$ (= conv $16\times16$, stride 16)" into a row of token bars $t_1, t_2, \dots, t_{196}$, each of size $d$. Above each token a small "+" with a gray bar $p_i$ labeled "position". At the start of the row, an extra orange token labeled $[\texttt{CLS}]$.

**Center-right:** a tall rounded panel labeled "$\times D$ blocks" containing, bottom to top: "LN" → "attention" → circled "+" (with a teal skip line around LN and attention), then "LN" → "MLP" → circled "+" (with a teal skip line around them).

**Right:** an arrow from the $[\texttt{CLS}]$ output at the top of the panel to an orange box "head" and the label "class scores".

Small gray note under the panel: "ViT-B/16: $d = 768$, $D = 12$".

Check: 196 tokens plus one CLS token; the norm comes **before** attention and before the MLP (pre-norm); only the CLS output feeds the head.

---

## S03-F13 · Inductive bias versus data (pending, after slide 62)

File: `s03_f13_inductive_bias_spectrum.png`

A single schematic plot. x-axis: "pretraining / training images (log scale)", with ticks $10^3$, $10^5$, $10^7$, $10^9$. y-axis: "test accuracy", no numbers. No grid.

Three smooth curves:

- **CNN** (blue, solid): rises early and is the highest curve at the left; keeps improving slowly at the right.
- **ViT** (orange, solid): starts below the CNN, rises more steeply, crosses the CNN between $10^7$ and $10^8$, and ends highest at the right.
- **MLP** (gray, solid): the lowest curve everywhere, rising slowly.

A light blue shaded region at the left, labeled "the CNN's assumptions win". A light orange shaded region at the right, labeled "enough data to learn the assumptions". A small dashed circle at the crossing point.

Below the plot, three tags aligned left to right: "CNN: assumes locality and translation equivariance", "ViT: learns them from data", "MLP: assumes nothing about position".

Gray note in the bottom right corner: "schematic, not measured".

Check: the ViT crosses the CNN only once, at large data; the MLP never crosses either.

---

## S03-F14 · What one unit can see: MLP, CNN, ViT (pending, after slide 67)

File: `s03_f14_architecture_summary.png`

Three columns with headers "MLP", "CNN", "ViT". In each column, the same $8 \times 8$ gray input grid at the bottom and one highlighted unit (orange dot) above it.

1. **MLP:** the orange unit has thin gray lines to **all 64** input cells. Caption below: "every input, its own weight, no notion of neighbor".
2. **CNN:** two layers. A unit in layer 1 connects to a $3 \times 3$ teal window of the input; a unit in layer 2 connects to a $3 \times 3$ window of layer 1, and a lighter teal $5 \times 5$ region on the input shows what it sees. Caption: "a neighborhood, growing with depth".
3. **ViT:** the input grid is split into a $4 \times 4$ grid of patches (each $2 \times 2$ cells). One patch token (orange) has blue lines to **all 16** patches, with line thickness varying (three thick, the rest thin). Caption: "every patch, weighted by content".

Bottom banner across the three columns: "What each architecture assumes decides what it must learn from data."

Check: the CNN regions are $3 \times 3$ and $5 \times 5$; the ViT has 16 patches; the MLP unit connects to all 64 cells.
