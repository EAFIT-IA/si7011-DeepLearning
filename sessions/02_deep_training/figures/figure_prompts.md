# S02 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. For each figure, build the full prompt as: **common style block** + the figure's section.
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section.
   Do not deliver JPG (earlier JPGs arrived truncated).
4. Before delivering, check the image against the **review checklist** and the figure's own "Check" line.
   If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the current `../slides/S02_slides.marp.md`.

## Status

| Figure | Concept | Status |
|---|---|---|
| S02-F06 | Symmetry problem with identical initialization | done |
| S02-F11 | BatchNorm: normalize, then learn scale and shift | done |
| S02-F12 | BatchNorm in training versus evaluation mode | done |
| S02-F13 | Underfitting, healthy fit, overfitting | done |
| S02-F14 | Effect of weight decay | done |
| S02-F15 | Dropout as stochastic subnetworks | done |
| S02-F16 | Depth degradation in plain networks | done |
| S02-F17 | Residual block | done |
| S02-F19 | Training diagnostics map | done |
| S02-F20 | Why deep training works: integrated view | done |
| S02-F22 | Mini-batch gradient noise versus batch size | done |
| S02-F23 | BatchNorm versus LayerNorm: which axis is normalized | done |
| S02-F24 | Label-preserving data augmentation | done |
| S02-F01 | Correction: typo and logo fixed; script 𝒥 remains (optional regeneration) | done |
| S02-F02 | Correction: std(x) label and consistent numbers | **pending** |
| S02-F04 | Correction: backward direction and notation | **pending** |
| S02-F07 | Correction: Xavier variance, tanh vs ReLU | **pending** |

## Common style block (prepend to every prompt)

> Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. **No title at the top: the slide provides the title.** Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only.
>
> Notation (use it exactly): layers $l = 1,\dots,D$; pre-activation $z_l = W_l a_{l-1} + b_l$; activation $a_l = \phi(z_l)$ with $a_0 = x$; per-example loss $L$; objective $J$; $\sigma$ means **sigmoid only**, never a standard deviation (write std(·)).
>
> Palette: blue #1f6fb4, teal #167d78, orange #d9631e, red #c0392b, gray #8a94a3, navy text #14213d.

## Review checklist (every image)

- Opens correctly and is 16:9 (PNG preferred; avoid JPG).
- No title inside the image; no production codes such as "S02-F11".
- Notation matches the block above ($a_l$, $J$, $D$; no $h^{(l)}$, no $\mathcal L$, no $L$ for depth).
- Numbers and arrows point in the direction stated in the prompt.

---

## S02-F06 · Symmetry problem with identical initialization (done, slide 24)

File: `s02_f06_initialization_symmetry.png`

Two side-by-side panels, each showing the same tiny network: 2 inputs $x_1, x_2$ → 2 hidden units $h_1, h_2$ → 1 output $\hat y$.

**Left panel, red tag "identical initialization":** both hidden units drawn in the same red color, every incoming weight labeled $w = 0.5$, both outgoing weights labeled $v = 0.5$. Below the network, three stacked rows with small equal-sign badges:
1. "forward: $a_1 = a_2$"
2. "backward: $\partial J/\partial W_{1,:} = \partial J/\partial W_{2,:}$"
3. "after the update: still $W_{1,:} = W_{2,:}$"

A small looping arrow beside the rows with the text "every step: the two units stay copies of each other".

**Right panel, green tag "random initialization":** hidden units in two different colors (blue, teal), incoming weights with different small values (e.g. 0.31, −0.12, 0.08, 0.44). Rows:
1. "forward: $a_1 \neq a_2$"
2. "backward: different gradients"
3. "the units learn different features"

**Bottom banner:** "Identical units compute the same value and receive the same gradient, so they never diverge. Random initialization breaks the symmetry."

---

## S02-F11 · BatchNorm: normalize, then learn scale and shift (done)

File: `s02_f11_batchnorm_transform.png`

Three panels left to right, connected by two large arrows. Each panel shows **three small histograms stacked vertically**, one per feature: "feature 1" in blue (#1f6fb4), "feature 2" in teal (#167d78), "feature 3" in orange (#d9631e). Same colors and same vertical order in all three panels. Each histogram represents the values of that feature across one mini-batch of 64 examples.

**Panel 1, header "pre-activations $z$ in the batch":** the three histograms have very different centers and widths, all on a shared x-axis from −10 to 10:
- feature 1: centered at 5, medium width (std ≈ 2)
- feature 2: centered at −1, very narrow (std ≈ 0.3)
- feature 3: centered at 0.5, very wide (std ≈ 6)

Each histogram has a thin vertical dashed line at its mean, labeled $\mu_{\mathcal B,1}$, $\mu_{\mathcal B,2}$, $\mu_{\mathcal B,3}$.

**Arrow 1 label:** $\hat z = \dfrac{z - \mu_{\mathcal B}}{\sqrt{s_{\mathcal B}^2 + \epsilon}}$

**Panel 2, header "normalized $\hat z$":** all three histograms identical in position and width: centered at 0, std 1, on a shared x-axis from −3 to 3. Small gray text below: "mean 0 and variance 1 for every feature".

**Arrow 2 label:** $\mathrm{BN}(z) = \gamma \odot \hat z + \beta$

**Panel 3, header "learned scale $\gamma$ and shift $\beta$":** each histogram shifted and stretched differently again, on a shared x-axis from −3 to 5, with a small label beside each:
- feature 1: "$\gamma_1 = 1.5,\ \beta_1 = 0.5$" (centered at 0.5, wider)
- feature 2: "$\gamma_2 = 0.7,\ \beta_2 = -0.3$" (centered at −0.3, narrower)
- feature 3: "$\gamma_3 = 1.0,\ \beta_3 = 1.2$" (centered at 1.2, same width as in panel 2)

**Bottom banner:** "Statistics are computed per feature, across the batch. The learned $\gamma$ and $\beta$ let the network recover any scale and offset it needs."

**Check:** in panel 2 the three histograms are identical (same center, same width); the square root covers only $s_{\mathcal B}^2 + \epsilon$.

---

## S02-F12 · BatchNorm in training versus evaluation mode (done)

File: `s02_f12_batchnorm_train_eval.png`

Two panels side by side.

**Left panel, blue header `model.train()`:** a mini-batch drawn as a stack of 4 example cards, one of them highlighted and labeled "example $x$". Arrows feed the whole batch into a box "compute $\mu_{\mathcal B},\ s^2_{\mathcal B}$ from this batch", then into "normalize". Beside it, a small side box "running estimates" with the update rule:
$\hat\mu \leftarrow (1 - m)\,\hat\mu + m\,\mu_{\mathcal B}$, $\quad \hat s^2 \leftarrow (1 - m)\,\hat s^2 + m\,s^2_{\mathcal B}$, $\quad m = 0.1$ (PyTorch default).

Under the panel: "the output for $x$ depends on the other examples in the batch". Show two different batches containing the same $x$ giving slightly different outputs: $\mathrm{BN}(x) = 0.83$ and $\mathrm{BN}(x) = 0.71$.

**Right panel, green header `model.eval()`:** a single example card $x$ goes into a box "use stored $\hat\mu,\ \hat s^2$", then "normalize". Under the panel: "deterministic: the same $x$ always gives the same output", with a single output $\mathrm{BN}(x) = 0.78$.

**Bottom banner (orange outline):** "Forgetting `model.eval()` at validation or test time is a common bug: predictions change with the batch."

---

## S02-F13 · Underfitting, healthy fit, overfitting (done)

File: `s02_f13_train_validation_dynamics.png`

Three plots side by side, each with x-axis "epoch" (0 to 100) and y-axis "objective $J$" (0 to 1.0), no grid. Two curves per plot: $J_\text{train}$ solid blue, $J_\text{val}$ dashed orange. Header tags above each plot.

1. **Red tag "underfitting":** both curves decrease slowly and plateau high, near 0.6 and 0.65, very close together. Annotation: "both high: not enough capacity or training".
2. **Green tag "healthy fit":** both curves decrease and flatten low ($J_\text{train} \approx 0.12$, $J_\text{val} \approx 0.18$), with a small gap. A small bracket between the curves labeled "small gap".
3. **Orange tag "overfitting":** $J_\text{train}$ keeps decreasing toward 0.02. $J_\text{val}$ decreases until epoch ≈ 30 (minimum ≈ 0.25), then rises to ≈ 0.55 by epoch 100. A vertical dashed gray line at the validation minimum labeled "best epoch (early stopping)". A bracket between the curves at epoch 100 labeled "gap $= J_\text{val} - J_\text{train}$".

**Bottom banner:** "Read both curves together: the gap, not the training loss alone, tells you about generalization."

---

## S02-F14 · Effect of weight decay (done)

File: `s02_f14_weight_decay_effect.png`

Top row: three plots of the same 1D regression task, x from −1 to 1. Each shows the same 15 noisy data points (navy dots) sampled around a smooth sine-like curve (thin gray dashed line, labeled "true function" in the first plot only). The learned function is drawn as a thick colored curve.

1. **Tag "$\lambda = 0$":** orange curve that wiggles strongly and passes through almost every noisy point. Caption: "fits the noise".
2. **Tag "$\lambda$ moderate":** teal curve smooth and close to the true function. Caption: "smooth, generalizes".
3. **Tag "$\lambda$ too large":** blue curve almost flat. Caption: "too constrained: underfits".

Bottom row, spanning the width: a small bar chart "$\lVert \theta \rVert_2$ after training" with three bars matching the three plots (orange tall, teal medium, blue short).

Small formula at the bottom left: $J_\lambda(\theta) = J(\theta) + \lambda \lVert \theta \rVert_2^2$.

**Bottom banner:** "Weight decay prefers small weights; small weights give smoother functions."

---

## S02-F15 · Dropout as stochastic subnetworks (done)


File: `s02_f15_dropout_subnetworks.png`

Four small copies of the same MLP (4 inputs → 6 hidden → 6 hidden → 2 outputs) in a row.

The first three copies are labeled "training step $t$", "step $t+1$", "step $t+2$". In each, a **different random half** of the hidden units is dropped: dropped units drawn as light gray circles with a small gray ×, and their connections removed. The remaining units are blue, with their connections drawn. Under each copy a tiny mask vector, e.g. $r = (1,0,1,1,0,0)$, different in each copy. Header over these three: "training: a new random mask $r$ every step, $p = 0.5$".

The fourth copy, separated by a thin vertical divider, labeled "inference (`model.eval()`)": all units active and blue, all connections drawn.

Formulas under the divider:
- left side: $\tilde a = \dfrac{r \odot a}{1-p}$, $\quad r_i \sim \mathrm{Bernoulli}(1-p)$
- right side: "no mask, no rescaling"

**Bottom banner:** "Each step trains a different thinned network; at inference the full network approximates their average. In `nn.Dropout(p)`, $p$ is the probability of dropping a unit."

---

## S02-F16 · Depth degradation in plain networks (done)

File: `s02_f16_depth_degradation.png`

Two plots side by side with the same axes: x-axis "iterations (×10⁴)" from 0 to 6, y-axis "error (%)" from 0 to 20. Each plot has two curves: "20-layer plain" in blue and "56-layer plain" in orange.

1. **Left plot, header "training error":** both curves decrease, but the **orange 56-layer curve stays above the blue 20-layer curve the whole time**, ending around 7 % vs 4 %.
2. **Right plot, header "test error":** same ordering, orange above blue, ending around 13 % vs 9 %.

A callout pointing at the left plot: "the deeper network is worse even on the **training** data → this is not overfitting, it is an optimization problem".

Small gray footnote at the bottom right: "Schematic after He et al., 2016 (CIFAR-10); values illustrative."

**Bottom banner:** "A deeper plain network could copy the shallower one by learning identity layers, but gradient descent does not find that solution."

---

## S02-F17 · Residual block (done)

File: `s02_f17_residual_block.png`

Large, clear diagram, horizontal, left to right.

Input label $a_{l-1}$ on the left. The main path goes through three boxes: blue "Affine $W_1, b_1$" → green "$\phi$" → blue "Affine $W_2, b_2$", grouped under a brace labeled $F(a_{l-1})$, then into a circle with "+". An **identity path** leaves $a_{l-1}$ before the first box, arcs **above** the main path as a thick gray line labeled "identity (skip connection)", and enters the "+" from the top. The output of "+" is labeled $a_l$.

Under the diagram, centered: $a_l = a_{l-1} + F(a_{l-1})$.

Bottom-right inset, small: three such blocks stacked left to right with their skip arcs, labeled "a ResNet stacks many blocks".

Bottom-left note, small gray: "requires $F(a_{l-1})$ and $a_{l-1}$ to have the same shape".

**Bottom banner:** "If the best block is close to the identity, $F$ only has to learn a small correction."

---

## S02-F19 · Training diagnostics map (done)

File: `s02_f19_training_diagnostics_map.png`

A three-column flow map with five rows. Column headers: "symptom" (red tags), "what to measure" (blue tags), "what to try" (green tags). Thin arrows connect each row left to right.

1. **"loss is NaN or diverges"** → "gradient norms per layer; learning rate" → "lower $\eta$; warmup; gradient clipping"
2. **"loss does not move"** → "std$(a_l)$ per layer; fraction of dead ReLUs; gradient norms" → "He initialization; Leaky ReLU / GELU; normalization; check $\eta$"
3. **"train improves, validation worsens"** → "gap $J_\text{val} - J_\text{train}$ over epochs" → "weight decay (AdamW); Dropout; data augmentation; early stopping"
4. **"curves very noisy or unstable"** → "batch size; BatchNorm mode" → "larger batch or smaller $\eta$; `model.train()` / `model.eval()`; LayerNorm"
5. **"initial loss far from $\ln C$"** → "loss and label encoding; output layer" → "fix the bug before tuning anything"

Above the map, a thin gray strip: "Before a long run: initial loss ≈ ln C · overfit one batch · finite non-zero gradients".

**Bottom banner:** "Diagnose with measurements, then change one thing at a time."

---

## S02-F20 · Why deep training works: integrated view (done)


File: `s02_f20_deep_training_summary.png`

**Center band (vertical middle of the canvas):** a horizontal deep network: $x$ → 6 identical layer blocks (rounded rectangles, each with three small circles and a small "Norm" sub-box at its bottom; the third block shows a $\phi$ symbol in its middle circle) → $\hat y$ → a loss box labeled $J$. Thin arrows between blocks. One orange residual skip arc **above** blocks 4 and 5, from the input of block 4 to the output of block 5. One purple curved arrow **below** the network from $J$ back to block 1, labeled "update".

**Six callout boxes**, each 400 px wide, title in bold plus at most two short lines, text fully inside the box. One straight, short leader line per box; no line crosses another line or the network.

Above the network, left to right:
1. **initialization** (blue tag) — "start at a stable scale · $\mathrm{Var}(W)=2/n_\text{in}$" → points at block 1
2. **activations** (green tag) — "keep gradients alive · ReLU, GELU" → points at the $\phi$ in block 3
3. **residual paths** (orange tag) — "a direct route for gradients" → points at the skip arc

Below the network, left to right:
4. **normalization** (teal tag) — "stable internal scales · BatchNorm, LayerNorm" → points at the "Norm" sub-box of block 2
5. **optimizer** (purple tag) — "gradients into good steps · Momentum, Adam, schedules" → points at the purple "update" arrow
6. **regularization** (gray tag) — "narrow the train/validation gap · weight decay, Dropout, early stopping" → points at the $J$ box

Do not print the numbers 1–6; they only fix the order.

**Bottom banner (inside the canvas, max two lines):** "No single trick makes deep networks trainable: each mechanism fixes one way the signal or the optimization can fail."

---

## S02-F22 · Mini-batch gradient noise versus batch size (done)

File: `s02_f22_batch_size_noise.png`

Three square panels side by side, same scale, each a 2-D parameter-space view. Panel headers: "$|\mathcal B| = 4$", "$|\mathcal B| = 16$", "$|\mathcal B| = 64$".

In every panel:
- a thick **navy arrow** from a common origin dot, labeled $\nabla_\theta J$ ("full-batch gradient"; write the words only in the first panel), identical in all three panels;
- about 40 thin semi-transparent **blue arrows** from the same origin: mini-batch gradients $g_t$, scattered around the navy arrow;
- a dashed **orange ellipse** around the arrow tips, centered on the tip of the navy arrow.

The spread halves from one panel to the next: the ellipse in panel 2 is **half** the size of panel 1, and panel 3 is half of panel 2 (each panel has 4× the batch). The center of the cloud always stays on the navy arrow tip (unbiased).

Under each panel a small label: panel 1 "std ∝ 1/√4 = 0.50", panel 2 "std ∝ 1/√16 = 0.25", panel 3 "std ∝ 1/√64 = 0.125".

**Bottom banner:** "Same expected direction, less noise: 4× the batch halves the standard deviation, at 4× the cost per step."

**Check:** ellipse sizes in ratio 1 : 1/2 : 1/4; navy arrow identical in the three panels; no σ symbol anywhere.

---

## S02-F23 · BatchNorm versus LayerNorm: which axis is normalized (done)

File: `s02_f23_batchnorm_vs_layernorm.png`

Two identical grids side by side, each 6 rows × 8 columns of small square cells representing the pre-activation matrix $Z$ of a mini-batch. In both grids: rows labeled on the left "example $i$" with a vertical arrow and the word "batch"; columns labeled on top "feature $k$" with a horizontal arrow and the word "features"; light gray cell fill.

1. **Left grid, header "BatchNorm":** one full **column** highlighted in blue (all 6 cells), with a blue bracket along it labeled "$\mu_k,\ s_k^2$ over the batch". Small caption under the grid: "one mean and variance per **feature** · depends on the batch · running averages at inference".
2. **Right grid, header "LayerNorm":** one full **row** highlighted in teal (all 8 cells), with a teal bracket along it labeled "$\mu_i,\ s_i^2$ over the features". Small caption under the grid: "one mean and variance per **example** · same in training and inference · works with batch size 1".

Both grids exactly the same size and position height; only the highlighted direction differs.

**Bottom banner:** "Same formula $\gamma\odot\hat z+\beta$; BatchNorm averages down a column, LayerNorm across a row."

**Check:** BatchNorm = vertical (column, across examples); LayerNorm = horizontal (row, across features). Do not swap them.

---

## S02-F24 · Label-preserving data augmentation (done)

File: `s02_f24_data_augmentation.png`

Top row: one original image on the left, a simple flat illustration of a **cat** sitting (not a photo, no real brand or character), labeled "original $x_i$ · label: cat". An arrow labeled "$T\sim\mathcal T$" fans out to five transformed versions of the same cat, each with a small caption and the same label "cat":
1. "random crop" (zoomed-in, partly cropped),
2. "horizontal flip" (mirrored),
3. "color jitter" (shifted hue and brightness),
4. "small rotation" (about 15°),
5. "noise" (light grain).

Bottom row, a separate small panel with a light red border, header "the transformation must not change the label": a handwritten-style digit **"6"** → arrow "rotate 180°" → a digit that now reads **"9"**, with a red ✗ and the caption "label changed: not a valid augmentation for digits".

**Bottom banner:** "Augmentation encodes invariances we know in advance: every $T(x_i)$ keeps the label $y_i$."

**Check:** all five augmented cats are clearly the same cat and keep the label "cat"; the 6 → 9 example is marked invalid.

---

# Corrections to figures already in the deck

These four figures are in the deck but contain content errors. Regenerate each one **keeping its layout**, and change only what is listed. Same file name, PNG, 1672 × 941.

## S02-F01 · From the learning loop to deep training — correction (done; only the script 𝒥 → J change remains, optional)

File: `s02_f01_learning_to_deep_training.png`

Keep the current two-panel layout (left: "Single learning loop (shallow model)" cycle; right: "Same loop, deep trainable system" with the deep network, the four-step strip and the three bottom boxes), the code box and the bottom banner. Fix:

1. Typo: the purple box in the right panel must read **"Update"**, not "Updsle".
2. Objective symbol: use a plain italic **$J$** everywhere (boxes "Loss $J$", "$J(\hat y, y)$", "$\nabla_\theta J$", "$\theta \leftarrow \theta - \eta\nabla_\theta J$"), never a script $\mathcal J$.
3. Replace the PyTorch flame logo with the plain word "PyTorch" in navy text (no logos).
4. In the code box keep exactly: `optimizer.zero_grad()`, `y_hat = model(x)`, `J = loss_fn(y_hat, y)`, `J.backward()`, `optimizer.step()`.

**Check:** no "Updsle"; no script J; no logo.

## S02-F02 · Signal propagation through depth — correction (slide 10)

File: `s02_f02_signal_propagation.png`

Keep the current layout: top band with the network and $a_l=\mathrm{ReLU}(W_l a_{l-1})$, $(W_l)_{ij}\sim\mathcal N(0,\,c\cdot 2/n_\text{in})$; a 3 × 5 grid of histograms (rows $c=0.5$, $c=1$ (He), $c=2$; columns input $x$, layer 1, layer 5, layer 10, layer 30); right-side labels; bottom banner. Fix the numbers so they follow $\operatorname{std}(a_l)\approx 0.82\,c^{l/2}$:

1. Input column: label every input histogram **"std(x) = 1.0"** (not std($a_l$)).
2. Row $c=0.5$: layer 1 "std($a_l$) = 0.58", layer 5 "0.15", layer 10 "0.026", layer 30 "2.5 · 10⁻⁵". x-axis ranges: [0, 2], [0, 0.5], [0, 0.1], [0, 10⁻⁴].
3. Row $c=1$ (He): **all four layers "std($a_l$) = 0.82"**, identical histograms, x-axis [0, 3] in every panel (scale preserved).
4. Row $c=2$: layer 1 "1.2", layer 5 "4.7", layer 10 "26", layer 30 "2.7 · 10⁴". x-axis ranges: [0, 4], [0, 15], [0, 80], [0, 10⁵].
5. ReLU histograms: a tall bar at 0 (half the units are exactly 0) and a decaying right tail; input histograms stay Gaussian on [−3, 3].

**Check:** the He row shows the same number (0.82) in all four layers; the input column says std(x).

## S02-F04 · Gradient norm through depth — correction (slide 16)

File: `s02_f04_gradient_norm_depth.png`

Keep the three-panel layout ("Vanishing gradients", "Stable gradients", "Exploding gradients"), the small network sketch with a red arrow "backward signal (backpropagation)" pointing **from output to input**, and the bottom banner. Fix the notation and the direction:

1. Layer labels under each network: $l = 0$ (input), 1, 2, …, $D-1$, $D$ (output). Use **$D$** for depth, never $L$.
2. Tag under each sketch: "$\lVert\partial a_l/\partial a_{l-1}\rVert < 1$ on average", "$\approx 1$", "$> 1$". Do not use $J^{(l)}$ (J is the objective in this course).
3. Plot y-axis label: **"$\lVert\delta_{a_l}\rVert$"** (backward signal at layer $l$), log scale 10⁻⁸ to 10²; x-axis "layer $l$" from 0 to 30 with **$D = 30$ at the right end**.
4. The backward signal starts at the output ($l=D=30$) with norm ≈ 1, then travels left:
   - **Vanishing:** curve ≈ 10⁰ at $l=30$ and **decreases toward the left**, reaching ≈ 10⁻⁸ at $l=0$.
   - **Stable:** flat ≈ 10⁰ across all layers.
   - **Exploding:** curve ≈ 10⁰ at $l=30$ and **increases toward the left**, reaching ≈ 10² at $l=0$.
5. Add a small gray arrow under each x-axis pointing left: "backward direction".

**Check:** in every panel the norm is ≈ 1 at the right end ($l=D$); vanishing is smallest at $l=0$, exploding is largest at $l=0$. No $h^{(l)}$, no $L$ for depth.

## S02-F07 · Initialization compared through variance — correction (slide 29)

File: `s02_f07_initialization_variance.png`

Keep the three-column layout (header box, small network sketch, four small histograms "layer 0, 1, 10, 50", activation-variance plot, gradient-variance plot; x-axis "layer $l$" 0–50, log y-axis). Fix:

1. **Column 1 header:** "Small initialization · ReLU", formula $W\sim\mathcal N(0,\,0.01^2)$, small gray note "$n_\text{in}=256$: variance × 0.013 per layer". Both curves fall by about two orders of magnitude **per layer** and hit the bottom of the plot (10⁻¹²) by layer 6; the histograms collapse to a spike at 0 from layer 1 on.
2. **Column 2 header:** "Xavier (Glorot) · tanh", formula $W\sim\mathcal N\!\left(0,\,\dfrac{2}{n_\text{in}+n_\text{out}}\right)$. Curves approximately flat near 10⁰ (slight decay allowed). Histograms: bounded in (−1, 1), symmetric (tanh outputs).
3. **Column 3 header:** "He (Kaiming) · ReLU", formula $W\sim\mathcal N\!\left(0,\,\dfrac{2}{n_\text{in}}\right)$. Curves flat near 10⁰. Histograms: ReLU shape, tall bar at 0 and a right tail, same width at layers 1, 10, 50.
4. Y-axis ticks: 10⁻¹², 10⁻⁸, 10⁻⁴, 10⁰ on all plots (no repeated tick labels such as two "10⁻²").
5. Small gray banner under the columns: "Match the initialization to the activation: Xavier for tanh, He for ReLU."

**Check:** Xavier formula is $2/(n_\text{in}+n_\text{out})$ and its column says tanh; He column says ReLU; small-init curves collapse within a few layers.
