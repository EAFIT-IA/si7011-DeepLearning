# SI7011 — Session 03

## Figure and animation inventory

Session 03 focuses on the question:

> **How does data structure shape the network?**

Pending figures are hidden HTML comments in the slide source (`<!-- Pending figure S03-Fnn ... -->`, `<!-- Pending animation S03-Ann ... -->`), placed at the end of the slide they follow. Nothing unfinished is shown to students. To integrate a figure, insert a new slide right after that comment, using the same blocks as Session 02:

```markdown
---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain <alt text>](../figures/s03_fnn_<name>.png)
```

Animations use the `media` class with a poster image, as in Session 02 (V34, V44, V82).

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper or another course.

## Deck structure

**80 slides** in five parts, each opened by a divider slide that recaps the previous part: 56 content slides, 20 figures and 4 animations (S02 has 78).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Images are structured data | V05–V14 | |
| 2 · Convolution | V15–V33 | |
| 3 · CNNs and residual networks | V34–V49 | Practice A (V49): notebooks 3.1–3.3 |
| 4 · Transfer learning | V50–V61 | |
| 5 · From patches to attention | V62–V75 | Practice B (V75): notebooks 3.4–3.5 |
| Closure | V76–V80 | Integrating exercise (V76), exit question (V80) |

Self-attention gets one slide (V67); its full mechanics belong to Session 05.

Slide numbers below are those of the current source; every figure is integrated.

## Production routes

Three routes, depending on what the figure must show:

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md), same workflow as Session 02.
- **Script** — figures that show real data, a real model or numbers that must be exact (output sizes, pooled values): a `make_s03_fnn_*.py` script in this folder (matplotlib, fixed seeds). Numbers come from the notebook named in the table, so the slide and the notebook always agree.
- **Manim** — animations, in `escenas_s03.py`, with the same conventions as `escenas_s02.py`.

## Planned figures

| ID | After slide | Purpose | Planned filename | Route | Status |
|---|---:|---|---|---|---|
| **S03-F01** | V07 (slide) | Image as a C×H×W tensor; one position across channels; batch dimension | `s03_f01_image_tensor.png` | Generator | Included — see review note |
| **S03-F02** | V11 (slide) | Permutation experiment: original vs permuted images, MLP and CNN accuracy on each | `s03_f02_permutation_experiment.png` | Script: `make_s03_figs_from_notebooks.py` (images and numbers from the saved outputs of notebook 3.1, section 4) | Included |
| **S03-F03** | V19 (slide) | One image, four hand-made kernels and their feature maps | `s03_f03_kernels_feature_maps.png` | Script: `make_s03_f03_kernels_feature_maps.py` (skimage `camera`, 128×128) | Included |
| **S03-F04** | V21 (slide) | Multichannel convolution: one kernel spans all input channels; C_out kernels stack into C_out maps | `s03_f04_multichannel_conv.png` | Generator | Included — see review note |
| **S03-F16** | V23 (slide) | Padding and stride: the same 6×6 input with (p, s) = (0, 1), (1, 1), (1, 2), output sizes from the formula | `s03_f16_padding_stride.png` | Script: `make_s03_f16_padding_stride.py` | Included |
| **S03-F17** | V26 (slide) | Equivariance as a commuting square: conv(shift(x)) = shift(conv(x)), with the difference map | `s03_f17_equivariance.png` | Script: `make_s03_f17_equivariance.py` (same image and kernel as F03) | Included |
| **S03-F18** | V28 (slide) | 2×2 max and average pooling on a 4×4 map and on the map shifted 1 px | `s03_f18_pooling.png` | Script: `make_s03_f18_pooling.py` | Included |
| **S03-F20** | V32 (slide) | 1×1 convolution: the same $W \in \mathbb{R}^{C_\text{out} \times C_\text{in}}$ applied to the channel vector at every position | `s03_f20_one_by_one_conv.png` | Script: `make_s03_f20_one_by_one_conv.py` | Included — see review note |
| **S03-F19** | V41 (slide) | BatchNorm2d: one channel averaged over $N$, $H$, $W$, next to the BatchNorm1d column of S02-F23 | `s03_f19_batchnorm2d.png` | Generator | Included — see review note |
| **S03-F05** | V36 (slide) | CNN anatomy as tensor blocks, 3×32×32 → logits | `s03_f05_cnn_anatomy.png` | Generator | Included |
| **S03-F15** | V39 (slide) | VGG-style block (plain, no shortcut) with shapes; two stacked 3×3 see 5×5 with 18C² < 25C² parameters | `s03_f15_vgg_block.png` | Generator | Included — see review note |
| **S03-F06** | V44 (slide) | Feature hierarchy: first-layer kernels and top-activating patches per stage | `s03_f06_feature_hierarchy.png` | Script: `make_s03_figs_from_notebooks.py` (the two images of notebook 3.4, section 7) | Included |
| **S03-F07** | V46 (slide) | ResNet basic block and downsampling block, with shapes | `s03_f07_resnet_blocks.png` | Generator | Included |
| **S03-F08** | V53 (slide) | Pretrained backbone: general early features, specific late features, new head | `s03_f08_transfer_backbone.png` | Generator | Included |
| **S03-F09** | V55 (slide) | From scratch, linear probe and fine-tuning: frozen vs trained layers | `s03_f09_transfer_strategies.png` | Generator | Included |
| **S03-F10** | V60 (slide) | Validation accuracy vs images per class for the three strategies | `s03_f10_transfer_vs_data.png` | Script: `make_s03_figs_from_notebooks.py` (numbers from the saved outputs of notebook 3.4, section 4) | Included |
| **S03-F11** | V68 (slide) | Attention of [CLS] in a pretrained ViT-B/16: last layer and attention rollout | `s03_f11_attention_map.png` | Script: `make_s03_figs_from_notebooks.py` (the image of notebook 3.5, section 6) | Included — see review note |
| **S03-F12** | V71 (slide) | ViT architecture: patches, position, CLS, pre-norm blocks, head | `s03_f12_vit_architecture.png` | Generator | Included — see review note |
| **S03-F13** | V73 (slide) | Spectrum of assumptions MLP → CNN → ViT against data needed | `s03_f13_inductive_bias_spectrum.png` | Generator | Included |
| **S03-F14** | V79 (slide) | Summary: what one unit can see in the first layers of an MLP, a CNN and a ViT | `s03_f14_architecture_summary.png` | Generator + script: CNN panel redrawn by `fix_s03_f14_cnn_panel.py` | Included — see review note |

## Planned animations (Manim)

Only concepts where motion changes understanding.

| ID | Slide | Media slide title | What it shows | Rendered file | Status |
|---|---:|---|---|---|---|
| **S03-A01** | V14 | From dense to convolution | Dense → locally connected → shared: connections disappear, weights collapse to one kernel, parameter counter on screen (1 049 600 → 10 240 → 10) | `s03_a01_dense_to_conv.mp4` | Included |
| **S03-A02** | V17 | A kernel sliding over an image | 3×3 kernel over a 6×6 integer image; products and sum written at each position; 4×4 feature map filling in | `s03_a02_kernel_sliding.mp4` | Included |
| **S03-A03** | V30 | The receptive field grows with depth | One output unit; its input region highlighted layer by layer: three 3×3 layers with stride 1 (3, 5, 7), then with stride 2 (3, 7, 15) | `s03_a03_receptive_field.mp4` | Included |
| **S03-A04** | V66 | From image to tokens | Image → 4×4 grid of patches → each patch flattened → same linear map → token sequence, plus positional embeddings | `s03_a04_patches_to_tokens.mp4` | Included |

## Review notes

- **S03-F19:** headers, orange column and orange channel, formulas and the $2C$ box are exact; no $\sigma$. Each channel slab shows about 7 × 3 cells on its top face, not the $4 \times 4$ of its label; the label is what the slide uses. Delivered at 2:1 (1774 × 887), letterboxed by `bg contain`.
- **S03-F20:** the generated version drew the highlighted position as a column of pixels on the front face (several positions) instead of one cell going through all channels, which contradicts the point of the slide; it also read "+, b". Redrawn by `make_s03_f20_one_by_one_conv.py` with the same composition: depth is the channel axis, one 1 × 1 rod through 64 channels in, 16 out, two other positions in dashed gray using the same $W$.
- **S03-F11:** the last-layer attention of [CLS] lands on a few background tokens (the attention sinks of Darcet et al., 2024), not on the object. Rollout spreads it: it covers the car and the front of the second truck, and stays scattered on the frog. The banner says so and does not claim that attention finds the object. Class names are re-typed in the script; if notebook 3.5 changes `idx`, update them.

- **S03-F01:** channel shading does not follow the pixel values. The white background has value 1 in all three channels, so it should be the darkest cell in every grid; the image instead shades the leaf dark in R and B and the background light. The callout $(0.82, 0.41, 0.13)$ (from the prompt) is a reddish pixel, not a green leaf pixel. Structure, labels and shapes are correct. Usable as is; regeneration with the corrected prompt is optional.
- **S03-F04:** a stray white "C" next to the G label on the input block was removed by cloning the green cell below it (the delivered PNG is 1671 px wide; harmless with `bg contain`). The dashed arrow to the kernel starts at the edge of the green block, not at the dashed window on the R face; acceptable.
- **S03-F15:** left panel and parameter boxes are exact. In the receptive-field sketch the conv-1 unit fans down to a separate 3×3 patch and the 5×5 input grid is not connected to anything, so the 5×5 claim rests on the caption. Acceptable because S03-A03 (V27) already shows the growth with real connections; an optional 1-D redraw is in the prompt. V29 and V38 now state that padding does not change the receptive field, so the window labels read correctly.
- **S03-F12:** architecture, pre-norm order, token count (197) and ViT-B/16 sizes are exact. The arrow from $W_E$ runs into the [CLS] bar, which suggests [CLS] is a projected patch; V70 now says it is a learned token. Delivered at 2:1 (1774 × 887), letterboxed by `bg contain`.
- **S03-F14:** the generated CNN panel drew a 4×3 grid labelled "3 × 3" and a 6×6 region labelled "5 × 5", off the input grid. That panel is redrawn on the grid by `fix_s03_f14_cnn_panel.py` from `s03_f14_architecture_summary_generated.png` (kept for reproducibility); MLP and ViT panels, header and captions are the generated ones.

## Animation sources

All four scenes live in [escenas_s03.py](escenas_s03.py); every number on screen is computed with numpy when the scene is built (fixed seeds):

| Rendered animation | Scene | What is computed |
|---|---|---|
| `s03_a01_dense_to_conv.mp4` | `A01_DenseToConv` | parameter counts for 8 → 8 drawn and for 32×32 → 32×32 (dense, local 3×3, shared 3×3) |
| `s03_a02_kernel_sliding.mp4` | `A02_KernelSliding` | valid cross-correlation of a 6×6 integer image with the kernel [-1 0 1] (3 rows); 4×4 output |
| `s03_a03_receptive_field.mp4` | `A03_ReceptiveField` | receptive field indices through three 3×3 layers, no padding, stride 1 and stride 2 |
| `s03_a04_patches_to_tokens.mp4` | `A04_PatchesToTokens` | skimage `chelsea` at 64×64, 16 patches, a random $W_E$ with $d = 8$, random $p_i$ |

Render with `S01_VIDEO=1 manim -qh escenas_s03.py <Scene>` (Manim Community 0.22, LaTeX, Inter font). Posters are the final frame (`ffmpeg -sseof -0.1 -i <mp4> -frames:v 1 <poster>`).

## Core visual priorities

All figures (S03-F01 to F20) and the four animations (S03-A01 to A04) are included. Optional: regenerate S03-F01 and S03-F15 with the corrected prompts.

## Notes for slide integration

- Do not place figure titles inside the image. The slide title provides the title.
- Notation follows S01–S02: $a_l$ for activations, $\phi$ for a generic activation ($\sigma$ is only the sigmoid), $L$ per-example loss, $J$ objective, $D$ depth (here also the number of ViT blocks), std(·) for standard deviations.
- New S03 symbols: $C, H, W$ channels and spatial size; $K$ kernel size, $p$ padding, $s$ stride; $r_l$ receptive field; $g$ a pretrained backbone; $t_i$ tokens of size $d$; $W_E$ patch embedding; $p_i$ positional embedding; $d_k$ key size in attention.
