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

Animations use the `media` class with a poster image, as in Session 02 (V33, V40, V72).

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper or another course.

## Deck structure

64 slides in five parts, each opened by a divider slide that recaps the previous part: 56 content slides, plus 4 figures and 4 animations already integrated. With the 10 remaining figures the deck has **74 slides** (S02 has 78).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Images are structured data | V05–V13 | |
| 2 · Convolution | V14–V28 | |
| 3 · CNNs and residual networks | V29–V40 | Practice A (V40): notebooks 3.1–3.3 |
| 4 · Transfer learning | V41–V49 | |
| 5 · From patches to attention | V50–V60 | Practice B (V60): notebooks 3.4–3.5 |
| Closure | V61–V64 | Integrating exercise (V61), exit question (V64) |

Self-attention gets one slide (V55); its full mechanics belong to Session 05.

Slide numbers below are those of the current source. For pending figures, "after slide" is the slide the hidden comment sits on; each inserted figure shifts the later numbers by one.

## Production routes

Three routes, depending on what the figure must show:

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md), same workflow as Session 02.
- **Script** — figures that show real data or a real model: a `make_s03_fnn_*.py` script in this folder (matplotlib, fixed seeds). Numbers come from the notebook named in the table, so the slide and the notebook always agree.
- **Manim** — animations, in `escenas_s03.py`, with the same conventions as `escenas_s02.py`.

## Planned figures

| ID | After slide | Purpose | Planned filename | Route | Status |
|---|---:|---|---|---|---|
| **S03-F01** | V07 (slide) | Image as a C×H×W tensor; one position across channels; batch dimension | `s03_f01_image_tensor.png` | Generator | Included — see review note |
| **S03-F02** | V10 | Permutation experiment: original vs permuted images, MLP and CNN accuracy on each | `s03_f02_permutation_experiment.png` | Script (numbers from notebook 3.1) | Pending |
| **S03-F03** | V18 (slide) | One image, four hand-made kernels and their feature maps | `s03_f03_kernels_feature_maps.png` | Script: `make_s03_f03_kernels_feature_maps.py` (skimage `camera`, 128×128) | Included |
| **S03-F04** | V20 (slide) | Multichannel convolution: one kernel spans all input channels; C_out kernels stack into C_out maps | `s03_f04_multichannel_conv.png` | Generator | Included — see review note |
| **S03-F05** | V31 (slide) | CNN anatomy as tensor blocks, 3×32×32 → logits | `s03_f05_cnn_anatomy.png` | Generator | Included |
| **S03-F06** | V36 | Feature hierarchy: first-layer kernels and top-activating patches per stage | `s03_f06_feature_hierarchy.png` | Script (ImageNet ResNet-18, Oxford Pets images, notebook 3.4) | Pending |
| **S03-F07** | V37 | ResNet basic block and downsampling block, with shapes | `s03_f07_resnet_blocks.png` | Generator | Pending |
| **S03-F08** | V43 | Pretrained backbone: general early features, specific late features, new head | `s03_f08_transfer_backbone.png` | Generator | Pending |
| **S03-F09** | V44 | From scratch, linear probe and fine-tuning: frozen vs trained layers | `s03_f09_transfer_strategies.png` | Generator | Pending |
| **S03-F10** | V48 | Test accuracy vs images per class for the three strategies | `s03_f10_transfer_vs_data.png` | Script (numbers from notebook 3.4) | Pending |
| **S03-F11** | V55 | Attention map of a pretrained ViT for one query patch | `s03_f11_attention_map.png` | Script (pretrained ViT, notebook 3.5) | Pending |
| **S03-F12** | V57 | ViT architecture: patches, position, CLS, pre-norm blocks, head | `s03_f12_vit_architecture.png` | Generator | Pending |
| **S03-F13** | V58 | Spectrum of assumptions MLP → CNN → ViT against data needed | `s03_f13_inductive_bias_spectrum.png` | Generator | Pending |
| **S03-F14** | V63 | Summary: what one unit can see in the first layers of an MLP, a CNN and a ViT | `s03_f14_architecture_summary.png` | Generator | Pending |

## Planned animations (Manim)

Only concepts where motion changes understanding.

| ID | Slide | Media slide title | What it shows | Rendered file | Status |
|---|---:|---|---|---|---|
| **S03-A01** | V13 | From dense to convolution | Dense → locally connected → shared: connections disappear, weights collapse to one kernel, parameter counter on screen (1 049 600 → 10 240 → 10) | `s03_a01_dense_to_conv.mp4` | Included |
| **S03-A02** | V16 | A kernel sliding over an image | 3×3 kernel over a 6×6 integer image; products and sum written at each position; 4×4 feature map filling in | `s03_a02_kernel_sliding.mp4` | Included |
| **S03-A03** | V26 | The receptive field grows with depth | One output unit; its input region highlighted layer by layer: three 3×3 layers with stride 1 (3, 5, 7), then with stride 2 (3, 7, 15) | `s03_a03_receptive_field.mp4` | Included |
| **S03-A04** | V54 | From image to tokens | Image → 4×4 grid of patches → each patch flattened → same linear map → token sequence, plus positional embeddings | `s03_a04_patches_to_tokens.mp4` | Included |

## Review notes

- **S03-F01:** channel shading does not follow the pixel values. The white background has value 1 in all three channels, so it should be the darkest cell in every grid; the image instead shades the leaf dark in R and B and the background light. The callout $(0.82, 0.41, 0.13)$ (from the prompt) is a reddish pixel, not a green leaf pixel. Structure, labels and shapes are correct. Usable as is; regeneration with the corrected prompt is optional.
- **S03-F04:** a stray white "C" next to the G label on the input block was removed by cloning the green cell below it (the delivered PNG is 1671 px wide; harmless with `bg contain`). The dashed arrow to the kernel starts at the edge of the green block, not at the dashed window on the R face; acceptable.

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

If production time is limited, prioritize:

1. **S03-F02** — the permutation experiment: the argument of the whole session (needs notebook 3.1);
2. **S03-F10** — transfer learning against the amount of data (needs notebook 3.4);
3. **S03-F12** — ViT architecture;
4. **S03-F07** — ResNet blocks.

Already included: S03-F01, S03-F03, S03-F04, S03-F05 and the four animations (S03-A01 to A04).

## Notes for slide integration

- Do not place figure titles inside the image. The slide title provides the title.
- Notation follows S01–S02: $a_l$ for activations, $\phi$ for a generic activation ($\sigma$ is only the sigmoid), $L$ per-example loss, $J$ objective, $D$ depth (here also the number of ViT blocks), std(·) for standard deviations.
- New S03 symbols: $C, H, W$ channels and spatial size; $K$ kernel size, $p$ padding, $s$ stride; $r_l$ receptive field; $g$ a pretrained backbone; $t_i$ tokens of size $d$; $W_E$ patch embedding; $p_i$ positional embedding; $d_k$ key size in attention.
