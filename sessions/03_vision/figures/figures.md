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

Animations use the `media` class with a poster image, as in Session 02 (V30, V37, V69).

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper or another course.

## Deck structure

56 content slides in five parts, each opened by a divider slide that recaps the previous part. With the 14 figures and 4 animations integrated, the deck has **74 slides** (S02 has 78).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Images are structured data | V05–V11 | |
| 2 · Convolution | V12–V22 | |
| 3 · CNNs and residual networks | V23–V33 | Practice A (V33): notebooks 3.1–3.3 |
| 4 · Transfer learning | V34–V42 | |
| 5 · From patches to attention | V43–V52 | Practice B (V52): notebooks 3.4–3.5 |
| Closure | V53–V56 | Integrating exercise (V53), exit question (V56) |

Self-attention gets one slide (V47); its full mechanics belong to Session 05.

Slide numbers below are those of the current source, **before** any figure is inserted. Each inserted figure shifts the later numbers by one.

## Production routes

Three routes, depending on what the figure must show:

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md), same workflow as Session 02.
- **Script** — figures that show real data or a real model: a `make_s03_fnn_*.py` script in this folder (matplotlib, fixed seeds). Numbers come from the notebook named in the table, so the slide and the notebook always agree.
- **Manim** — animations, in `escenas_s03.py`, with the same conventions as `escenas_s02.py`.

## Planned figures

| ID | After slide | Purpose | Planned filename | Route | Status |
|---|---:|---|---|---|---|
| **S03-F01** | V06 | Image as a C×H×W tensor; one position across channels; batch dimension | `s03_f01_image_tensor.png` | Generator | Pending |
| **S03-F02** | V09 | Permutation experiment: original vs permuted images, MLP and CNN accuracy on each | `s03_f02_permutation_experiment.png` | Script (numbers from notebook 3.1) | Pending |
| **S03-F03** | V14 | One image, four hand-made kernels and their feature maps | `s03_f03_kernels_feature_maps.png` | Script (independent of notebooks) | Pending |
| **S03-F04** | V15 | Multichannel convolution: one kernel spans all input channels; C_out kernels stack into C_out maps | `s03_f04_multichannel_conv.png` | Generator | Pending |
| **S03-F05** | V24 | CNN anatomy as tensor blocks, 3×32×32 → logits | `s03_f05_cnn_anatomy.png` | Generator | Pending |
| **S03-F06** | V29 | Feature hierarchy: first-layer kernels and top-activating patches per stage | `s03_f06_feature_hierarchy.png` | Script (ImageNet ResNet-18, Oxford Pets images, notebook 3.4) | Pending |
| **S03-F07** | V30 | ResNet basic block and downsampling block, with shapes | `s03_f07_resnet_blocks.png` | Generator | Pending |
| **S03-F08** | V36 | Pretrained backbone: general early features, specific late features, new head | `s03_f08_transfer_backbone.png` | Generator | Pending |
| **S03-F09** | V37 | From scratch, linear probe and fine-tuning: frozen vs trained layers | `s03_f09_transfer_strategies.png` | Generator | Pending |
| **S03-F10** | V41 | Test accuracy vs images per class for the three strategies | `s03_f10_transfer_vs_data.png` | Script (numbers from notebook 3.4) | Pending |
| **S03-F11** | V47 | Attention map of a pretrained ViT for one query patch | `s03_f11_attention_map.png` | Script (pretrained ViT, notebook 3.5) | Pending |
| **S03-F12** | V49 | ViT architecture: patches, position, CLS, pre-norm blocks, head | `s03_f12_vit_architecture.png` | Generator | Pending |
| **S03-F13** | V50 | Spectrum of assumptions MLP → CNN → ViT against data needed | `s03_f13_inductive_bias_spectrum.png` | Generator | Pending |
| **S03-F14** | V55 | Summary: what one unit can see in the first layers of an MLP, a CNN and a ViT | `s03_f14_architecture_summary.png` | Generator | Pending |

## Planned animations (Manim)

Only concepts where motion changes understanding.

| ID | After slide | Media slide title | What it shows | Planned filename | Status |
|---|---:|---|---|---|---|
| **S03-A01** | V11 | From dense to convolution | Dense → locally connected → shared: connections disappear, weights collapse to one kernel, parameter counter on screen (1 049 600 → 10 240 → 10) | `s03_a01_dense_to_conv.mp4` | Pending |
| **S03-A02** | V13 | A kernel sliding over an image | 3×3 kernel over a 6×6 integer image; products and sum written at each position; 4×4 feature map filling in | `s03_a02_kernel_sliding.mp4` | Pending |
| **S03-A03** | V20 | The receptive field grows with depth | One output unit; its input region highlighted layer by layer: three 3×3 layers with stride 1 (3, 5, 7), then with stride 2 (3, 7, 15) | `s03_a03_receptive_field.mp4` | Pending |
| **S03-A04** | V46 | From image to tokens | Image → 4×4 grid of patches → each patch flattened → same linear map → token sequence, plus positional embeddings | `s03_a04_patches_to_tokens.mp4` | Pending |

Each animation needs a poster (`*_poster.png`, its final frame) for the PDF, as in Session 02.

## Core visual priorities

If production time is limited, prioritize:

1. **S03-A02** — the sliding kernel: the definition of convolution;
2. **S03-F02** — the permutation experiment: the argument of the whole session;
3. **S03-A03** — receptive field growth;
4. **S03-F10** — transfer learning against the amount of data;
5. **S03-F12** — ViT architecture.

## Notes for slide integration

- Do not place figure titles inside the image. The slide title provides the title.
- Notation follows S01–S02: $a_l$ for activations, $\phi$ for a generic activation ($\sigma$ is only the sigmoid), $L$ per-example loss, $J$ objective, $D$ depth (here also the number of ViT blocks), std(·) for standard deviations.
- New S03 symbols: $C, H, W$ channels and spatial size; $K$ kernel size, $p$ padding, $s$ stride; $r_l$ receptive field; $g$ a pretrained backbone; $t_i$ tokens of size $d$; $W_E$ patch embedding; $p_i$ positional embedding; $d_k$ key size in attention.
