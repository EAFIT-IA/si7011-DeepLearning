# SI7011 — Session 03

## Figure and animation inventory

Session 03 focuses on the question:

> **How does data structure shape the network?**

Pending figures are HTML comments in the slide source (`<!-- Pending figure: ... -->` and `<!-- Pending animation: ... -->`), so nothing unfinished is shown to students. Figures should keep white backgrounds, academic visual style, preserved proportions and filenames without spaces. All figures are drawn from scratch; figures that show a trained model's behavior (F02, F06, F10, F11) use models and numbers produced by the session notebooks.

## Deck structure

74 slides in five parts, each opened by a divider slide that recaps the previous part. Self-attention is introduced at the level of one slide (V61); its full mechanics belong to Session 05.

## Planned figures

| ID | Current slide(s) | Purpose | Planned filename | Recommended format | Status |
|---|---:|---|---|---|---|
| **S03-F01** | V07 | Image as a C×H×W tensor; one position across channels; batch dimension | `s03_f01_image_tensor.png` | PNG / SVG | Pending |
| **S03-F02** | V11 | Permutation experiment: original vs permuted images, MLP and CNN accuracy on each | `s03_f02_permutation_experiment.png` | PNG (matplotlib, numbers from notebook 3.1) | Pending |
| **S03-F03** | V19 | One image, four hand-made kernels and their feature maps | `s03_f03_kernels_feature_maps.png` | PNG (matplotlib) | Pending |
| **S03-F04** | V21 | Multichannel convolution: C_in kernels per output map, C_out maps stacked | `s03_f04_multichannel_conv.png` | PNG / SVG | Pending |
| **S03-F05** | V32 | CNN anatomy as tensor blocks, 3×32×32 → logits | `s03_f05_cnn_anatomy.png` | PNG / SVG | Pending |
| **S03-F06** | V38 | Feature hierarchy: first-layer kernels and top-activating patches per stage | `s03_f06_feature_hierarchy.png` | PNG (from a trained ResNet-18) | Pending |
| **S03-F07** | V40 | ResNet basic block and downsampling block, with shapes | `s03_f07_resnet_blocks.png` | PNG / SVG | Pending |
| **S03-F08** | V47 | Pretrained backbone: general early features, specific late features, new head | `s03_f08_transfer_backbone.png` | PNG / SVG | Pending |
| **S03-F09** | V49 | From scratch, linear probe and fine-tuning: frozen vs trained layers | `s03_f09_transfer_strategies.png` | PNG / SVG | Pending |
| **S03-F10** | V54 | Test accuracy vs images per class for the three strategies | `s03_f10_transfer_vs_data.png` | PNG (matplotlib, numbers from notebook 3.4) | Pending |
| **S03-F11** | V62 | Attention map of a pretrained ViT for one query patch | `s03_f11_attention_map.png` | PNG (from notebook 3.5) | Pending |
| **S03-F12** | V65 | ViT architecture: patches, position, CLS, pre-norm blocks, head | `s03_f12_vit_architecture.png` | PNG / SVG | Pending |
| **S03-F13** | V67 | Spectrum of assumptions MLP → CNN → ViT against data needed | `s03_f13_inductive_bias_spectrum.png` | PNG / SVG | Pending |
| **S03-F14** | V73 | Summary: what each layer of MLP, CNN and ViT can see | `s03_f14_architecture_summary.png` | PNG / SVG | Pending |

## Planned animations (Manim)

| ID | Current slide | What it shows | Planned filename | Status |
|---|---:|---|---|---|
| **S03-A01** | V14 | Dense → locally connected → shared: connections disappear, weights collapse to one kernel, parameter counter on screen | `s03_a01_dense_to_conv.mp4` | Pending |
| **S03-A02** | V17 | 3×3 kernel sliding over a small integer image; products and sum at each position; feature map filling in | `s03_a02_kernel_sliding.mp4` | Pending |
| **S03-A03** | V27 | Receptive field of one output unit, layer by layer, with stride 1 and with stride 2 | `s03_a03_receptive_field.mp4` | Pending |
| **S03-A04** | V60 | Image → patch grid → flattened patches → linear map → token sequence with positions | `s03_a04_patches_to_tokens.mp4` | Pending |

Each animation needs a poster PNG (`*_poster.png`) for the PDF, as in Session 02.
