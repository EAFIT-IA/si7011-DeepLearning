# SI7011 — Session 02

## Figure and animation inventory

Session 02 focuses on the question:

> **Why can we train deep neural networks?**

Pending figures are HTML comments in the slide source (`<!-- Pending figure: ... -->`), so nothing unfinished is shown to students. Figures should keep white backgrounds, academic visual style, preserved proportions and filenames without spaces.

## Deck structure

78 slides in five parts, each opened by a divider slide that recaps the previous part.

## Planned figures

| ID | Current slide(s) | Purpose | Planned filename | Recommended format | Status |
|---|---:|---|---|---|---|
| **S02-F01** | V04 | From the single learning loop to a deep trainable system | `s02_f01_learning_to_deep_training.png` | PNG | Included |
| **S02-F02** | V10 | Activation distributions propagating through many layers | `s02_f02_signal_propagation.png` | PNG | Included |
| **S02-F03** | V30 | Vanishing, stable and exploding activation scales across depth | `s02_f03_activation_scale_depth.png` | PNG / plot | Covered by S02-A01 |
| **S02-F04** | V16 | Gradient norm versus layer depth for vanishing, stable and exploding regimes | `s02_f04_gradient_norm_depth.png` | PNG | Included |
| **S02-F05** | V18 | Sigmoid, tanh and ReLU: functions and derivative regions | `s02_f05_activation_derivatives.png` | PNG | Included |
| **S02-F06** | V24 | Symmetry problem when hidden units share identical initialization | `s02_f06_initialization_symmetry.png` | PNG / SVG | Included |
| **S02-F07** | V29 | Naive, Xavier and He initialization compared through activation variance | `s02_f07_initialization_variance.png` | PNG | Included |
| **S02-F08** | V37 | SGD and Momentum trajectories in a narrow valley | `s02_f08_sgd_momentum_valley.png` | PNG | Covered by S02-A02 |
| **S02-F09** | V40 | Optimizer trajectories on the same ill-conditioned objective | `s02_f09_optimizer_trajectories.svg` | SVG | Included |
| **S02-F10** | V41 | Learning-rate schedules | `s02_f10_learning_rate_schedules.svg` | SVG | Included |
| **S02-F11** | V46 | BatchNorm: normalize, then learn scale and shift | `s02_f11_batchnorm_transform.png` | PNG | Included |
| **S02-F12** | V48 | BatchNorm behavior in training mode versus evaluation mode | `s02_f12_batchnorm_train_eval.png` | PNG / SVG | Included |
| **S02-F13** | V54 | Underfitting, healthy fitting and overfitting in train/validation curves | `s02_f13_train_validation_dynamics.png` | PNG / plot | Included |
| **S02-F14** | V56 | Effect of weight decay on learned functions or weight norms | `s02_f14_weight_decay_effect.png` | PNG | Included |
| **S02-F15** | V59 | Dropout as stochastic subnetworks during training | `s02_f15_dropout_subnetworks.png` | PNG | Included |
| **S02-F16** | V65 | Depth degradation: deeper plain networks can be harder to optimize | `s02_f16_depth_degradation.png` | PNG / plot | Included |
| **S02-F17** | V67 | Residual block: transformation path plus identity path | `s02_f17_residual_block.png` | PNG | Included |
| **S02-F18** | V69 | Gradient flow through the identity path in a residual block | `s02_f18_residual_gradient_path.png` | PNG | Covered by S02-A03 |
| **S02-F19** | V72 | Map from symptoms to measurements and training interventions | `s02_f19_training_diagnostics_map.png` | PNG | Included |
| **S02-F20** | V77 | Integrated view of mechanisms that make deep training possible | `s02_f20_deep_training_summary.png` | PNG | Included |
| **S02-F21** | V20 | ReLU, Leaky ReLU and GELU with derivatives; dead region highlighted | `s02_f21_relu_variants.png` | PNG | Included |
| **S02-F22** | V35 | Gradient-estimate spread versus batch size around the full-batch gradient | `s02_f22_batch_size_noise.png` | PNG (matplotlib, `make_s02_f22_batch_size_noise.py`) | Included |
| **S02-F23** | V51 | BatchNorm versus LayerNorm: normalized axis of the (batch × features) tensor | `s02_f23_batchnorm_vs_layernorm.png` | PNG | Included |
| **S02-F24** | V62 | One image and several label-preserving augmentations | `s02_f24_data_augmentation.png` | PNG | Included |
| **S02-F25** | V07 | Deep MLP forward pass with example activations and softmax output | `s02_f25_deep_mlp_forward.png` | PNG | Included — notation differs from deck |
| **S02-F26** | V14 | Backward pass as a product of layer Jacobians | `s02_f26_backprop_jacobians.png` | PNG | Included — notation differs; Jacobian product needs transposes |

## Review notes

- **S02-F25 / S02-F26:** these use superscript notation ($a^{(l)}$, $h^{(l)}$, $\\mathcal L$) and $L$ for depth, while the deck uses $a_l$, $D$ and $J$. F26's Jacobian product also requires transposes under the column-vector convention used in the deck.

### Current production backlog

Figures still to produce:

None. All S02 figures are integrated.

## Recommended animations

Only animate concepts where motion changes understanding.

| ID | Related figure | Concept | Planned rendered file | Status |
|---|---|---|---|---|
| **S02-A01** | S02-F02 / S02-F03 | Activations shrink, explode or stabilize layer by layer | `s02_a01_signal_propagation.mp4` | Included (V30) |
| **S02-A02** | S02-F08 | SGD oscillates in a narrow valley while Momentum accumulates useful direction | `s02_a02_momentum_valley.mp4` | Included (V37) |
| **S02-A03** | S02-F17 / S02-F18 | Residual block forward path and direct gradient route | `s02_a03_residual_gradient_path.mp4` | Included (V69) |

## Animation sources

All three scenes live in [escenas_s02.py](escenas_s02.py) and compute their data with numpy (fixed seeds):

| Rendered animation | Scene | What is computed |
|---|---|---|
| `s02_a01_signal_propagation.mp4` | `A01_SignalPropagation` | std(a_l) and std(δ_{a_l}) in a 30-layer ReLU MLP (width 256) for W ~ N(0, c·2/n), c = 0.5, 1 (He), 2 |
| `s02_a02_momentum_valley.mp4` | `A02_MomentumValley` | SGD (η = 0.09) and Momentum (η = 0.04, β = 0.7) on J = ½(u² + 20v²) |
| `s02_a03_residual_gradient_path.mp4` | `A03_ResidualGradient` | ‖δ_{a_l}‖ in 20 tanh layers (width 64), plain vs residual, same weights |

Render with `S01_VIDEO=1 manim -qh escenas_s02.py <Scene>` (Manim Community 0.21, LaTeX, Inter font).
Each animation slide uses its final frame (`s02_aNN_..._poster.png`) as poster and print fallback.

## Core visual priorities

If production time is limited, prioritize:

1. **S02-F03** — activation scale through depth;
2. **S02-F04** — gradient norm through depth;
3. **S02-F07** — naive vs Xavier vs He initialization;
4. **S02-F11** — BatchNorm transformation;
5. **S02-F17/S02-F18** — residual block and gradient path;
6. **S02-F19** — diagnostics map.

## Notes for slide integration

- Do not place figure titles inside the image. The slide title provides the title.
- Use PNG for plotted or illustrative academic visuals.
- Use SVG only for simple clean diagrams where text remains readable in Marp.
- Keep filenames lowercase with underscores.
- Pending figures are hidden comments, not visible placeholders or broken image links.
- Notation follows S01: $z_l$, $a_l=\phi(z_l)$, depth $D$, $\delta_{a_l}$, $J$ for objectives, std(·) instead of $\sigma$ ($\sigma$ is the sigmoid).
