# SI7011 — Session 02

## Figure and animation inventory

Session 02 focuses on the question:

> **Why can we train deep neural networks?**

Pending figures are HTML comments in the slide source (`<!-- Pending figure: ... -->`), so nothing unfinished is shown to students. Figures should keep white backgrounds, academic visual style, preserved proportions and filenames without spaces.

## Deck structure

65 slides in five parts, each opened by a divider slide that recaps the previous part.

## Planned figures

| ID | Current slide(s) | Purpose | Planned filename | Recommended format | Status |
|---|---:|---|---|---|---|
| **S02-F01** | V04 | From the single learning loop to a deep trainable system | `s02_f01_learning_to_deep_training.png` | PNG | Included |
| **S02-F02** | V10 | Activation distributions propagating through many layers | `s02_f02_signal_propagation.png` | PNG | Included |
| **S02-F03** | V30 | Vanishing, stable and exploding activation scales across depth | `s02_f03_activation_scale_depth.png` | PNG / plot | Covered by S02-A01 |
| **S02-F04** | V16 | Gradient norm versus layer depth for vanishing, stable and exploding regimes | `s02_f04_gradient_norm_depth.png` | PNG | Included — gradient direction reversed; explain with A01 |
| **S02-F05** | V18 | Sigmoid, tanh and ReLU: functions and derivative regions | `s02_f05_activation_derivatives.png` | PNG | Included |
| **S02-F06** | V24 | Symmetry problem when hidden units share identical initialization | `s02_f06_initialization_symmetry.png` | PNG / SVG | Included |
| **S02-F07** | V29 | Naive, Xavier and He initialization compared through activation variance | `s02_f07_initialization_variance.png` | PNG | Included — needs correction to deck Xavier variance |
| **S02-F08** | V36 | SGD and Momentum trajectories in a narrow valley | `s02_f08_sgd_momentum_valley.png` | PNG | Covered by S02-A02 |
| **S02-F09** | V39 | Optimizer trajectories on the same ill-conditioned objective | `s02_f09_optimizer_trajectories.svg` | SVG | Included |
| **S02-F10** | V40 | Learning-rate schedules | `s02_f10_learning_rate_schedules.svg` | SVG | Included |
| **S02-F11** | V44 | BatchNorm: normalize, then learn scale and shift | `s02_f11_batchnorm_transform.png` | PNG | TODO |
| **S02-F12** | V45 | BatchNorm behavior in training mode versus evaluation mode | `s02_f12_batchnorm_train_eval.png` | PNG / SVG | TODO |
| **S02-F13** | V49 | Underfitting, healthy fitting and overfitting in train/validation curves | `s02_f13_train_validation_dynamics.png` | PNG / plot | TODO |
| **S02-F14** | V50 | Effect of weight decay on learned functions or weight norms | `s02_f14_weight_decay_effect.png` | PNG | TODO |
| **S02-F15** | V52 | Dropout as stochastic subnetworks during training | `s02_f15_dropout_subnetworks.png` | PNG | TODO |
| **S02-F16** | V56 | Depth degradation: deeper plain networks can be harder to optimize | `s02_f16_depth_degradation.png` | PNG / plot | TODO |
| **S02-F17** | V57 | Residual block: transformation path plus identity path | `s02_f17_residual_block.png` | PNG | TODO |
| **S02-F18** | V59 | Gradient flow through the identity path in a residual block | `s02_f18_residual_gradient_path.png` | PNG | Covered by S02-A03 |
| **S02-F19** | V61 | Map from symptoms to measurements and training interventions | `s02_f19_training_diagnostics_map.png` | PNG | TODO |
| **S02-F20** | V64 | Integrated view of mechanisms that make deep training possible | `s02_f20_deep_training_summary.png` | PNG | TODO |
| **S02-F21** | V20 | ReLU, Leaky ReLU and GELU with derivatives; dead region highlighted | `s02_f21_relu_variants.png` | PNG | Included |
| **S02-F22** | V34 | Gradient-estimate spread versus batch size around the full-batch gradient | `s02_f22_batch_size_noise.svg` | SVG / plot | Deferred — not in current production backlog |
| **S02-F23** | V47 | BatchNorm versus LayerNorm: normalized axis of the (batch × features) tensor | `s02_f23_batchnorm_vs_layernorm.svg` | SVG | Deferred — not in current production backlog |
| **S02-F24** | V54 | One image and several label-preserving augmentations | `s02_f24_data_augmentation.png` | PNG | Deferred — not in current production backlog |
| **S02-F25** | V07 | Deep MLP forward pass with example activations and softmax output | `s02_f25_deep_mlp_forward.png` | PNG | Included — notation differs from deck |
| **S02-F26** | V14 | Backward pass as a product of layer Jacobians | `s02_f26_backprop_jacobians.png` | PNG | Included — notation differs; Jacobian product needs transposes |

## Review notes

- **S02-F04:** the static figure reverses the backpropagation direction. Gradients originate at the output ($l=D$) and propagate toward the input ($l=0$). In the vanishing case the norm should be smallest near $l=0$; in the exploding case it should be largest near $l=0$. **S02-A01 is correct and should be used for explanation.**
- **S02-F07:** the Xavier panel currently labels $W\\sim\\mathcal N(0,1/n_\\text{in})$, while the deck derives $\\operatorname{Var}(W)=2/(n_\\text{in}+n_\\text{out})$. The shown stable behavior is also inconsistent with a ReLU network, where Xavier would halve the variance approximately layer by layer. The small-initialization activation axis repeats $10^{-2}$.
- **S02-F25 / S02-F26:** these use superscript notation ($a^{(l)}$, $h^{(l)}$, $\\mathcal L$) and $L$ for depth, while the deck uses $a_l$, $D$ and $J$. F26's Jacobian product also requires transposes under the column-vector convention used in the deck.

### Current production backlog

The 12 figures still to produce are:

**S02-F11, S02-F12, S02-F13, S02-F14, S02-F15, S02-F16, S02-F17, S02-F19, S02-F20.**

F22–F24 remain documented in the deck but are outside the current figure-production backlog.

## Recommended animations

Only animate concepts where motion changes understanding.

| ID | Related figure | Concept | Planned rendered file | Status |
|---|---|---|---|---|
| **S02-A01** | S02-F02 / S02-F03 | Activations shrink, explode or stabilize layer by layer | `s02_a01_signal_propagation.mp4` | Included (V30) |
| **S02-A02** | S02-F08 | SGD oscillates in a narrow valley while Momentum accumulates useful direction | `s02_a02_momentum_valley.mp4` | Included (V36) |
| **S02-A03** | S02-F17 / S02-F18 | Residual block forward path and direct gradient route | `s02_a03_residual_gradient_path.mp4` | Included (V59) |

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
