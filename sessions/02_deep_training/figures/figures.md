# SI7011 — Session 02

## Figure and animation inventory

Session 02 focuses on the question:

> **Why can we train deep neural networks?**

Pending figures are HTML comments in the slide source (`<!-- Pending figure: ... -->`), so nothing unfinished is shown to students. Figures should keep white backgrounds, academic visual style, preserved proportions and filenames without spaces.

## Planned figures

| ID | Current slide(s) | Purpose | Planned filename | Recommended format | Status |
|---|---:|---|---|---|---|
| **S02-F01** | V02 | From the single learning loop to a deep trainable system | `s02_f01_learning_to_deep_training.png` | PNG | TODO |
| **S02-F02** | V06 | Activation distributions propagating through many layers | `s02_f02_signal_propagation.png` | PNG | TODO |
| **S02-F03** | V11 | Vanishing, stable and exploding activation scales across depth | `s02_f03_activation_scale_depth.png` | PNG / plot | Covered by S02-A01 |
| **S02-F04** | V11 | Gradient norm versus layer depth for vanishing, stable and exploding regimes | `s02_f04_gradient_norm_depth.png` | PNG / plot | Covered by S02-A01 |
| **S02-F05** | V12 | Sigmoid, tanh and ReLU: functions and derivative regions | `s02_f05_activation_derivatives.png` | PNG | TODO |
| **S02-F06** | V14 | Symmetry problem when hidden units share identical initialization | `s02_f06_initialization_symmetry.png` | PNG / SVG | TODO |
| **S02-F07** | V17 | Naive, Xavier and He initialization compared through activation variance | `s02_f07_initialization_variance.png` | PNG / plot | TODO |
| **S02-F08** | V22 | SGD and Momentum trajectories in a narrow valley | `s02_f08_sgd_momentum_valley.png` | PNG | Covered by S02-A02 |
| **S02-F09** | V25 | Optimizer trajectories on the same ill-conditioned objective | `s02_f09_optimizer_trajectories.svg` | SVG | Included |
| **S02-F10** | V26 | Learning-rate schedules | `s02_f10_learning_rate_schedules.svg` | SVG | Included |
| **S02-F11** | V29 | BatchNorm: normalize, then learn scale and shift | `s02_f11_batchnorm_transform.png` | PNG | TODO |
| **S02-F12** | V30 | BatchNorm behavior in training mode versus evaluation mode | `s02_f12_batchnorm_train_eval.png` | PNG / SVG | TODO |
| **S02-F13** | V31 | Underfitting, healthy fitting and overfitting in train/validation curves | `s02_f13_train_validation_dynamics.png` | PNG / plot | TODO |
| **S02-F14** | V32 | Effect of weight decay on learned functions or weight norms | `s02_f14_weight_decay_effect.png` | PNG | TODO |
| **S02-F15** | V34 | Dropout as stochastic subnetworks during training | `s02_f15_dropout_subnetworks.png` | PNG | TODO |
| **S02-F16** | V36 | Depth degradation: deeper plain networks can be harder to optimize | `s02_f16_depth_degradation.png` | PNG / plot | TODO |
| **S02-F17** | V37 | Residual block: transformation path plus identity path | `s02_f17_residual_block.png` | PNG | TODO |
| **S02-F18** | V39 | Gradient flow through the identity path in a residual block | `s02_f18_residual_gradient_path.png` | PNG | Covered by S02-A03 |
| **S02-F19** | V41 | Map from symptoms to measurements and training interventions | `s02_f19_training_diagnostics_map.png` | PNG | TODO |
| **S02-F20** | V43 | Integrated view of mechanisms that make deep training possible | `s02_f20_deep_training_summary.png` | PNG | TODO |

## Recommended animations

Only animate concepts where motion changes understanding.

| ID | Related figure | Concept | Planned rendered file | Status |
|---|---|---|---|---|
| **S02-A01** | S02-F02 / S02-F03 | Activations shrink, explode or stabilize layer by layer | `s02_a01_signal_propagation.mp4` | Included (V11) |
| **S02-A02** | S02-F08 | SGD oscillates in a narrow valley while Momentum accumulates useful direction | `s02_a02_momentum_valley.mp4` | Included (V22) |
| **S02-A03** | S02-F17 / S02-F18 | Residual block forward path and direct gradient route | `s02_a03_residual_gradient_path.mp4` | Included (V39) |

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
