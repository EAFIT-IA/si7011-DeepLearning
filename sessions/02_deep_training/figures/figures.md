# SI7011 — Session 02

## Figure and animation inventory

Session 02 focuses on the question:

> **Why can we train deep neural networks?**

The slide deck should use placeholders until figures are produced. Figures should keep white backgrounds, academic visual style, preserved proportions and filenames without spaces.

## Planned figures

| ID | Current slide(s) | Purpose | Planned filename | Recommended format | Status |
|---|---:|---|---|---|---|
| **S02-F01** | V02 | From the single learning loop to a deep trainable system | `s02_f01_learning_to_deep_training.png` | PNG | TODO |
| **S02-F02** | V06 | Activation distributions propagating through many layers | `s02_f02_signal_propagation.png` | PNG | TODO |
| **S02-F03** | V08 | Vanishing, stable and exploding activation scales across depth | `s02_f03_activation_scale_depth.png` | PNG / plot | TODO |
| **S02-F04** | V10 | Gradient norm versus layer depth for vanishing, stable and exploding regimes | `s02_f04_gradient_norm_depth.png` | PNG / plot | TODO |
| **S02-F05** | V11 | Sigmoid, tanh and ReLU: functions and derivative regions | `s02_f05_activation_derivatives.png` | PNG | TODO |
| **S02-F06** | V13 | Symmetry problem when hidden units share identical initialization | `s02_f06_initialization_symmetry.png` | PNG / SVG | TODO |
| **S02-F07** | V17 | Naive, Xavier and He initialization compared through activation variance | `s02_f07_initialization_variance.png` | PNG / plot | TODO |
| **S02-F08** | V22 | SGD and Momentum trajectories in a narrow valley | `s02_f08_sgd_momentum_valley.png` | PNG | TODO |
| **S02-F09** | V23 | SGD, Momentum and Adam update geometry on the same surface | `s02_f09_optimizer_trajectories.png` | PNG | TODO |
| **S02-F10** | V24 | Fixed, step, exponential and cosine learning-rate schedules | `s02_f10_learning_rate_schedules.png` | PNG / plot | TODO |
| **S02-F11** | V26 | BatchNorm: normalize, then learn scale and shift | `s02_f11_batchnorm_transform.png` | PNG | TODO |
| **S02-F12** | V27 | BatchNorm behavior in training mode versus evaluation mode | `s02_f12_batchnorm_train_eval.png` | PNG / SVG | TODO |
| **S02-F13** | V28 | Underfitting, healthy fitting and overfitting in train/validation curves | `s02_f13_train_validation_dynamics.png` | PNG / plot | TODO |
| **S02-F14** | V29 | Effect of weight decay on learned functions or weight norms | `s02_f14_weight_decay_effect.png` | PNG | TODO |
| **S02-F15** | V30 | Dropout as stochastic subnetworks during training | `s02_f15_dropout_subnetworks.png` | PNG | TODO |
| **S02-F16** | V31 | Depth degradation: deeper plain networks can be harder to optimize | `s02_f16_depth_degradation.png` | PNG / plot | TODO |
| **S02-F17** | V32 | Residual block: transformation path plus identity path | `s02_f17_residual_block.png` | PNG | TODO |
| **S02-F18** | V33 | Gradient flow through the identity path in a residual block | `s02_f18_residual_gradient_path.png` | PNG | TODO |
| **S02-F19** | V35 | Map from symptoms to measurements and training interventions | `s02_f19_training_diagnostics_map.png` | PNG | TODO |
| **S02-F20** | V37 | Integrated view of mechanisms that make deep training possible | `s02_f20_deep_training_summary.png` | PNG | TODO |

## Recommended animations

Only animate concepts where motion changes understanding.

| ID | Related figure | Concept | Planned rendered file | Status |
|---|---|---|---|---|
| **S02-A01** | S02-F02 / S02-F03 | Activations shrink, explode or stabilize layer by layer | `s02_a01_signal_propagation.mp4` | TODO |
| **S02-A02** | S02-F08 | SGD oscillates in a narrow valley while Momentum accumulates useful direction | `s02_a02_momentum_valley.mp4` | TODO |
| **S02-A03** | S02-F17 / S02-F18 | Residual block forward path and direct gradient route | `s02_a03_residual_gradient_path.mp4` | TODO |

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
- The current slide deck intentionally uses placeholders instead of broken image links.
