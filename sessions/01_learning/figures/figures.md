# SI7011 — Session 01

## Figure and animation inventory

The slide references below match the current order of `../slides/S01_slides.marp.md`.
V01 is the cover. Figure-only and animation slides count in the numbering.
The production IDs identify concepts, independently of historical filenames.

White backgrounds, preserved image proportions and the existing course style apply throughout.

## Available and included

| ID | Current slide(s) | Purpose | Format | Repository file | Status |
|---|---|---|---|---|---|
| **S01-F01** | V51 | Full learning loop: forward, loss, backward and parameter update | SVG | [s01_f01_learning_loop.svg](s01_f01_learning_loop.svg) | Included |
| **S01-F01A** | V02 | Data, model and prediction | PNG | [s01_f01_supervised_learning_data_to_prediction.png](s01_f01_supervised_learning_data_to_prediction.png) | Included |
| **S01-F02** | V04 | From observed data to a learning problem | PNG | [s01_f02_from_data_to_learning_problem.png](s01_f02_from_data_to_learning_problem.png) | Included |
| **S01-F04** | V19 | Training versus inference | SVG | [s01_f04_training_vs_inference.svg](s01_f04_training_vs_inference.svg) | Included |
| **S01-F05** | V12 | Parameter dependency chain | SVG | [s01_f05_parameter_dependency.svg](s01_f05_parameter_dependency.svg) | Included |
| **S01-F05A** | V13 | Binary classification: one logit, sigmoid and BCE | SVG | [s01_f05a_binary_logits_sigmoid_bce.svg](s01_f05a_binary_logits_sigmoid_bce.svg) | Included |
| **S01-F05B** | V14 | Typical binary cross-entropy loss curves for y=1 and y=0 | SVG | [s01_f05b_binary_cross_entropy_curves.svg](s01_f05b_binary_cross_entropy_curves.svg) | Included |
| **S01-F05C** | V15 | Multiclass classification: C logits, softmax and cross-entropy | SVG | [s01_f05c_multiclass_logits_softmax_cross_entropy.svg](s01_f05c_multiclass_logits_softmax_cross_entropy.svg) | Included |
| **S01-F05D** | V16 | Cross-entropy intuition: confidence and confident errors | SVG | [s01_f05d_cross_entropy_confidence.svg](s01_f05d_cross_entropy_confidence.svg) | Included |
| **S01-F05E** | V17 | PyTorch inputs for MSELoss, BCEWithLogitsLoss and CrossEntropyLoss | SVG | [s01_f05e_pytorch_loss_expectations.svg](s01_f05e_pytorch_loss_expectations.svg) | Included |
| **S01-F06** | V22 | Derivative as local slope | SVG | [s01_f06_local_slope.svg](s01_f06_local_slope.svg) | Included |
| **S01-F12A** | V35, V36 | Two shifted and scaled ReLU components and their sum | PNG | [s01_f12_two_relu_components.png](s01_f12_two_relu_components.png) | Included |
| **S01-F12B** | V36 | Progressive approximation by sums of ReLU components | MP4 / Manim | [E7_AproximacionReLU.mp4](E7_AproximacionReLU.mp4) | Included |
| **S01-F13** | V38 | MLP as a composition of affine maps and generic activation $\phi$ | SVG | [s01_f13_mlp_composition.svg](s01_f13_mlp_composition.svg) | Included |
| **S01-F13A** | V32 | Input, affine map, tanh and ReLU representations | MP4 / Manim | [E9_EjemploActivaciones.mp4](E9_EjemploActivaciones.mp4) | Included |
| **S01-F13A-P** | V32 | Static view of the activation example | PNG | [s01_transformation_overview.png](s01_transformation_overview.png) | Poster / static fallback |
| **S01-F14** | V41, V45 | Forward values through affine and nonlinear layers | SVG | [s01_f14_computational_graph_forward.svg](s01_f14_computational_graph_forward.svg) | Included |
| **S01-F15** | V44 | Backward signals $\delta$ and parameter gradients $\nabla L$ | SVG | [s01_f15_backward_gradients.svg](s01_f15_backward_gradients.svg) | Included |
| **S01-F15B** | V45 | Animated forward values and backward signals ($\phi$, $\delta$, $\nabla L$) | MP4 / Manim | [E10_GrafoForwardBackward.mp4](E10_GrafoForwardBackward.mp4) | Included |
| **S01-F16** | V50 | Mathematics and the PyTorch training loop | SVG | [s01_f16_math_pytorch_mapping.svg](s01_f16_math_pytorch_mapping.svg) | Included |

Learning-rate schedules and optimizer comparisons moved to S02. F12A uses `s01_f12_two_relu_components.png`.
F12B uses the rendered Manim scene `E7_AproximacionReLU.mp4`.

## Pending production

These resources have no final corresponding asset in the repository. Their notes remain in
the slide source, without displaying broken links or production labels to students.

| ID | Current slide(s) | Purpose | Planned filename | Status |
|---|---|---|---|---|
| **S01-F03** | V07 | Model as a parameterized function | `s01_f03_data_model.svg` | TODO |
| **S01-F07** | V24 | Small, suitable and excessive learning rates | `s01_f07_learning_rates.png` | TODO |
| **S01-F08** | V25 | Loss surface | `s01_f08_loss_surface.png` | TODO |
| **S01-F10** | V27 | Batch strategies | `s01_f10_batch_strategies.svg` | TODO |
| **S01-F11** | V29 | Linear and nonlinear model capacity | `s01_f11_linear_vs_nonlinear.png` | TODO |

## Animation sources

| Rendered animation | Source | Scene |
|---|---|---|
| `E7_AproximacionReLU.mp4` | [escena_relu.py](escena_relu.py) | `E7_AproximacionReLU` |
| `E9_EjemploActivaciones.mp4` | [escena_activaciones.py](escena_activaciones.py) | `E9_EjemploActivaciones` |
| `E10_GrafoForwardBackward.mp4` | [escena_grafo.py](escena_grafo.py) | `E10_GrafoForwardBackward` |

HTML presentations embed the videos with playback controls and a static poster.
Each animation also has a relative link for viewers and static exports.
The ReLU animation uses F12A as its introductory poster. The activation example
uses F13A-P, and the forward/backward animation uses the forward diagram F14.
The deck has 56 slides.
