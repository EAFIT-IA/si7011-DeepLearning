# SI7011 — Session 01

## Figure and animation inventory

The slide references below match the current order of `../slides/S01_slides.marp.md`.
V01 is the cover. Figure-only and animation slides count in the numbering.
The production IDs identify concepts, independently of historical filenames.

White backgrounds, preserved image proportions and the existing course style apply throughout.

## Available and included

| ID | Current slide(s) | Purpose | Format | Repository file | Status |
|---|---|---|---|---|---|
| **S01-F01** | V59 | Full learning loop: forward, loss, backward and parameter update | SVG | [s01_f01_learning_loop.svg](s01_f01_learning_loop.svg) | Included |
| **S01-F01A** | V02 | Data, model and prediction | PNG | [s01_f01_supervised_learning_data_to_prediction.png](s01_f01_supervised_learning_data_to_prediction.png) | Included |
| **S01-F02** | V04 | From observed data to a learning problem | PNG | [s01_f02_from_data_to_learning_problem.png](s01_f02_from_data_to_learning_problem.png) | Included |
| **S01-F04** | V21 | Training versus inference | SVG | [s01_f04_training_vs_inference.svg](s01_f04_training_vs_inference.svg) | Included |
| **S01-F05** | V13 | Parameter dependency chain | SVG | [s01_f05_parameter_dependency.svg](s01_f05_parameter_dependency.svg) | Included |
| **S01-F05A** | V14 | Binary classification: one logit, sigmoid and BCE | SVG | [s01_f05a_binary_logits_sigmoid_bce.svg](s01_f05a_binary_logits_sigmoid_bce.svg) | Included |
| **S01-F05B** | V15 | Typical binary cross-entropy loss curves for y=1 and y=0 | SVG | [s01_f05b_binary_cross_entropy_curves.svg](s01_f05b_binary_cross_entropy_curves.svg) | Included |
| **S01-F05C** | V17 | Multiclass classification: C logits, softmax and cross-entropy | SVG | [s01_f05c_multiclass_logits_softmax_cross_entropy.svg](s01_f05c_multiclass_logits_softmax_cross_entropy.svg) | Included |
| **S01-F05D** | V18 | Cross-entropy intuition: confidence and confident errors | SVG | [s01_f05d_cross_entropy_confidence.svg](s01_f05d_cross_entropy_confidence.svg) | Included |
| **S01-F05E** | V19 | PyTorch inputs for MSELoss, BCEWithLogitsLoss and CrossEntropyLoss | SVG | [s01_f05e_pytorch_loss_expectations.svg](s01_f05e_pytorch_loss_expectations.svg) | Included |
| **S01-F06** | V24 | Derivative as local slope | SVG | [s01_f06_local_slope.svg](s01_f06_local_slope.svg) | Included |
| **S01-F12A** | V41, V42 | Two shifted and scaled ReLU components and their sum | PNG | [s01_f12_two_relu_components.png](s01_f12_two_relu_components.png) | Included |
| **S01-F12B** | V42 | Progressive approximation by sums of ReLU components | MP4 / Manim | [E7_AproximacionReLU.mp4](E7_AproximacionReLU.mp4) | Included |
| **S01-F13** | V45 | MLP as a composition of affine maps and generic activation $\phi$ | SVG | [s01_f13_mlp_composition.svg](s01_f13_mlp_composition.svg) | Included |
| **S01-F13A** | V38 | Input, affine map, tanh and ReLU representations | MP4 / Manim | [E9_EjemploActivaciones.mp4](E9_EjemploActivaciones.mp4) | Included |
| **S01-F13A-P** | V38 | Static view of the activation example | PNG | [s01_transformation_overview.png](s01_transformation_overview.png) | Poster / static fallback |
| **S01-F14** | V49, V53 | Forward values through affine and nonlinear layers | SVG | [s01_f14_computational_graph_forward.svg](s01_f14_computational_graph_forward.svg) | Included |
| **S01-F15** | V52 | Backward signals $\delta$ and parameter gradients $\nabla L$ | SVG | [s01_f15_backward_gradients.svg](s01_f15_backward_gradients.svg) | Included |
| **S01-F15B** | V53 | Animated forward values and backward signals ($\phi$, $\delta$, $\nabla L$) | MP4 / Manim | [E10_GrafoForwardBackward.mp4](E10_GrafoForwardBackward.mp4) | Included |
| **S01-F16** | V58 | Mathematics and the PyTorch training loop | SVG | [s01_f16_math_pytorch_mapping.svg](s01_f16_math_pytorch_mapping.svg) | Included |
| **S01-A01** | V12 | Parameters, residuals, MSE and level curves of J(w, b) | MP4 / Manim | [E1_ParametrosPerdida.mp4](E1_ParametrosPerdida.mp4) | Included |
| **S01-A02** | V16 | Score z, sigmoid probability and BCE for confident errors | MP4 / Manim | [E2_PuntajeProbabilidad.mp4](E2_PuntajeProbabilidad.mp4) | Included |
| **S01-A03** | V26 | Derivative, exact gradient steps on L(w), and gradient of J on level curves | MP4 / Manim | [E3_DescensoGradiente.mp4](E3_DescensoGradiente.mp4) | Included |
| **S01-A04** | V28 | Three learning rates on the same parabola | MP4 / Manim | [E4_TasaAprendizaje.mp4](E4_TasaAprendizaje.mp4) | Included |
| **S01-A11** | V31 | Each parameter as a knob; gradient components and GD steps on J | MP4 / Manim | [E11_PerillasGradiente.mp4](E11_PerillasGradiente.mp4) | Included |
| **S01-A05** | V37 | Affine composition collapses; ReLU adds breakpoints; ReLU vs sigmoid | MP4 / Manim | [E5_Activacion.mp4](E5_Activacion.mp4) | Included |
| **S01-A06** | V44 | Trained 2–2–1 MLP: affine map, ReLU fold, linear boundary in a1 | MP4 / Manim | [E6_TransformacionEspacio.mp4](E6_TransformacionEspacio.mp4) | Included |
| **S01-A08** | V46 | Trained 2–2–2–2–1 tanh MLP: each layer bends the space | MP4 / Manim | [E8_CapasTransformanEspacio.mp4](E8_CapasTransformanEspacio.mp4) | Included |

Learning-rate schedules and optimizer comparisons moved to S02. The former pending figures F07 (learning rates) and F08 (loss surface) are covered by the animations A04 and A03. F12A uses `s01_f12_two_relu_components.png`.
F12B uses the rendered Manim scene `E7_AproximacionReLU.mp4`.

## Pending production

These resources have no final corresponding asset in the repository. Their notes remain in
the slide source, without displaying broken links or production labels to students.

| ID | Current slide(s) | Purpose | Planned filename | Status |
|---|---|---|---|---|
| **S01-F03** | V07 | Model as a parameterized function | JPG | [s01_f03_data_model.jpg](s01_f03_data_model.jpg) | Included |
| **S01-F10** | V32 | Batch, SGD and mini-batch gradient estimation | JPG | [s01_f10_batch_strategies.jpg](s01_f10_batch_strategies.jpg) | Included |
| **S01-F11** | V34 | Linear and nonlinear model capacity | `s01_f11_linear_vs_nonlinear.png` | TODO |

## Animation sources

| Rendered animation | Source | Scene |
|---|---|---|
| `E1_ParametrosPerdida.mp4` | [escenas_s01.py](escenas_s01.py) | `E1_ParametrosPerdida` |
| `E2_PuntajeProbabilidad.mp4` | [escenas_s01.py](escenas_s01.py) | `E2_PuntajeProbabilidad` |
| `E3_DescensoGradiente.mp4` | [escenas_s01.py](escenas_s01.py) | `E3_DescensoGradiente` |
| `E4_TasaAprendizaje.mp4` | [escenas_s01.py](escenas_s01.py) | `E4_TasaAprendizaje` |
| `E5_Activacion.mp4` | [escenas_s01.py](escenas_s01.py) | `E5_Activacion` |
| `E6_TransformacionEspacio.mp4` | [escenas_s01.py](escenas_s01.py) | `E6_TransformacionEspacio` |
| `E7_AproximacionReLU.mp4` | [escena_relu.py](escena_relu.py) | `E7_AproximacionReLU` |
| `E8_CapasTransformanEspacio.mp4` | [escena_capas.py](escena_capas.py) | `E8_CapasTransformanEspacio` |
| `E9_EjemploActivaciones.mp4` | [escena_activaciones.py](escena_activaciones.py) | `E9_EjemploActivaciones` |
| `E10_GrafoForwardBackward.mp4` | [escena_grafo.py](escena_grafo.py) | `E10_GrafoForwardBackward` |
| `E11_PerillasGradiente.mp4` | [escena_perillas.py](escena_perillas.py) | `E11_PerillasGradiente` |

E1–E6 share styles, data and helpers in [comun.py](comun.py) (fixed seeds).
Render any scene from this folder with `S01_VIDEO=1 manim -qh <file>.py <Scene>`
(Manim Community 0.21, LaTeX; E10 and E11 use the Inter font). Without `S01_VIDEO=1`
and with `manim-slides` installed, each pause becomes a stop for student predictions.
Each new animation slide uses a still of its final frame (`s01_eN_poster.png`) as poster.

HTML presentations embed the videos with playback controls and a static poster.
Each animation also has a relative link for viewers and static exports.
The ReLU animation uses F12A as its introductory poster. The activation example
uses F13A-P, and the forward/backward animation uses the forward diagram F14.
The deck has 64 slides.
