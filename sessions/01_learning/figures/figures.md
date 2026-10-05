# SI7011 — Session 01

## Figure and animation inventory

The slide references below match the current order of `../slides/S01_slides.marp.md`.
V01 is the cover. Figure-only and animation slides count in the numbering.
The production IDs identify concepts, independently of historical filenames.

White backgrounds, preserved image proportions and the existing course style apply throughout.

## Available and included

| ID | Current slide(s) | Purpose | Format | Repository file | Status |
|---|---|---|---|---|---|
| **S01-F01** | V49 | Full learning loop: forward, loss, backward and parameter update | SVG / PNG | [SVG](s01_f01_learning_loop.svg), [PNG](s01_f01_learning_loop.png) | Included |
| **S01-F01A** | V02 | Data, model and prediction | PNG | [s01_f01_supervised_learning_data_to_prediction.png](s01_f01_supervised_learning_data_to_prediction.png) | Included |
| **S01-F02** | V04 | From observed data to a learning problem | PNG | [s01_f02_from_data_to_learning_problem.png](s01_f02_from_data_to_learning_problem.png) | Included |
| **S01-F03A** | V05 | Training, validation and test partitions | PNG | [s01_f03_dataset_splits.png](s01_f03_dataset_splits.png) | Included |
| **S01-F04** | V15 | Training versus inference | PNG | [s01_f04_Training vs. Inference Pipeline.png](s01_f04_Training%20vs.%20Inference%20Pipeline.png) | Included |
| **S01-F05** | V13 | Parameter dependency chain | SVG | [s01_f05_parameter_dependency.svg](s01_f05_parameter_dependency.svg) | Included |
| **S01-F06** | V18 | Derivative as local slope | SVG | [s01_f06_local_slope.svg](s01_f06_local_slope.svg) | Included |
| **S01-F07B** | V21 | Learning-rate schedules | PNG | [s01_f12b_Learning Rate Scheduling Strategies.png](s01_f12b_Learning%20Rate%20Scheduling%20Strategies.png) | Included |
| **S01-F09** | V24 | Optimizer trajectories and illustrative loss curves | PNG | [s01_f12_Optimizer Trajectories and Training Loss Comparison.png](s01_f12_Optimizer%20Trajectories%20and%20Training%20Loss%20Comparison.png) | Included |
| **S01-F12A** | V33, V34 | Two shifted and scaled ReLU components and their sum | PNG | [s01_f12_two_relu_components.png](s01_f12_two_relu_components.png) | Included |
| **S01-F12B** | V34 | Progressive approximation by sums of ReLU components | MP4 / Manim | [E7_AproximacionReLU.mp4](E7_AproximacionReLU.mp4) | Included |
| **S01-F13** | V36 | MLP as function composition | PNG | [s01_f13_MLP Function Composition Diagram.png](s01_f13_MLP%20Function%20Composition%20Diagram.png) | Included |
| **S01-F13A** | V30 | Input, affine map, tanh and ReLU representations | MP4 / Manim | [E9_EjemploActivaciones.mp4](E9_EjemploActivaciones.mp4) | Included |
| **S01-F13A-P** | V30 | Static view of the activation example | PNG | [s01_transformation_overview.png](s01_transformation_overview.png) | Poster / static fallback |
| **S01-F14** | V39, V43 | Forward through linear and nonlinear layers | PNG | [s01_f14_computational_graph_forward.png](s01_f14_computational_graph_forward.png) | Included |
| **S01-F15** | V42 | Backward through linear and nonlinear layers with equations | PNG | [s01_f15_backward gradients in neural network.png](s01_f15_backward%20gradients%20in%20neural%20network.png) | Included |
| **S01-F15B** | V43 | Animated forward and backward through the MLP | MP4 / Manim | [E10_GrafoForwardBackward.mp4](E10_GrafoForwardBackward.mp4) | Included |
| **S01-F16** | V48 | Mathematics and the PyTorch training loop | SVG | [s01_f16_math_pytorch_mapping.svg](s01_f16_math_pytorch_mapping.svg) | Included |

The two optimization images retain their existing filenames beginning with `s01_f12`.
They belong to F07B and F09, respectively. They do not represent ReLU components.
F12A uses the previously created `s01_f12_two_relu_components.png`.
F12B uses the rendered Manim scene `E7_AproximacionReLU.mp4`.

## Pending production

These resources have no final corresponding asset in the repository. Their notes remain in
the slide source, without displaying broken links or production labels to students.

| ID | Current slide(s) | Purpose | Planned filename | Status |
|---|---|---|---|---|
| **S01-F03** | V08 | Model as a parameterized function | `s01_f03_data_model.svg` | TODO |
| **S01-F07** | V20 | Small, suitable and excessive learning rates | `s01_f07_learning_rates.png` | TODO |
| **S01-F08** | V22 | Loss surface | `s01_f08_loss_surface.png` | TODO |
| **S01-F10** | V26 | Batch strategies | `s01_f10_batch_strategies.svg` | TODO |
| **S01-F11** | V27 | Linear and nonlinear model capacity | `s01_f11_linear_vs_nonlinear.png` | TODO |

## Previous versions

- `s01_f14_computational_graph.svg` and `s01_f15_forward_backward.svg` remain available as previous scalar examples. The presentation uses the new PNG figures with linear and nonlinear layers.
- The earlier planned names `s01_f12a_two_relu_components.png`, `s01_f12b_relu_approximation.mp4` and `s01_f13_mlp_composition.svg` are not references in the current deck.

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
