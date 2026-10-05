# SI7011 — Session 01
## Figure production list

Visual assets required for **S01 — Learning**.

### Visual rules

- White background and minimal visual clutter.
- One dominant idea per figure.
- Prefer Manim/SVG for mathematical or algorithmic transformations.
- Prefer Matplotlib for experimental plots.
- Use academic illustration only when it contributes intuition.
- Preserve the same visual grammar across the session.

| ID | Slide(s) | Purpose | Preferred format | Filename | Status |
|---|---|---|---|---|---|
| **S01-F01** | V01, V36 | Full learning loop | Manim/SVG | `s01_f01_learning_loop.svg` | generated, pending final asset |
| **S01-F02** | V02 | Dataset as observed pairs | illustration | `s01_f02_observed_pairs.png` | TODO |
| **S01-F03** | V05 | Model as parameterized function | SVG | `s01_f03_data_model.svg` | TODO |
| **S01-F04** | V07 | Training vs inference | illustration | `s01_f04_training_vs_inference.png` | generated, pending final asset |
| **S01-F05** | V10 | Parameter dependency chain | SVG | `s01_f05_parameter_dependency.svg` | **DONE** |
| **S01-F06** | V12 | Derivative as local slope | SVG/Matplotlib | `s01_f06_local_slope.svg` | **DONE** |
| **S01-F07** | V14 | Learning-rate regimes | Matplotlib/Manim | `s01_f07_learning_rates.png` | generated, pending final asset |
| **S01-F08** | V15 | Loss surface for multiple parameters | Matplotlib | `s01_f08_loss_surface.png` | TODO |
| **S01-F09** | V16 | Optimization trajectory | Matplotlib/Manim | `s01_f09_optimization_path.png` | TODO |
| **S01-F10** | V18 | Batch / SGD / mini-batch | SVG | `s01_f10_batch_strategies.svg` | TODO |
| **S01-F11** | V19 | Linear vs nonlinear capacity | Matplotlib | `s01_f11_linear_vs_nonlinear.png` | generated, pending final asset |
| **S01-F12A** | V24 | Two ReLU components | Matplotlib/Manim | `s01_f12a_two_relu_components.png` | TODO |
| **S01-F12B** | V25 | Progressive ReLU approximation | Manim | `s01_f12b_relu_approximation.mp4` | generated static draft |
| **S01-F13** | V27 | MLP as function composition | SVG | `s01_f13_mlp_composition.svg` | TODO |
| **S01-F14** | V29 | Computational graph | SVG/Manim | `s01_f14_computational_graph.svg` | **DONE** |
| **S01-F15** | V31 | Forward values / backward gradients | SVG/Manim | `s01_f15_forward_backward.svg` | **DONE** |
| **S01-F16** | V35 | Mathematics ↔ PyTorch | SVG | `s01_f16_math_pytorch_mapping.svg` | **DONE** |

## Priority

**A:** F01, F04, F07, F11, F12B, F14, F15, F16  
**B:** F05, F06, F08, F09, F10, F13  
**C:** F02, F03, F12A

## Visual sequence

The following figures should look like progressive views of the same process:

```text
S01-F03
   ↓
S01-F05
   ↓
S01-F01
   ↓
S01-F14
   ↓
S01-F15
   ↓
S01-F16
```

The goal is for students to recognize a single learning process rather than unrelated diagrams.
