# Session 01 slides · How does a neural network learn?

**[Download the slides (PDF)](S01_slides.pdf)**

The PDF cannot play videos, so the slides with animations show a still image. These are the animations; click to watch them:

| Animation | What it shows |
|---|---|
| [E1 · Parameters and loss](../figures/E1_ParametrosPerdida.mp4) | How changing the parameters moves the residuals, the MSE and the level curves of $J(w, b)$ |
| [E2 · Score and probability](../figures/E2_PuntajeProbabilidad.mp4) | From a score $z$ to a sigmoid probability, and why confident errors cost so much in BCE |
| [E3 · Gradient descent](../figures/E3_DescensoGradiente.mp4) | The derivative, gradient steps on $L(w)$, and the gradient of $J$ on its level curves |
| [E4 · Learning rate](../figures/E4_TasaAprendizaje.mp4) | Three learning rates on the same parabola |
| [E11 · Parameters as knobs](../figures/E11_PerillasGradiente.mp4) | Each parameter as a knob, and how the gradient tells you which way to turn it |
| [E5 · Activation](../figures/E5_Activacion.mp4) | Why stacking affine maps collapses into one, and how ReLU adds breakpoints |
| [E9 · Activations in action](../figures/E9_EjemploActivaciones.mp4) | One input through an affine map, tanh and ReLU |
| [E7 · Approximating with ReLUs](../figures/E7_AproximacionReLU.mp4) | Building a curve by adding ReLU pieces |
| [E6 · Transforming the space](../figures/E6_TransformacionEspacio.mp4) | A trained 2–2–1 MLP folds the plane until the classes become linearly separable |
| [E8 · Layers transform the space](../figures/E8_CapasTransformanEspacio.mp4) | Each layer of a deeper tanh MLP bends the space a little more |
| [E10 · Forward and backward](../figures/E10_GrafoForwardBackward.mp4) | Values flowing forward and gradient signals flowing backward through the graph |

Practice for this session: [notebooks](../notebooks/README.md).
