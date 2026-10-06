---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
title: SI7011 — Session 01 — Learning
description: How does a neural network learn?
style: |
  section {
    background: #ffffff;
    color: #172033;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 28px;
    padding: 46px 58px;
  }
  h1 {
    color: #153b63;
    font-size: 44px;
    margin-bottom: 0.35em;
  }
  h2 {
    color: #153b63;
    font-size: 34px;
  }
  strong {
    color: #146c72;
  }
  blockquote {
    border-left: 5px solid #d38b33;
    background: #f6f8fb;
    padding: 16px 22px;
    font-size: 23px;
  }
  code {
    font-size: 0.82em;
  }
  table {
    font-size: 22px;
  }
  .small {
    font-size: 22px;
  }
  .tiny {
    font-size: 18px;
  }
  .center {
    text-align: center;
  }
  section.media video {
    display: block;
    width: 100%;
    height: 460px;
    object-fit: contain;
    background: #ffffff;
  }
  section.media p { margin: 8px 0; }
  section.media a { font-size: 21px; }
  .print-poster { display: none; }
  @media print {
    section.media video { display: none; }
    section.media .print-poster {
      display: block;
      width: 100%;
      height: 460px;
      object-fit: contain;
    }
  }
  section.figure { padding: 0; }
  .placeholder {
    border: 2px dashed #a8b1bd;
    border-radius: 12px;
    padding: 28px;
    margin-top: 18px;
    color: #5b6775;
    background: #fafbfc;
    text-align: center;
    font-size: 22px;
  }
footer: SI7011 — Deep Learning
---

# How does a neural network learn?

<div class="center">

$$
x \longrightarrow f_\theta(x) \longrightarrow \hat y
\longrightarrow L
\longrightarrow \nabla_\theta
\longrightarrow \theta'
$$

</div>

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Supervised learning: data, model and prediction](../figures/s01_f01_supervised_learning_data_to_prediction.png)

---

# We observe data

$$
\mathcal D=\{(x_i,y_i)\}_{i=1}^{N}
$$

We start from **observed examples**, not from explicit rules.

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain From data to a learning problem](../figures/s01_f02_from_data_to_learning_problem.png)



# The input

$$
x
$$

The information available **before** making a prediction.

- tabular attributes
- images
- sequences
- signals

<!-- Pending visual: VISUAL — multiple data modalities converging to $x$ -->

---

# The target

$$
y
$$

The value or class we want to predict.

$$
(x,y)
$$

The target is combined with the prediction to compute the loss during training.

<!-- Pending visual: VISUAL — input $x$ and target $y$ as an observed pair -->

---

# A model

$$
f_\theta
$$

A model is a **parameterized function**.

$$
x \longrightarrow \boxed{f_\theta} \longrightarrow ?
$$

<!-- Pending visual: FIGURE S01-F03 — Data → model -->

---

# Prediction

$$
\hat y=f_\theta(x)
$$

Inference means using the **current parameters** to produce an output.

<!-- Pending visual: VISUAL — $x \rightarrow f_\theta(x) \rightarrow \hat y$ -->

---

# Is the prediction good?

$$
y \qquad\qquad \hat y
$$

How do we transform the quality of a prediction into something we can optimize?

<!-- Pending visual: VISUAL — target and prediction separated by an explicit error gap -->

---

# Loss

$$
L(y,\hat y)
$$

For regression, one possibility is:

$$
L=(y-\hat y)^2
$$

A loss converts the task objective into a **scalar quantity**.

<!-- Pending visual: VISUAL — prediction and target on a line with squared error -->

---

# Loss depends on the parameters

$$
\hat y=f_\theta(x)
$$

therefore,

$$
L(y,\hat y)
=
L(y,f_\theta(x))
$$

and we can reason about:

$$
L(\theta)
$$

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain How parameters determine the loss](../figures/s01_f05_parameter_dependency.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Binary classification: logits, sigmoid and binary cross-entropy](../figures/s01_f05a_binary_logits_sigmoid_bce.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Binary cross-entropy loss curves](../figures/s01_f05b_binary_cross_entropy_curves.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Multiclass classification: logits, softmax and cross-entropy](../figures/s01_f05c_multiclass_logits_softmax_cross_entropy.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Cross-entropy intuition: confidence matters](../figures/s01_f05d_cross_entropy_confidence.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain What the loss function expects in PyTorch](../figures/s01_f05e_pytorch_loss_expectations.svg)

---

# Training is not inference

| Inference | Training |
|---|---|
| $x \rightarrow f_\theta(x)\rightarrow \hat y$ | $(x,y)\rightarrow f_\theta(x)\rightarrow\hat y\rightarrowL\rightarrow\theta'$ |
| parameters are **used** | parameters are **changed** |

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Training versus inference](../figures/s01_f04_training_vs_inference.svg)

---

# One parameter

Suppose:

$$
\hat y = wx
$$

Changing $w$ changes the prediction and therefore changes the loss.

<!-- Pending visual: VISUAL — curve $L(w)$ with current parameter $w_t$ -->

---

# The gradient

$$
\frac{\partial L}{\partial w}
$$

The gradient gives **local information** about how the loss changes.

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain The gradient is local information](../figures/s01_f06_local_slope.svg)

---

# Gradient descent

$$
w_{t+1}
=
w_t-\eta
\frac{\partial L}{\partial w}
$$

Two ingredients:

- **direction** from the gradient
- **step size** from the learning rate

<!-- Pending visual: VISUAL — one optimization step on $L(w)$ -->

---

# Learning rate

$$
\eta
$$

What changes if the step is:

- too small?
- appropriate?
- too large?

<!-- Pending visual: FIGURE S01-F07 — Three learning rates -->

---

# More than one parameter

$$
\theta=(w_1,w_2,\ldots,w_p)
$$

$$
\nabla_\theta L
=
\begin{bmatrix}
\partial L/\partial w_1\\
\vdots\\
\partial L/\partial w_p
\end{bmatrix}
$$

<!-- Pending visual: FIGURE S01-F08 — Loss surface -->

---

# From example loss to dataset objective

For one observation:

$$
L_i(\theta)=L\!\left(y_i,f_\theta(x_i)\right)
$$

For the full dataset:

$$
J(\theta)
=
\frac{1}{N}
\sum_{i=1}^{N}
L_i(\theta)
$$

**Convention:** $L$ is a per-example loss; $J$ is the dataset objective.

---

# Batch, SGD and mini-batch

For a subset $\mathcal B$:

$$
J_{\mathcal B}(\theta)
=
\frac{1}{|\mathcal B|}
\sum_{i\in\mathcal B}
L_i(\theta)
$$

- **batch:** $\mathcal B=\mathcal D$
- **SGD:** $|\mathcal B|=1$
- **mini-batch:** $1<|\mathcal B|<N$

The update uses $g_t=\nabla_\theta J_{\mathcal B_t}(\theta_t)$.

<!-- Pending visual: FIGURE S01-F10 — Batch / SGD / mini-batch -->

---

# Optimization is a trajectory

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

$$
\theta_0,\theta_1,\theta_2,\ldots
$$

Training moves through parameter space using successive gradient estimates.

---

# What can a linear model represent?

Optimization can work perfectly while the model still lacks enough capacity.

<!-- Pending visual: FIGURE S01-F11 — Linear vs nonlinear structure -->

---

# Stacking linear transformations

$$
h=W_1x+b_1
$$

$$
y=W_2h+b_2
$$

Therefore:

$$
y=W_2W_1x+W_2b_1+b_2
$$

Several linear layers can collapse into a single affine transformation.

---

# Nonlinearity changes the game

$$
z=Wx+b,\qquad a=\phi(z)
$$

A nonlinear activation prevents the composition from collapsing into one linear map.

<!-- Pending visual: VISUAL — insert a nonlinear transformation between two affine maps -->

---

<!-- _class: media -->

# Affine maps and nonlinear activations

<video controls preload="none" poster="../figures/s01_transformation_overview.png" aria-label="Affine maps and nonlinear activations">
  <source src="../figures/E9_EjemploActivaciones.mp4" type="video/mp4">
</video>
<img class="print-poster" src="../figures/s01_transformation_overview.png" alt="Static view of Affine maps and nonlinear activations">

[Open animation](../figures/E9_EjemploActivaciones.mp4)

<!-- Compare input coordinates, the affine transformation, tanh and ReLU. Which representations allow a linear separator? -->

---

# ReLU

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

A very simple nonlinear function can become a powerful building block.

<!-- Pending visual: VISUAL — clean ReLU plot -->

---

# One ReLU as a building block

A single unit produces a simple piecewise-linear transformation.

<!-- Pending visual: VISUAL — one shifted/scaled ReLU component -->

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Two shifted and scaled ReLU components and their sum](../figures/s01_f12_two_relu_components.png)

---

<!-- _class: media -->

# Approximation with ReLU components

<video controls preload="none" poster="../figures/s01_f12_two_relu_components.png" aria-label="Approximation with ReLU components">
  <source src="../figures/E7_AproximacionReLU.mp4" type="video/mp4">
</video>
<img class="print-poster" src="../figures/s01_f12_two_relu_components.png" alt="Static view of Approximation with ReLU components">

[Open animation](../figures/E7_AproximacionReLU.mp4)

<!-- As the number of components grows, identify the new breakpoints. This animation changes capacity, without training parameters. -->

---

# A hidden layer

$$
z_1=W_1x+b_1,\qquad a_1=\phi(z_1)
$$

$$
\hat y=W_2a_1+b_2
$$

A hidden layer learns an **intermediate representation**.

<!-- Pending visual: VISUAL — hidden units as learned features -->

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain MLP as a composition of affine maps and activations](../figures/s01_f13_MLP%20Function%20Composition%20Diagram.png)

---

# Forward pass

$$
x
\rightarrow
z_1
\rightarrow
a_1
\rightarrow
z_2
\rightarrow
a_2
\rightarrow
\hat y
\rightarrow
L
$$

The forward pass evaluates the composed function.

<!-- Pending visual: VISUAL — values propagating left → right -->

---

# Computational graph

Each hidden layer contains an affine map and an activation:

$$
z_l=W_la_{l-1}+b_l,\qquad a_l=\phi(z_l),\qquad a_0=x
$$

For a linear output and squared loss:

$$
\hat y=W_3a_2+b_3,\qquad L=\frac{1}{2}\lVert\hat y-y\rVert^2
$$

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Forward pass through linear and nonlinear layers](../figures/s01_f14_computational_graph_forward.png)

---

# Chain rule

If:

$$
L=L(\hat y(a(w)))
$$

then:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial a}
\frac{\partial a}{\partial w}
$$

<!-- Pending visual: VISUAL — equation aligned with computational graph -->

---

# Backward pass

The same graph is traversed in the opposite direction to propagate gradients.

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Backpropagation through linear and nonlinear layers with local equations](../figures/s01_f15_backward%20gradients%20in%20neural%20network.png)



# Gradients for every parameter

Each trainable parameter receives:

$$
\frac{\partial L}{\partial w_i}
$$

The backward pass produces the information required by the optimizer.

<!-- Pending visual: VISUAL — parameter nodes annotated with local gradients -->

---

# Backpropagation is not gradient descent

| Backpropagation | Optimizer |
|---|---|
| computes $\nabla_\theta L$ | uses $\nabla_\theta L$ |
| differentiates the graph | updates the parameters |

$$
\text{backward} \neq \text{update}
$$

---

# Automatic differentiation

Manual view:

$$
\text{operations}
\rightarrow
\text{local derivatives}
\rightarrow
\text{chain rule}
$$

PyTorch:

```python
loss.backward()
```

Autograd automates differentiation of the computational graph.

---

# The PyTorch training loop

```python
optimizer.zero_grad()

y_hat = model(x)
loss = criterion(y_hat, y)

loss.backward()
optimizer.step()
```

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain From mathematics to PyTorch](../figures/s01_f16_math_pytorch_mapping.svg)

---

<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain Full learning loop: forward, loss, backward and parameter update](../figures/s01_f01_learning_loop.png)

---

# What changes in deep learning?

The learning loop remains the same.

What changes is:

- $f_\theta$
- architecture
- loss
- optimizer
- data
- scale

CNNs, RNNs, ViTs and Transformers do **not** replace the learning loop.

---

# Exit question

A student says:

> “Backpropagation updates the weights to minimize the loss.”

What is incorrect in that statement?

---

# Guided practice — PyTorch in one hour

[Open notebook](../notebooks/sesion_01_pytorch_60min.ipynb)

- tensors and affine transformations
- scalar autograd and parameter updates
- `TensorDataset` and `DataLoader`
- MLP for nonlinear classification
- loss curves and decision boundaries

> Predict the behavior before running each experiment.

---

# Exercise — Multiclass MLP with tabular data

[Open notebook](../notebooks/sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb)

Dry Bean dataset from Kaggle: 16 attributes and 7 classes.

- split and standardize the data
- create a `DataLoader`
- implement an MLP and the training loop
- use `CrossEntropyLoss`
- evaluate accuracy, macro F1 and the confusion matrix

---

# Session 01 — Mental model

$$
\boxed{
\text{forward}
\rightarrow
\text{loss}
\rightarrow
\text{backward}
\rightarrow
\text{update}
}
$$

If this loop is clear, the rest of the course has a common foundation.
