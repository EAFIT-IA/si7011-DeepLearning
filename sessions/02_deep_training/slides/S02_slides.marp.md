---
marp: true
theme: default
paginate: true
math: mathjax
size: 16:9
title: SI7011 — Session 02 — Deep Training
description: Why can we train deep neural networks?
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
  .placeholder {
    border: 2px dashed #a8b1bd;
    border-radius: 12px;
    padding: 26px;
    margin-top: 18px;
    color: #5b6775;
    background: #fafbfc;
    text-align: center;
    font-size: 21px;
  }
  .placeholder strong {
    color: #153b63;
  }
  .twocol {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    align-items: start;
  }
footer: SI7011 — Deep Learning
---

# Why can we train deep neural networks?

<div class="center">

$$
\text{depth}
\rightarrow
\text{stability}
\rightarrow
\text{optimization}
\rightarrow
\text{regularization}
\rightarrow
\text{residual learning}
$$

</div>

---

# From learning to deep training

Session 01 established the learning loop:

$$
\text{forward}
\rightarrow
\text{loss}
\rightarrow
\text{backward}
\rightarrow
\text{update}
$$

Now we ask what happens when this loop is applied to **deep compositions**.

<div class="placeholder">
<strong>S02-F01</strong><br>
From the single learning loop to a deep trainable system.
</div>

---

# The loop remains the same

The PyTorch loop does not change:

```python
optimizer.zero_grad()
y_hat = model(x)
loss = criterion(y_hat, y)
loss.backward()
optimizer.step()
```

What changes is the behavior of the model as depth increases.

---

# A deep network is a composition

For layer $l$:

$$
z^{(l)}=W^{(l)}h^{(l-1)}+b^{(l)}
$$

$$
h^{(l)}=\phi(z^{(l)})
$$

Therefore:

$$
f_\theta(x)
=
f^{(L)}\circ f^{(L-1)}\circ \cdots \circ f^{(1)}(x)
$$

---

# Depth changes the training problem

Each layer transforms:

- the representation;
- the scale of activations;
- the direction and magnitude of gradients;
- the geometry seen by the optimizer.

The issue is not only **expressiveness**.  
The issue is also **trainability**.

---

# Signal propagation

A useful deep network should keep information in a usable numerical range.

$$
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
\cdots
\rightarrow
h^{(L)}
$$

<div class="placeholder">
<strong>S02-F02</strong><br>
Activation distributions propagating through many layers.
</div>

---

# Prediction before experiment

Suppose we build a 50-layer MLP with random weights.

What do you expect to happen to:

$$
\operatorname{mean}(h^{(l)}),
\qquad
\operatorname{std}(h^{(l)})
$$

as $l$ increases?

---

# Activations may vanish or explode

If layer transformations systematically shrink the signal:

$$
\operatorname{std}(h^{(l)}) \rightarrow 0
$$

If they systematically amplify it:

$$
\operatorname{std}(h^{(l)}) \rightarrow \infty
$$

Both cases make optimization difficult.

<div class="placeholder">
<strong>S02-F03</strong><br>
Vanishing, stable and exploding activation scales across depth.
</div>

---

# Gradients also propagate through depth

Backpropagation traverses the graph in reverse.

For many layers, gradients are repeatedly transformed:

$$
\frac{\partial L}{\partial h^{(l)}}
=
\frac{\partial h^{(l+1)}}{\partial h^{(l)}}
\frac{\partial L}{\partial h^{(l+1)}}
$$

The chain rule becomes a long product of local effects.

---

# Vanishing and exploding gradients

The optimizer updates parameters using gradients.

If gradients vanish:

$$
\left\lVert
\frac{\partial L}{\partial W^{(l)}}
\right\rVert
\approx 0
$$

early layers barely learn.

If gradients explode, updates become unstable.

<div class="placeholder">
<strong>S02-F04</strong><br>
Gradient norm versus layer depth for vanishing, stable and exploding regimes.
</div>

---

# Activation functions shape gradient flow

Saturating activations can compress gradients.

ReLU-like activations can preserve stronger gradient paths, but they also introduce zero-gradient regions.

<div class="placeholder">
<strong>S02-F05</strong><br>
Sigmoid, tanh and ReLU: functions and derivative regions.
</div>

---

# Initialization is not a detail

Training starts before the first update.

The initial distribution of weights determines the first forward pass and the first backward pass.

$$
W^{(l)}_{ij}
\sim
\text{some distribution}
$$

The question is: **which distribution keeps signals stable?**

---

# Why not initialize everything equally?

If two hidden units start with the same parameters, they compute the same value.

They also receive the same gradient.

So they remain identical.

<div class="placeholder">
<strong>S02-F06</strong><br>
Symmetry problem when hidden units share identical initialization.
</div>

---

# Preserve variance across layers

A useful initialization should avoid systematic shrinkage or growth:

$$
\operatorname{Var}(h^{(l)})
\approx
\operatorname{Var}(h^{(l-1)})
$$

and ideally also:

$$
\operatorname{Var}(\nabla h^{(l)})
\approx
\operatorname{Var}(\nabla h^{(l+1)})
$$

---

# Xavier / Glorot initialization

For activations such as $\tanh$, Xavier initialization chooses a scale based on layer fan-in and fan-out:

$$
W_{ij}
\sim
\mathcal U
\left(
-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}},
\sqrt{\frac{6}{n_\text{in}+n_\text{out}}}
\right)
$$

The goal is to preserve signal scale.

---

# He / Kaiming initialization

For ReLU-like activations, approximately half of the units may be inactive.

A common choice is:

$$
W_{ij}
\sim
\mathcal N
\left(
0,
\frac{2}{n_\text{in}}
\right)
$$

Initialization and activation must be considered together.

<div class="placeholder">
<strong>S02-F07</strong><br>
Naive, Xavier and He initialization compared through activation variance.
</div>

---

# Practice A — Inspect a deep MLP

The goal is not accuracy yet.

The goal is to measure:

$$
\mu(h^{(l)}),
\qquad
\sigma(h^{(l)}),
\qquad
\left\lVert
\nabla W^{(l)}
\right\rVert
$$

for different depths, activations and initializations.

> Can we detect a bad training setup before many epochs?

---

# Optimization dynamics

A stable initialization helps.

But training still depends on how the optimizer moves through parameter space.

S01 introduced the learning loop.  
Now we compare update dynamics.

---

# SGD is simple but noisy

Mini-batch gradients estimate the dataset gradient.

$$
J_{\mathcal B_t}(\theta)
=
\frac{1}{|\mathcal B_t|}
\sum_{i\in\mathcal B_t}L_i(\theta)

$

$
g_t
=
\nabla_\theta J_{\mathcal B_t}(\theta_t)
$$

$$
\theta_{t+1}
=
\theta_t-\eta g_t
$$

Noise can help exploration, but it can also slow progress.

---

# Momentum accumulates direction

Momentum keeps a velocity vector:

$$
v_t
=
\beta v_{t-1}
+
g_t
$$

$$
\theta_{t+1}
=
\theta_t-\eta v_t
$$

It can reduce oscillation and accelerate movement along persistent descent directions.

<div class="placeholder">
<strong>S02-F08</strong><br>
SGD and Momentum trajectories in a narrow valley.
</div>

---

# Adam adapts the update

Adam estimates both first and second moments:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Each parameter receives an adaptive step.

<div class="center">
<img src="../figures/s02_f09_optimizer_trajectories.svg" style="width:92%;max-height:330px;object-fit:contain;" alt="Optimizer trajectories on an ill-conditioned objective">
</div>

---

# Learning rate schedules

A fixed learning rate may be too large late in training or too small early in training.

Schedulers change $\eta$ across time:

$$
\eta_t = s(t)\eta_0
$$

Common patterns:

- step decay;
- exponential decay;
- cosine annealing.

<div class="center">
<img src="../figures/s02_f10_learning_rate_schedules.svg" style="width:92%;max-height:330px;object-fit:contain;" alt="Learning-rate schedules">
</div>

---

# Normalization as a training tool

Normalization controls scale.

We already standardize inputs:

$$
x'
=
\frac{x-\mu}{\sigma}
$$

Batch Normalization extends this idea to internal activations.

---

# Batch Normalization

For activations in a mini-batch:

$$
\hat z
=
\frac{z-\mu_\mathcal B}
{\sqrt{\sigma_\mathcal B^2+\epsilon}}
$$

Then the layer learns scale and shift:

$$
y=\gamma \hat z+\beta
$$

<div class="placeholder">
<strong>S02-F11</strong><br>
BatchNorm: normalize, then learn scale and shift.
</div>

---

# BatchNorm: training versus inference

During training:

$$
\mu_\mathcal B,\sigma_\mathcal B^2
$$

come from the current mini-batch.

During inference, the model uses running estimates accumulated during training.

<div class="placeholder">
<strong>S02-F12</strong><br>
BatchNorm behavior in training mode versus evaluation mode.
</div>

---

# Regularization

Deep networks can fit complex functions.

Training loss alone is not enough.

We care about performance on unseen data:

$$
L_\text{train}
\qquad
L_\text{val}
$$

<div class="placeholder">
<strong>S02-F13</strong><br>
Underfitting, healthy fitting and overfitting in train/validation curves.
</div>

---

# Weight decay

Weight decay discourages large weights:

$$
L_\text{total}
=
L_\text{data}
+
\lambda
\lVert \theta \rVert_2^2
$$

It changes the preference among solutions, not the architecture.

<div class="placeholder">
<strong>S02-F14</strong><br>
Effect of weight decay on learned functions or weight norms.
</div>

---

# Dropout

Dropout randomly masks activations during training:

$$
\tilde h = m \odot h,
\qquad
m_i \sim \operatorname{Bernoulli}(p)
$$

The network cannot rely on a single fixed path through hidden units.

<div class="placeholder">
<strong>S02-F15</strong><br>
Dropout as stochastic subnetworks during training.
</div>

---

# Deeper is not automatically better

A deeper model can represent at least as much as a shallow model in principle.

But optimization may become harder.

The problem is not only capacity.  
It is also how gradients and representations move through depth.

<div class="placeholder">
<strong>S02-F16</strong><br>
Depth degradation: deeper plain networks can be harder to optimize.
</div>

---

# Residual learning

Instead of learning the whole transformation:

$$
H(x)
$$

learn a residual:

$$
F(x)=H(x)-x
$$

so that:

$$
H(x)=F(x)+x
$$

<div class="placeholder">
<strong>S02-F17</strong><br>
Residual block: transformation path plus identity path.
</div>

---

# A direct path for gradients

If:

$$
H(x)=F(x)+x
$$

then:

$$
\frac{\partial H}{\partial x}
=
\frac{\partial F}{\partial x}
+
1
$$

The identity path gives gradients a direct route through the block.

<div class="placeholder">
<strong>S02-F18</strong><br>
Gradient flow through the identity path in a residual block.
</div>

---

# Diagnostics

When training fails, ask what failed.

| Symptom | What to inspect |
|---|---|
| loss diverges | learning rate, gradient norms |
| loss does not move | initialization, dead activations, gradients |
| train improves but validation worsens | regularization, data split |
| unstable curves | batch size, optimizer, normalization |

---

# Training diagnostics map

A training run should produce more than accuracy.

Track:

$$
L_\text{train},
\quad
L_\text{val},
\quad
\|\nabla W^{(l)}\|,
\quad
\mu(h^{(l)}),
\quad
\sigma(h^{(l)})
$$

<div class="placeholder">
<strong>S02-F19</strong><br>
Map from symptoms to measurements and training interventions.
</div>

---

# Practice B — Build a robust training recipe

Start from a deep MLP baseline.

Add one decision at a time:

1. He initialization
2. Adam or Momentum
3. BatchNorm
4. weight decay or Dropout
5. learning-rate schedule

Explain the curves after each change.

---

# Why deep training works

Deep networks train because several mechanisms cooperate:

<div class="center">

$$
\boxed{
\text{initialization}
+
\text{activations}
+
\text{optimizer}
+
\text{normalization}
+
\text{regularization}
+
\text{residual paths}
}
$$

</div>

<div class="placeholder">
<strong>S02-F20</strong><br>
Integrated view of mechanisms that make deep training possible.
</div>

---

# Exit question

A 40-layer network has almost constant training loss.

What would you inspect first?

Choose two and justify:

- activation statistics;
- gradient norms;
- learning rate;
- initialization;
- BatchNorm mode;
- train/validation gap.
