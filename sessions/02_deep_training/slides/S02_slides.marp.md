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

<!-- Pending figure: S02-F01 — From the single learning loop to a deep trainable system. -->

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

For layer $l=1,\ldots,D$, with $a_0=x$:

$$
z_l=W_la_{l-1}+b_l,
\qquad
a_l=\phi(z_l)
$$

Therefore, for a network of depth $D$:

$$
f_\theta(x)
=
f_D\circ f_{D-1}\circ \cdots \circ f_1(x)
$$

Same notation as S01: $\phi$ is a generic activation, $L$ is the loss.

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
a_1
\rightarrow
a_2
\rightarrow
\cdots
\rightarrow
a_D
$$

<!-- Pending figure: S02-F02 — Activation distributions propagating through many layers. -->

---

# Prediction before experiment

Suppose we build a 50-layer MLP with random weights.

What do you expect to happen to:

$$
\operatorname{mean}(a_l),
\qquad
\operatorname{std}(a_l)
$$

as $l$ increases?

---

# Activations may vanish or explode

If layer transformations systematically shrink the signal:

$$
\operatorname{std}(a_l) \rightarrow 0
$$

If they systematically amplify it:

$$
\operatorname{std}(a_l) \rightarrow \infty
$$

Both cases make optimization difficult.

<!-- Pending figure: S02-F03 — Vanishing, stable and exploding activation scales across depth. -->

---

# Gradients also propagate through depth

Backpropagation traverses the graph in reverse. With $\delta_{a_l}=\partial L/\partial a_l$ as in S01:

$$
\delta_{a_l}
=
\left(\frac{\partial a_{l+1}}{\partial a_l}\right)^{\!\top}
\delta_{a_{l+1}}
=
W_{l+1}^{\top}\left(\delta_{a_{l+1}}\odot\phi'(z_{l+1})\right)
$$

Across $D$ layers, the chain rule becomes a long **product** of local factors.

---

# Vanishing and exploding gradients

The optimizer updates parameters using gradients.

If gradients vanish:

$$
\left\lVert
\nabla_{W_l}J
\right\rVert
\approx 0
$$

early layers barely learn.

If gradients explode, updates become unstable.

<!-- Pending figure: S02-F04 — Gradient norm versus layer depth for vanishing, stable and exploding regimes. -->

---

# Activation functions shape gradient flow

Saturating activations can compress gradients:

$$
\sigma'(z)\le 0.25,
\qquad
\tanh'(z)\le 1
$$

ReLU-like activations can preserve stronger gradient paths, but they also introduce zero-gradient regions.

<!-- Pending figure: S02-F05 — Sigmoid, tanh and ReLU: functions and derivative regions. -->

---

# Initialization is not a detail

Training starts before the first update.

The initial distribution of weights determines the first forward pass and the first backward pass.

$$
(W_l)_{ij}
\sim
\text{some distribution}
$$

The question is: **which distribution keeps signals stable?**

---

# Why not initialize everything equally?

If two hidden units start with the same parameters, they compute the same value.

They also receive the same gradient.

So they remain identical.

<!-- Pending figure: S02-F06 — Symmetry problem when hidden units share identical initialization. -->

---

# Preserve variance across layers

A useful initialization should avoid systematic shrinkage or growth in the forward pass:

$$
\operatorname{Var}(a_l)
\approx
\operatorname{Var}(a_{l-1})
$$

and ideally also in the backward pass:

$$
\operatorname{Var}(\delta_{a_l})
\approx
\operatorname{Var}(\delta_{a_{l+1}})
$$

---

# Xavier / Glorot initialization

For activations such as $\tanh$, Xavier initialization chooses a scale based on layer fan-in and fan-out:

$$
(W_l)_{ij}
\sim
\mathcal U
\left(
-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}},
\sqrt{\frac{6}{n_\text{in}+n_\text{out}}}
\right)
$$

The goal is to preserve signal scale in both directions.

---

# He / Kaiming initialization

For ReLU-like activations, approximately half of the units may be inactive.

A common choice is:

$$
(W_l)_{ij}
\sim
\mathcal N
\left(
0,
\frac{2}{n_\text{in}}
\right)
$$

Initialization and activation must be considered together.

<!-- Pending figure: S02-F07 — Naive, Xavier and He initialization compared through activation variance. -->

---

# Practice A — Inspect a deep MLP

The goal is not accuracy yet.

The goal is to measure, layer by layer:

$$
\operatorname{mean}(a_l),
\qquad
\operatorname{std}(a_l),
\qquad
\left\lVert
\nabla_{W_l}J
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

Mini-batch gradients estimate the dataset gradient:

$$
J_{\mathcal B_t}(\theta)
=
\frac{1}{|\mathcal B_t|}
\sum_{i\in\mathcal B_t}L_i(\theta),
\qquad
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
g_t,
\qquad
\theta_{t+1}
=
\theta_t-\eta v_t
$$

It can reduce oscillation and accelerate movement along persistent descent directions.

<!-- Pending figure: S02-F08 — SGD and Momentum trajectories in a narrow valley. -->

---

# RMSProp rescales each parameter

RMSProp keeps a running average of squared gradients:

$$
s_t=\rho\,s_{t-1}+(1-\rho)\,g_t^2
$$

$$
\theta_{t+1}=\theta_t-\eta\,\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

Directions with consistently large gradients take smaller steps; flat directions take larger ones.

<span class="small">Operations on $g_t$ and $s_t$ are elementwise.</span>

---

# Adam combines both ideas

First moment (momentum) and second moment (RMSProp):

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
s_t=\beta_2s_{t-1}+(1-\beta_2)g_t^2
$$

Bias correction and update:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\quad
\hat s_t=\frac{s_t}{1-\beta_2^t},
\quad
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}
$$

<span class="small">PyTorch defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.</span>

---

# Comparing optimizers

<div class="center">
<img src="../figures/s02_f09_optimizer_trajectories.svg" style="width:96%;max-height:500px;object-fit:contain;" alt="Optimizer trajectories on an ill-conditioned objective">
</div>

---

# Learning rate schedules

A fixed learning rate may be too large late in training or too small early in training.

Schedulers change $\eta$ across time:

$$
\eta_t = s(t)\,\eta_0
$$

Common patterns: step decay, exponential decay, cosine annealing.

<div class="center">
<img src="../figures/s02_f10_learning_rate_schedules.svg" style="width:92%;max-height:330px;object-fit:contain;" alt="Learning-rate schedules">
</div>

---

# Gradient clipping

When a single batch produces a huge gradient, one update can undo many good ones.

Clipping by norm rescales the gradient before the step:

$$
g_t \leftarrow g_t\cdot\min\!\left(1,\ \frac{c}{\lVert g_t\rVert}\right)
$$

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

It treats the **symptom** of exploding gradients; initialization and normalization address the cause.

---

# Normalization as a training tool

Normalization controls scale.

We already standardize inputs:

$$
x'
=
\frac{x-\operatorname{mean}(x)}{\operatorname{std}(x)}
$$

Batch Normalization extends this idea to internal activations.

---

# Batch Normalization

For pre-activations in a mini-batch $\mathcal B$:

$$
\hat z
=
\frac{z-\mu_\mathcal B}
{\sqrt{s_\mathcal B^2+\epsilon}}
$$

Then the layer learns a scale $\gamma$ and a shift $\beta$:

$$
\operatorname{BN}(z)=\gamma \odot \hat z+\beta
$$

<span class="small">$\mu_\mathcal B$, $s_\mathcal B^2$: batch mean and variance. $\gamma$, $\beta$: BN parameters, not the optimizer's $\beta$.</span>

<!-- Pending figure: S02-F11 — BatchNorm: normalize, then learn scale and shift. -->

---

# BatchNorm: training versus inference

During training:

$$
\mu_\mathcal B,\ s_\mathcal B^2
$$

come from the current mini-batch.

During inference, the model uses running estimates accumulated during training.

```python
model.train()   # batch statistics
model.eval()    # running statistics
```

<!-- Pending figure: S02-F12 — BatchNorm behavior in training mode versus evaluation mode. -->

---

# Regularization

Deep networks can fit complex functions.

Training loss alone is not enough.

We care about performance on unseen data:

$$
J_\text{train}(\theta)
\qquad
J_\text{val}(\theta)
$$

<!-- Pending figure: S02-F13 — Underfitting, healthy fitting and overfitting in train/validation curves. -->

---

# Weight decay

Weight decay discourages large weights:

$$
J_\lambda(\theta)
=
J(\theta)
+
\lambda
\lVert \theta \rVert_2^2
$$

It changes the preference among solutions, not the architecture.

<!-- Pending figure: S02-F14 — Effect of weight decay on learned functions or weight norms. -->

---

# Weight decay with Adam: AdamW

With SGD, adding $\lambda\lVert\theta\rVert_2^2$ to the loss and shrinking the weights at each step are equivalent (up to rescaling $\lambda$).

With Adam they are **not**: the penalty gradient is also rescaled by $\sqrt{\hat s_t}$.

AdamW decouples the decay from the adaptive step:

$$
\theta_{t+1}
=
\theta_t
-\eta\left(\frac{\hat m_t}{\sqrt{\hat s_t}+\epsilon}+\lambda\,\theta_t\right)
$$

```python
torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

---

# Dropout

During training, each unit is dropped with probability $p$:

$$
r_i \sim \operatorname{Bernoulli}(1-p),
\qquad
\tilde a = \frac{r \odot a}{1-p}
$$

Dividing by $1-p$ keeps the expected activation unchanged, so at inference Dropout does nothing.

The network cannot rely on a single fixed path through hidden units.

<span class="small">`nn.Dropout(p)`: $p$ is the probability of **dropping** a unit.</span>

<!-- Pending figure: S02-F15 — Dropout as stochastic subnetworks during training. -->

---

# Early stopping

Keep the parameters with the best validation objective:

$$
\theta^\star=\theta_{t^\star},
\qquad
t^\star=\arg\min_t J_\text{val}(\theta_t)
$$

and stop when $J_\text{val}$ has not improved for a fixed number of epochs (patience).

The number of training steps acts as a regularizer.

<span class="small">The S01 exercise already saved the best validation state this way.</span>

---

# Deeper is not automatically better

A deeper model can represent at least as much as a shallow model in principle.

But optimization may become harder.

The problem is not only capacity.  
It is also how gradients and representations move through depth.

<!-- Pending figure: S02-F16 — Depth degradation: deeper plain networks can be harder to optimize. -->

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

<!-- Pending figure: S02-F17 — Residual block: transformation path plus identity path. -->

---

# A direct path for gradients

If:

$$
H(x)=F(x)+x
$$

then the Jacobian is:

$$
\frac{\partial H}{\partial x}
=
\frac{\partial F}{\partial x}
+
I
$$

The identity path gives gradients a direct route through the block, even when $\partial F/\partial x$ is small.

<!-- Pending figure: S02-F18 — Gradient flow through the identity path in a residual block. -->

---

# Diagnostics

When training fails, ask what failed.

| Symptom | What to inspect |
|---|---|
| loss diverges | learning rate, gradient norms, clipping |
| loss does not move | initialization, dead activations, gradients |
| train improves but validation worsens | regularization, early stopping, data split |
| unstable curves | batch size, optimizer, normalization |

---

# Training diagnostics map

A training run should produce more than accuracy.

Track:

$$
J_\text{train},
\quad
J_\text{val},
\quad
\lVert\nabla_{W_l}J\rVert,
\quad
\operatorname{mean}(a_l),
\quad
\operatorname{std}(a_l)
$$

<!-- Pending figure: S02-F19 — Map from symptoms to measurements and training interventions. -->

---

# Practice B — Build a robust training recipe

Start from a deep MLP baseline.

Add one decision at a time:

1. He initialization
2. Momentum or Adam
3. BatchNorm
4. weight decay (AdamW) or Dropout
5. learning-rate schedule and early stopping

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

<!-- Pending figure: S02-F20 — Integrated view of mechanisms that make deep training possible. -->

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
