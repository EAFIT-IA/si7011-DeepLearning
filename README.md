# SI7011 — Deep Learning

**Course repository for SI7011 — Deep Learning at Universidad EAFIT**

| **Instructor** | Juan David Martínez Vargas — jdmartinev@eafit.edu.co |
|---|---|

<p align="center">
  <img src="assets/DL.jpg" alt="Deep Learning course overview" width="900">
</p>

<p align="center"><em>Deep Learning overview illustration. Original credits are shown in the image.</em></p>

## Course overview

This course develops a practical and conceptual understanding of **deep learning**, from the mechanics of learning with gradient-based optimization to modern pretrained and foundation models.

The course is organized around six questions:

$$
\text{learn}
\rightarrow
\text{train}
\rightarrow
\text{structure}
\rightarrow
\text{represent}
\rightarrow
\text{pretrain}
\rightarrow
\text{adapt / evaluate / deploy}
$$

## We will learn in this course

- **Learning fundamentals:** parameterized models, loss functions, gradient descent, multilayer perceptrons, backpropagation, and automatic differentiation.
- **Training deep networks:** initialization, activation functions, SGD/Adam, regularization, normalization, residual connections, and training diagnostics.
- **Deep learning for vision:** convolutional neural networks, residual networks, transfer learning, attention, and **Vision Transformers (ViT)**.
- **Representation learning:** encoders and decoders, autoencoders, masked prediction, contrastive learning, embeddings, and self-supervised learning.
- **Sequential and foundation models:** RNNs, LSTM/GRU, attention, Transformers, pretraining, scaling, and foundation models for language, vision, time series, and tabular data.
- **Adaptation and deployment:** feature extraction, fine-tuning, PEFT/LoRA, distillation, quantization, distribution shift, evaluation, inference, and deployment.
- **Hands-on implementation:** using **PyTorch** and **Google Colab** to build, train, inspect, adapt, and evaluate deep learning models.

## Course structure

| Session | Guiding question | Main topics |
|---|---|---|
| **01 — Learning** | How does a neural network learn? | Model, loss, gradient descent, MLP, backprop, autograd |
| **02 — Deep Training** | Why can we train deep neural networks? | Initialization, optimization, normalization, residual learning, generalization |
| **03 — Architectures for Vision** | How does data structure shape the network? | CNN, ResNet, transfer learning, attention, ViT |
| **04 — Representation Learning** | How can a network learn without explicit labels? | Autoencoders, masked modeling, contrastive learning, SSL |
| **05 — Sequences & Foundation Models** | How do models learn context at scale? | RNN, LSTM/GRU, Transformer, pretraining, scaling, foundation models |
| **06 — Adaptation, Evaluation & Deployment** | How do we turn pretrained models into reliable applications? | Fine-tuning, PEFT, OOD, evaluation, inference, quantization, deployment |

Detailed course design: [`course/course_structure.md`](course/course_structure.md)

## Session 01 — Learning

- [Slides (PDF)](sessions/01_learning/slides/S01_slides.pdf) · [Marp source](sessions/01_learning/slides/S01_slides.marp.md)
- [Rendering instructions](sessions/01_learning/slides/README.md)
- [Figure and animation inventory](sessions/01_learning/figures/figures.md)
- [Guided practice and exercise](sessions/01_learning/notebooks/README.md)

The central mental model is:

$$
x
\rightarrow
f_\theta(x)
\rightarrow
\hat y
\rightarrow
L
\rightarrow
\nabla_\theta L
\rightarrow
\theta'
$$


## Session 02 — Deep Training

- [Slides (PDF)](sessions/02_deep_training/slides/S02_slides.pdf) · [Marp source](sessions/02_deep_training/slides/S02_slides.marp.md)
- [Figure and animation inventory](sessions/02_deep_training/figures/figures.md)
- [Practice A and Practice B notebooks](sessions/02_deep_training/notebooks/README.md)

Why can deep networks be trained? Signal propagation, initialization, optimizers, normalization and regularization, residual learning and training diagnostics. Both practices follow the data → training → deployment pipeline, with runs tracked in MLflow.

## Evaluation

- [Evaluation strategy](course/evaluation_strategy.md) · [Evaluation Event 1 (S01–S02)](course/evaluation_event_1.md)

## Evaluation

The course uses **three evaluation events aligned with pairs of sessions** and one transversal integrative project.

| Assessment | Sessions | Weight |
|---|---:|---:|
| **Evaluation Event 1 — Learning & Training** | S01–S02 | **20%** |
| **Evaluation Event 2 — Architecture & Representation** | S03–S04 | **20%** |
| **Evaluation Event 3 — Foundation Models & Adaptation** | S05–S06 | **25%** |
| **Integrative Project** | transversal | **35%** |

Each evaluation event is designed **after its corresponding pair of sessions has been completed**, so that the assessment reflects the concepts, experiments and level of depth actually developed in class.

The events emphasize **conceptual and experimental reasoning**, not simply implementation of a particular architecture.

Detailed strategy: [`course/evaluation_strategy.md`](course/evaluation_strategy.md)

## Repository structure

```text
SI7011-DeepLearning/
├── README.md
├── assets/
│   └── DL.jpg
├── course/
│   └── course_structure.md
├── sessions/
│   ├── 01_learning/
│   │   ├── slides/
│   │   ├── figures/
│   │   └── notebooks/
│   ├── 02_deep_training/
│   ├── 03_vision/
│   ├── 04_representation_learning/
│   ├── 05_foundation_models/
│   └── 06_adaptation_deployment/
├── assignments/
└── references/
```

Slides are primarily authored in **Marp**. Practical material is developed in **PyTorch/Colab**, and figures are treated as reusable teaching assets rather than decoration.

## Computational resources

Free accounts are recommended on:

- [Google Colab](https://colab.research.google.com/)
- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/)
- [Lightning AI](https://lightning.ai/)
- [Weights & Biases](https://wandb.ai/site)

## Books

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/)
- [Dive into Deep Learning](https://d2l.ai/)
- [Deep Learning: Foundations and Concepts — Bishop & Bishop](https://www.bishopbook.com/)
- [Deep Learning with Python — François Chollet](https://github.com/fchollet/deep-learning-with-python-notebooks)
- [The Little Book of Deep Learning — François Fleuret](https://fleuret.org/francois/lbdl.html)

## Reference courses

- [MIT 6.7960 — Deep Learning](https://deeplearning6-7960.github.io/)
- [MIT Introduction to Deep Learning](https://introtodeeplearning.com/)
- [François Fleuret — Deep Learning Course](https://fleuret.org/dlc/)
- [DeepMind Deep Learning Course](https://www.youtube.com/watch?v=7R52wiUgxZI)

## Design philosophy

The course does not treat deep learning as a catalog of architectures. Each session starts from a **problem or question**, develops the mathematical and computational ideas needed to address it, and then validates those ideas experimentally.

For the visual material, the guiding principle is:

> **one main conceptual transformation per slide**

Figures, animations, equations, and code are introduced only when they make that transformation easier to understand.
