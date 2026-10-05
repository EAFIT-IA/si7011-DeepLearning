# SI7011 — Deep Learning
## Course structure

**Format:** 6 sessions × 4 hours  
**Practical stack:** PyTorch / Google Colab  
**Slides:** Marp  
**Visual reference:** academic, progressive diagrams with one main idea per slide.

The course is organized around six questions rather than a historical catalog of architectures.

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
\text{adapt/evaluate/deploy}
$$

## S01 — Learning
### How does a neural network learn?

Core topics:

- parameterized models
- prediction and loss
- gradient descent
- mini-batch optimization
- nonlinearities and MLPs
- computational graphs
- backpropagation
- automatic differentiation

Core mental model:

$$
x \rightarrow f_\theta(x) \rightarrow \hat y
\rightarrow \mathcal L
\rightarrow \nabla_\theta \mathcal L
\rightarrow \theta'
$$

## S02 — Deep Training
### Why can we train deep neural networks?

- initialization
- activation functions
- SGD, momentum and Adam
- regularization
- normalization
- residual connections
- optimization diagnostics
- generalization intuition

## S03 — Architectures for Vision
### From convolution to Vision Transformers

- image tensors
- locality and weight sharing
- convolution
- receptive fields
- CNNs
- residual networks
- transfer learning
- patch embeddings
- self-attention
- Vision Transformers

## S04 — Representation Learning
### How can a network learn useful representations without explicit labels?

- encoders and decoders
- autoencoders
- denoising
- masked prediction
- embeddings
- contrastive learning
- InfoNCE
- self-supervised learning
- linear probing

## S05 — Sequences & Foundation Models
### From recurrent memory to pretrained models

- RNNs
- LSTM / GRU
- attention
- Transformers
- pretraining
- scaling
- foundation-model concept
- vision, tabular and time-series foundation models

## S06 — Adaptation, Evaluation & Deployment
### From pretrained models to reliable applications

- frozen feature extraction
- fine-tuning
- PEFT / LoRA
- distillation
- quantization
- distribution shift
- evaluation and calibration
- latency and memory
- inference pipelines
- deployment and monitoring

## Session format

Each four-hour session follows approximately:

| Time | Block |
|---|---|
| 00:00–00:55 | Theory A |
| 00:55–01:45 | Practice A |
| 01:45–01:55 | Break |
| 01:55–02:50 | Theory B |
| 02:50–03:45 | Practice B |
| 03:45–04:00 | Conceptual closure |

Each session should contain:

- 30–40 conceptual slides;
- two guided practical blocks;
- figures/animations tied directly to concepts;
- explicit prediction-before-experiment questions;
- an exit question.

## Main references

- MIT 6.7960 — Deep Learning
- François Fleuret — Deep Learning Course
- SI7011 original notebooks and course material

The goal is not to reproduce either reference course. MIT primarily informs contemporary scope; Fleuret primarily informs visual and explanatory structure.
