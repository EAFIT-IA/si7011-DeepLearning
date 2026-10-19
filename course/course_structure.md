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
\text{remember}
\rightarrow
\text{pretrain/adapt}
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
\rightarrow L
\rightarrow \nabla_\theta L
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

## S05 — Sequences
### How does a network remember the past?

- sequences as data: windows, temporal splits, leakage, baselines
- RNNs and backpropagation through time
- vanishing and exploding gradients, clipping
- LSTM / GRU
- text as a sequence: tokens, vocabulary, embeddings, padding
- sequence-to-sequence models
- attention as the fix for the seq2seq bottleneck

## S06 — Transformers & Foundation Models
### From attention to pretrained models

- self-attention: heads, causal masks, cost
- the Transformer block
- pretraining: next-token and masked prediction
- scaling
- foundation-model concept
- vision, time-series and tabular foundation models
- using a pretrained model: zero-shot, frozen features, fine-tuning
- PEFT / LoRA (mentioned)

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
