# SI7011 — Session 06

## Figure and animation inventory

Session 06 focuses on the question:

> **What if we keep the attention and drop the recurrence?**

Slide numbers are those of the current deck source, [`../slides/S06_slides.marp.md`](../slides/S06_slides.marp.md), which matches the 82-slide numbering of [`S06_plan.md`](../../../course/plans/S06_plan.md). Pending figures and animations are shown in the deck as dashed placeholder boxes with their ID and planned filename. To integrate a figure, replace the placeholder slide with:

```markdown
<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain <alt text>](../figures/s06_fnn_<name>.png)
```

Animations use the `media` class with a poster image and a print poster, as in Sessions 02–05.

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017, the illustrated-Transformer blog figures, the LoRA figure of Hu et al. 2021, the diagrams of the previous course's `Clase06b - Transformers.pdf`, or the figures of MIT 6.S191 Lab 3).

## Production routes

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md).
- **Script** — figures that show real data or a trained model: a `make_s06_*.py` script in this folder (matplotlib, fixed seeds), reading the JSON or `.npz` that the notebook saves, so the slide and the notebook always agree.
- **Manim** — animations in `escenas_s06.py`, with the conventions of `escenas_s02.py`–`escenas_s05.py`.

## Figures

| ID | Slide | Purpose | Filename | Route | Status |
|---|---:|---|---|---|---|
| **S06-F01** | V15 | Multi-head attention: $X$ → $H$ heads with their own $Q, K, V$ → concatenate → $W_O$, with shapes | `s06_f01_multihead.png` | Generator | Pending |
| **S06-F02** | V20 | Attention matrices of one sentence (no mask, padding mask, causal mask) and outputs before and after shuffling the tokens without position | `s06_f02_masks_permutation.png` | Script (notebook 6.1: `atencion_mascaras.npz`) | Pending |
| **S06-F03** | V27 | The pre-norm Transformer block with the residual stream as a straight band; post-norm inset | `s06_f03_block.png` | Generator | Pending |
| **S06-F04** | V32 | Encoder, decoder, encoder-decoder, with the mask each one uses | `s06_f04_three_architectures.png` | Generator | Pending |
| **S06-F05** | V39 | BPE on one Spanish sentence; vocabulary size vs sequence length | `s06_f05_bpe.png` | Script (notebook 6.3: `bpe_vocabularios.json`) | Pending |
| **S06-F06** | V48 | Training loss of the tiny GPT with three samples at their steps | `s06_f06_gpt_samples.png` | Script (notebook 6.3: `muestras_gpt.json`) | Pending |
| **S06-F07** | V52 | Test loss vs compute, log-log, from the published power-law fits; optional points from the companion run | `s06_f07_scaling.png` | Script (`make_s06_f07_scaling.py`; optional `resultados_escalado.json`) | Pending |
| **S06-F08** | V56 | CLIP zero-shot on one EuroSAT image: similarity to the 10 class prompts | `s06_f08_clip_zeroshot.png` | Script (notebook 6.4: `clip_zeroshot.json`) | Pending |
| **S06-F09** | V63 | Four ways to use a pretrained model: frozen, trained and added parameters | `s06_f09_four_ways.png` | Generator | Pending |
| **S06-F10** | V67 | From fine-tuning to LoRA in three steps, with the parameter counts | `s06_f10_lora.png` | Generator | Pending |
| **S06-F11** | V81 | Accuracy vs trainable parameters on IMDB: S05 models and the four strategies of 6.5 | `s06_f11_summary.png` | Script (notebook 6.5: `resultados_adaptacion.json`) | Pending |
| **S06-F12** | V72 | One chat example as tokens colored by role, per-token loss of the base model, masked positions greyed, masked vs unmasked mean | `s06_f12_masked_loss.png` | Script (notebook 6.6: `perdida_por_token.json`) | Pending |
| **S06-F13** | V07 | The same word, two meanings: "banco" in two sentences | `s06_f13_same_word_two_meanings.png` | Generator | Pending |
| **S06-F14** | V08 | Self-attention without weights for one token | `s06_f14_attention_no_weights.png` | Generator | Pending |
| **S06-F15** | V10 | Query, key and value for the same token (same layout as F14) | `s06_f15_query_key_value.png` | Generator | Pending |
| **S06-F16** | V19 | Adding position | `s06_f16_adding_position.png` | Generator | Pending |
| **S06-F17** | V29 | A whole GPT, from token ids to probabilities | `s06_f17_whole_gpt.png` | Generator | Pending |
| **S06-F18** | V42 | How BERT hides words | `s06_f18_bert_masking.png` | Generator | Pending |
| **S06-F19** | V44 | Training in parallel, generating one token at a time | `s06_f19_training_vs_generation.png` | Generator | Pending |

## Animations (Manim)

| ID | Slide | What it shows | Filename | Status |
|---|---:|---|---|---|
| **S06-A01** | V12 | One token's query against every key, a softmax row, the weighted average of values; then all rows as $QK^\top$ and the causal mask | `s06_a01_self_attention.mp4` | Pending |
| **S06-A02** | V28 | Tokens through one pre-norm block: attention mixes columns, the MLP acts on each column, the residual stream stays straight | `s06_a02_block.mp4` | Pending |
| **S06-A03** | V45 | Autoregressive generation with sampling; low and high temperature | `s06_a03_generation.mp4` | Pending |
| **S06-A04** | V70 | The masked next-token loss: shift by one, probability of the true token, $L_t$ bars, mask, mean over the answer; the bars after LoRA | `s06_a04_masked_loss.mp4` | Pending |
