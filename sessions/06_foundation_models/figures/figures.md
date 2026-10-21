# SI7011 — Session 06

## Figure and animation inventory

Session 06 focuses on the question:

> **What if we keep the attention and drop the recurrence?**

Slide numbers are those of the current deck source, [`../slides/S06_slides.marp.md`](../slides/S06_slides.marp.md), which matches the 83-slide numbering of [`S06_plan.md`](../../../course/plans/S06_plan.md). Pending figures and animations are shown in the deck as dashed placeholder boxes with their ID and planned filename. To integrate a figure, replace the placeholder slide with:

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
- **Script, drawn** — diagrams that need exact numbers or identical repeated elements, which the generator gets wrong: the token-column series of Part 1 (F01, F02, F03, F05) in `make_s06_f01_f05_tokens.py`. Scores are illustrative; weights, averages and outputs are computed in the script.
- **Manim** — animations in `escenas_s06.py`, with the conventions of `escenas_s02.py`–`escenas_s05.py`.

## Figures

| ID | Slide | Purpose | Filename | Route | Status |
|---|---:|---|---|---|---|
| **S06-F01** | V07 | The same word, two meanings: "banco" in two sentences | `s06_f01_same_word_two_meanings.png` | Script `make_s06_f01_f05_tokens.py` (drawn, illustrative scores) | Included |
| **S06-F02** | V08 | Self-attention without weights for one token | `s06_f02_attention_no_weights.png` | Script `make_s06_f01_f05_tokens.py` (drawn, illustrative scores) | Included |
| **S06-F03** | V10 | Query, key and value for the same token (same layout as F02) | `s06_f03_query_key_value.png` | Script `make_s06_f01_f05_tokens.py` (drawn, illustrative scores) | Included |
| **S06-F04** | V15 | Multi-head attention: $X$ → $H$ heads with their own $Q, K, V$ → concatenate → $W_O$, with shapes | `s06_f04_multihead.png` | Generator + script: `fix_s06_f04_multihead.py` (adds the $\sqrt{d_k}$; concatenation as vertical strips) | Included |
| **S06-F05** | V19 | Adding position | `s06_f05_adding_position.png` | Script `make_s06_f01_f05_tokens.py` (drawn, illustrative scores) | Included |
| **S06-F06** | V20 | Attention matrices of one sentence (no mask, padding mask, causal mask) and outputs before and after shuffling the tokens without position | `s06_f06_masks_permutation.png` | Script `make_s06_figuras_notebooks.py` (notebook 6.1, sections 5-6: `atencion_mascaras.npz`) | Included |
| **S06-F07** | V27 | The pre-norm Transformer block with the residual stream as a straight band; post-norm inset | `s06_f07_block.png` | Generator + script: `fix_s06_f07_block.py` (formulas, $X$, post-norm inset redrawn) | Included |
| **S06-F08** | V29 | A whole GPT, from token ids to probabilities | `s06_f08_whole_gpt.png` | Generator + script: `fix_s06_f08_whole_gpt.py` (all ids into the token embedding) | Included |
| **S06-F09** | V32 | Encoder, decoder, encoder-decoder, with the mask each one uses | `s06_f09_three_architectures.png` | Generator | Included |
| **S06-F10** | V39 | BPE on one Spanish sentence by hand (0, 10, 30, 60 merges); byte-level BPE at 256 / 1 000 / 4 000 tokens, characters per token | `s06_f10_bpe.png` | Script `make_s06_f10_bpe.py` (corpus: the Spanish text of the course notebooks, no download) | Included |
| **S06-F11** | V42 | How BERT hides words | `s06_f11_bert_masking.png` | Generator + script: `fix_s06_f11_bert_masking.py` (panel 4 redrawn bottom to top) | Included |
| **S06-F12** | V44 | Training in parallel, generating one token at a time | `s06_f12_training_vs_generation.png` | Generator | Included |
| **S06-F13** | V48 | A base model, asked a question: Qwen2.5-0.5B base on three prompts, stopped or cut at the limit | `s06_f13_base_model_answers.png` | Script `make_s06_figuras_notebooks.py` (notebook 6.3: `resultados_sft.json`) | Pending (run 6.3 on a T4) |
| **S06-F14** | V52 | Test loss vs compute, log-log, from the published power-law fits; optional points from the companion run | `s06_f14_scaling.png` | Script `make_s06_f14_scaling.py` (published fit, re-estimated by Besiroglu et al. 2024) | Included |
| **S06-F15** | V56 | CLIP zero-shot on one EuroSAT image: similarity to the 10 class prompts | `s06_f15_clip_zeroshot.png` | Script `make_s06_figuras_notebooks.py` (notebook 6.4, optional: `clip_zeroshot.npz`) | Pending (run 6.4 once) |
| **S06-F16** | V63 | Four ways to use a pretrained model: frozen, trained and added parameters | `s06_f16_four_ways.png` | Generator | Included |
| **S06-F17** | V67 | From fine-tuning to LoRA in three steps, with the parameter counts | `s06_f17_lora.png` | Generator | Included |
| **S06-F18** | V72 | One Dolly pair as tokens colored by role, per-token loss of Qwen2.5-0.5B base, masked positions faded, masked vs unmasked mean | `s06_f18_masked_loss.png` | Script `make_s06_figuras_notebooks.py` (notebook 6.3: `resultados_sft.json`) | Pending (run 6.3 on a T4) |
| **S06-F19** | V82 | Test accuracy vs trainable parameters on IMDB (2 000 reviews): LSTM from scratch and the three DistilBERT strategies of 6.5 | `s06_f19_summary.png` | Script (notebook 6.5: `resultados_adaptacion.json`) | Pending (run 6.5 on a T4) |

## Animations (Manim)

| ID | Slide | What it shows | Filename | Status |
|---|---:|---|---|---|
| **S06-A01** | V12 | One token's query against every key, a softmax row, the weighted average of values; then all rows as $QK^\top$ and the causal mask | `s06_a01_self_attention.mp4` | Included |
| **S06-A02** | V28 | Tokens through one pre-norm block: attention mixes columns, the MLP acts on each column, the residual stream stays straight | `s06_a02_block.mp4` | Included |
| **S06-A03** | V45 | Autoregressive generation with sampling; low and high temperature | `s06_a03_generation.mp4` | Included |
| **S06-A04** | V70 | The masked next-token loss: shift by one, probability of the true token, $L_t$ bars, mask, mean over the answer; the bars after LoRA | `s06_a04_masked_loss.mp4` | Included |
