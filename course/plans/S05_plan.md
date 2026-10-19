# SI7011 — Session 05 · Sequences & Foundation Models

## Slide-by-slide plan (draft)

> **How do models learn context at scale?**

Instructor planning document, same format as `S04_plan.md`. It fixes the deck structure, the content of each slide, the figures and animations, and the notebooks before anything is built. Conventions of Sessions 02–04: slides in English, notebooks in Spanish, five parts opened by divider slides that recap the previous part, one prediction-before-experiment slide per part, two practice blocks and an exit question. Folder: `sessions/05_foundation_models/` (the name already in the root README tree).

**Status: draft.** The decisions below are proposals, not yet adopted.

## Sources

| Source | What it gives this session |
|---|---|
| `course/course_structure.md` (EAFIT-IA) | Topic list for S05: RNNs, LSTM/GRU, attention, Transformers, pretraining, scaling, the foundation-model concept, vision, tabular and time-series foundation models |
| `course/evaluation_strategy.md` (EAFIT-IA) | Event 3 covers S05–S06. Students must "distinguish a Transformer architecture from a foundation model". In the project, S05 asks whether a pretrained or foundation model is relevant to the problem |
| Promises made in S02–S04 | S02: GELU, LayerNorm and the pre-norm block "are the Transformer of S05". S03 V-*Self-attention, in one slide*: "heads, masks, cost come in Session 05"; notebook 3.5 already has `attention(X, W_Q, W_K, W_V)` and the permutation test. S04: BERT masks 15 % (V34), CLIP is a foundation model (V51) |
| jdmartinev/si7011-deeplearning, `Lecture05` and `Lecture06` | Jena Climate RNN/LSTM/GRU notebooks, Bike Sharing exercise, the RNN lecture and the text/Transformer PDFs. Details in [Reusable material](#reusable-material-from-jdmartinevsi7011-deeplearning) |

## Boundary with S06

S06 already covers fine-tuning, PEFT/LoRA, distillation, quantization, evaluation, deployment and the FastAPI service. S05 therefore stops at **pretraining and using a pretrained model as is** (zero-shot or frozen features). Fine-tuning appears in S05 only as "next session".

## Decisions (proposals)

1. **Two running datasets, one per half.** Theory A and Practice A use **Jena Climate** (temperature forecasting, 14 variables every 10 min), which the old repo already prepares. Theory B uses **text**, because next-token prediction is the clearest way to show pretraining and scaling. For the text, a public-domain Spanish corpus (*Don Quijote*, Project Gutenberg) at character level. Jena comes back at the end with a time-series foundation model, so the session ends where it started.
2. **The integrating exercise answers the S05 project question.** *Does a foundation model that never saw your data beat a network trained on it?* Bike Sharing (hourly demand, UCI), the dataset of the old exercise: an LSTM/GRU trained by the students against Chronos-Bolt used zero-shot, on the same windows and the same metric, with MLflow. Four TODOs as in S02–S04.
3. **Scope cuts, so the session fits in four hours.** GRU, bidirectional and stacked RNNs: one slide. Encoder-decoder (T5): one line. Tabular foundation models (TabPFN): one slide plus an optional section in notebook 5.5 (Covertype subset, callback to S02). Scaling laws: concept plus a small measured sweep as homework. No tokenizer training: BPE gets one slide and one demo cell.
4. **Pretraining in class without a long run.** Notebook 5.4 trains a small character-level GPT for about 5 minutes on GPU, enough to see the loss fall and the samples change. A longer checkpoint is optional, following the SimCLR pattern of S04 (Kaggle model).
5. **Which notebook is homework.** Proposal: 5.4 (GPT and scaling) in class and 5.5 (foundation models zero-shot) as homework, because the integrating exercise repeats its main idea. The alternative is the reverse; see the open questions.

## Notation

Consistent with S01–S04 (φ is the activation, σ only the sigmoid, *L* the loss, *J* the objective). S01 and S04 already call a hidden representation **h**; the hidden state keeps that letter.

| Symbol | Meaning |
|---|---|
| $\mathbf x_t$ | input at time step $t$, $t = 1,\dots,T$ |
| $\mathbf h_t$ | hidden state; $\mathbf h_t = \varphi(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b)$ |
| $\mathbf c_t$ | LSTM cell state |
| $\mathbf f_t, \mathbf i_t, \mathbf o_t$ | forget, input and output gates (σ, values in $(0,1)$) |
| $\odot$ | element-wise product |
| $T$ | sequence length (window or context) |
| $X\in\mathbb R^{T\times d}$ | token matrix; row $t$ is token $\mathbf t_t$ (as in S03) |
| $Q, K, V$ | queries, keys, values; $d_k$ key dimension |
| $A$ | attention weights, $T\times T$, rows sum 1 |
| $H$ | number of heads, $d_k = d/H$ |
| $M$ | causal mask: $0$ on and below the diagonal, $-\infty$ above |
| $p_\theta(x_{t+1}\mid x_{\le t})$ | next-token model; $L_t = -\log p_\theta(x_{t+1}\mid x_{\le t})$ |
| $N$, $D$, $C$ | parameters, training tokens, compute (scaling laws) |

Clash to avoid: S04 used $N$ for images per batch. In S05, $N$ is the parameter count only on the scaling slides, and the slide says so.

## Timing

| Time | Block | Slides / notebooks |
|---|---|---|
| 00:00–00:55 | Theory A | Opening, Part 1, Part 2 (V01–V27) |
| 00:55–01:45 | Practice A | V28 · notebooks 5.1–5.2 |
| 01:45–01:55 | Break | |
| 01:55–02:50 | Theory B | Parts 3–5 (V29–V64) |
| 02:50–03:45 | Practice B | V65 · notebooks 5.3–5.4 (5.5 as homework) |
| 03:45–04:00 | Closure | V66–V70 |

## Deck structure

70 slides (S02 has 78, S03 75, S04 66).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Sequences and recurrence | V05–V17 | |
| 2 · Memory that lasts: LSTM and GRU | V18–V28 | Practice A (V28): notebooks 5.1–5.2 |
| 3 · Attention and the Transformer | V29–V45 | |
| 4 · Pretraining and scaling | V46–V56 | |
| 5 · Foundation models | V57–V65 | Practice B (V65): notebooks 5.3–5.4 |
| Closure | V66–V70 | Integrating exercise (V66), exit question (V70) |

Prediction-before-experiment slides: V15, V25, V42, V54, V63.

---

## Opening (V01–V04)

| # | Slide | Content | Visual |
|---|---|---|---|
| V01 | How do models learn context at scale? | Title, session question | |
| V02 | Roadmap | The five parts and the two practice blocks | |
| V03 | From representation to context | S04 recap: an encoder $g$ learned from the data itself. But each input was a fixed-size image. Now the input is a sequence whose length changes and whose order matters | |
| V04 | The question for today | Two questions: how does a network keep the past (Parts 1–3)? And what happens when that network is trained on almost all available data (Parts 4–5)? | |

## Part 1 · Sequences and recurrence (V05–V17)

| # | Slide | Content | Visual |
|---|---|---|---|
| V05 | Part 1 · Sequences and recurrence | Divider. Recap of S03–S04: architecture encodes assumptions; the objective chooses what $g$ learns | |
| V06 | Order matters | Text, sensors, audio, prices, gait. Shuffling the steps destroys the meaning. Callback to S03: shuffling pixels hurt the CNN but not the MLP | |
| V07 | Tasks on sequences | Many-to-one (classify a review, forecast the next value), many-to-many aligned (tagging), sequence-to-sequence (translation), one-to-many (generation) | |
| V08 | *(figure)* | **S05-F01** | F01 |
| V09 | Turning a series into examples | Sliding window: $T$ past steps → value $h$ steps ahead. Temporal split without shuffling across the cut. Normalize with training statistics only. Data-contract callback (S02 integrator) | |
| V10 | *(figure)* | **S05-F02** | F02 |
| V11 | The baseline to beat | Jena: "tomorrow's temperature = today's at the same hour" (commonsense). A model that does not beat it has learned nothing useful. Same role as the majority class in S01 | |
| V12 | Why not an MLP? | Flattening $T\times 14$ inputs: parameters grow with $T$, no sharing across time, and the order is just a position in a vector. A 1-D convolution shares weights but sees a fixed window (S03) | |
| V13 | The recurrent cell | $\mathbf h_t = \varphi(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b)$. **The same weights at every step**: the weight sharing of S03, now along time. $\mathbf h_t$ summarizes $\mathbf x_{1:t}$ | |
| V14 | Unrolling through time | **S05-A01** (media) | A01 |
| V15 | Prediction before experiment | Jena, $T = 120$ (20 h at one reading per 10 min): commonsense baseline, flattened MLP and a 16-unit RNN. Order them by validation MAE. Answer in notebook 5.1 | |
| V16 | Backpropagation through time | The unrolled graph is a deep network with tied weights. $\dfrac{\partial \mathbf h_T}{\partial \mathbf h_t} = \prod_{k=t+1}^{T} \operatorname{diag}(\varphi'(\cdot))\,W_{hh}$. Same product of Jacobians as S02 depth, now with the **same** matrix repeated | |
| V17 | Vanishing and exploding gradients | Spectral radius of $W_{hh}$ below 1: the signal from step $t$ fades; above 1: it explodes. Gradient clipping fixes explosions only. Truncated BPTT in one line | |

## Part 2 · Memory that lasts: LSTM and GRU (V18–V28)

| # | Slide | Content | Visual |
|---|---|---|---|
| V18 | Part 2 · LSTM and GRU | Divider. Recap: the repeated product of Jacobians wipes out the past | |
| V19 | An additive path for memory | $\mathbf c_t = \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\tilde{\mathbf c}_t$. Addition, not repeated multiplication by $W_{hh}$: the residual idea of S02 along time. With $\mathbf f_t\approx 1$ the gradient goes through untouched | |
| V20 | *(figure)* | **S05-F03** | F03 |
| V21 | Three gates | $\mathbf f_t, \mathbf i_t, \mathbf o_t = \sigma(\cdot)$ from $[\mathbf h_{t-1}, \mathbf x_t]$: what to forget, what to write, what to show. $\mathbf h_t = \mathbf o_t\odot\tanh(\mathbf c_t)$. A gate is a learned, input-dependent multiplier | |
| V22 | The memory in motion | **S05-A02** (media) | A02 |
| V23 | GRU, stacked, bidirectional | GRU: two gates, no separate $\mathbf c_t$, fewer parameters, often similar results. Stacking: $\mathbf h_t$ of one layer is $\mathbf x_t$ of the next. Bidirectional: only when the whole sequence is available at prediction time | |
| V24 | *(figure)* | **S05-F04** | F04 |
| V25 | Prediction before experiment | Same Jena setup with an LSTM. Does it beat the baseline? And if the window grows from 20 h to 5 days, does the RNN gain, lose or stay the same? And the LSTM? | |
| V26 | Regularizing recurrent networks | Dropout on the inputs and on the recurrent connection (same mask at every step), weight decay, early stopping. Callback to S02 | |
| V27 | What recurrence still cannot do | Strictly sequential: step $t$ waits for step $t-1$, so training does not parallelize along $T$. The whole past must fit in one vector $\mathbf h_t$. Bridge to Part 3 | |
| V28 | Practice A — Recurrent networks | Notebooks 5.1 and 5.2. What to compare, how long, what to write down | |

## Part 3 · Attention and the Transformer (V29–V45)

| # | Slide | Content | Visual |
|---|---|---|---|
| V29 | Part 3 · Attention and the Transformer | Divider. Recap: recurrence squeezes the past into $\mathbf h_t$ and runs step by step | |
| V30 | Look back instead of remembering | Attention, as it appeared in translation (Bahdanau, 2015): the decoder looks at all the encoder states and weights them. The bottleneck vector disappears | |
| V31 | Self-attention, recalled | The S03 formula. Each token is a weighted average of all tokens; the weights depend on content. Permutation-equivariant (S03 notebook 3.5 test) | |
| V32 | Queries, keys and values | Retrieval analogy: $\mathbf q_t$ asks, $\mathbf k_j$ advertises, $\mathbf v_j$ is what is delivered. Why divide by $\sqrt{d_k}$: dot products grow with $d_k$ and saturate the softmax (S02 signal scale) | |
| V33 | Multi-head attention | $H$ heads, each with $d_k = d/H$, in parallel; concatenate and project with $W_O$. Same cost as one big head; each head can attend to a different relation | |
| V34 | *(figure)* | **S05-F05** | F05 |
| V35 | The causal mask | To predict $x_{t+1}$, token $t$ must not see the future: $A = \operatorname{softmax}\!\big((QK^\top + M)/\sqrt{d_k}\big)$. One forward pass trains all $T$ predictions at once | |
| V36 | Masking in motion | **S05-A03** (media) | A03 |
| V37 | Position | Attention ignores order, so position is added. Learned (S03 ViT), sinusoidal (original Transformer), rotary (RoPE, used in most current LLMs), one line each | |
| V38 | The Transformer block | $\mathbf t \leftarrow \mathbf t + \operatorname{MHA}(\operatorname{LN}(\mathbf t))$; $\mathbf t \leftarrow \mathbf t + \operatorname{MLP}(\operatorname{LN}(\mathbf t))$. Every piece is from S02: residuals, LayerNorm, pre-norm, GELU. Attention mixes across tokens; the MLP processes each token | |
| V39 | *(figure)* | **S05-F06** | F06 |
| V40 | The price: $T^2$ | $A$ is $T\times T$: memory and compute grow quadratically with the context. Recurrence is linear in $T$ but sequential. Measured in notebook 5.3 | |
| V41 | *(figure)* | **S05-F07** | F07 |
| V42 | Prediction before experiment | A Transformer on Jena with no positional encoding and mean pooling. Shuffle the 120 steps of every validation window. What happens to its MAE? And to the LSTM's? | |
| V43 | Encoder, decoder, both | Encoder only, bidirectional attention (BERT): understand. Decoder only, causal (GPT): generate. Encoder-decoder (T5, the original Transformer): transform one sequence into another | |
| V44 | *(figure)* | **S05-F08** | F08 |
| V45 | Recurrence vs attention | Table: path length between two steps ($T$ vs 1), parallel training (no vs yes), cost per layer ($T d^2$ vs $T^2 d$), inference memory (constant vs grows with context). Footnote: state-space models (Mamba) try to have both | |

## Part 4 · Pretraining and scaling (V46–V56)

| # | Slide | Content | Visual |
|---|---|---|---|
| V46 | Part 4 · Pretraining and scaling | Divider. Recap: the Transformer parallelizes over the sequence. So it can be trained on far more data | |
| V47 | Tokens | Text → integer ids → `nn.Embedding` (S04 V08: one-hot × matrix). Characters (small vocabulary, long sequences), words (huge vocabulary, unknown words), subwords with BPE (the compromise). Spanish costs more tokens than English for most tokenizers | |
| V48 | Next-token prediction | $J(\theta) = \frac{1}{T}\sum_t -\log p_\theta(x_{t+1}\mid x_{\le t})$: the same cross-entropy as S01, over the vocabulary. The labels are the text shifted by one: self-supervision (S04) at the scale of the internet | |
| V49 | Generating text | **S05-A04** (media) | A04 |
| V50 | Sampling and temperature | Greedy, temperature $\tau$ (S04 used the same symbol in InfoNCE: same role, it sharpens or flattens a softmax), top-$k$. Perplexity $= e^{J}$ | |
| V51 | Masked language modeling | BERT: mask 15 % of the tokens and predict them (S04 V34: why 15 % here and 75 % for images). Produces an encoder for understanding, not a generator | |
| V52 | What pretraining learns | Grammar, facts and patterns that no one labeled. Representation view (S04): the pretrained model is a $g$ for text; its last-layer states are embeddings | |
| V53 | Scaling laws | Loss falls as a power law in $N$, $D$ and $C$ (Kaplan et al., 2020). Chinchilla (Hoffmann et al., 2022): for a fixed compute budget, grow $N$ and $D$ together, about 20 training tokens per parameter | |
| V54 | Prediction before experiment | Small GPT on *Don Quijote*, same number of training steps, $d$ = 32, 64, 128, 256. Plot validation loss against parameters on log–log axes: straight line, curve, or no trend? (Notebook 5.4, homework section) | |
| V55 | *(figure)* | **S05-F09** | F09 |
| V56 | Emergence, with care | Abilities that seem to appear suddenly with scale; part of the jump depends on the metric chosen (Schaeffer et al., 2023). Measure with a smooth metric before claiming emergence | |

## Part 5 · Foundation models (V57–V65)

| # | Slide | Content | Visual |
|---|---|---|---|
| V57 | Part 5 · Foundation models | Divider. Recap: one pretrained model with broad knowledge, many tasks downstream | |
| V58 | Architecture ≠ foundation model | "Foundation model" (Bommasani et al., 2021) is a role, not an architecture: trained on broad data, adapted to many tasks. A Transformer trained on Jena is not a foundation model; a ResNet trained on ImageNet almost is. Event 3 asks for this distinction | |
| V59 | Ways to use one | Zero-shot (prompt or description only), few-shot in context, frozen features plus probe (S04), fine-tuning and PEFT (S06). Today: the first three | |
| V60 | Vision and language | CLIP (S04 V51): contrastive image–text pretraining; zero-shot classification by comparing an image with the text "a photo of a {class}". DINOv2 for features without text | |
| V61 | Time series | Chronos: scale the series, quantize into bins, treat the bins as tokens, train a T5 on many series, and predict as with text. TimesFM and Moirai: the same idea with patches. Zero-shot forecasting of a series the model never saw | |
| V62 | *(figure)* | **S05-F10** | F10 |
| V63 | Prediction before experiment | Jena, 24 h ahead: Chronos-Bolt zero-shot, never trained on Jena, against the LSTM of notebook 5.2. Which has the lower MAE? Is it the same answer at 1 h and at 24 h? (Notebook 5.5) | |
| V64 | Tabular data and when not to use one | TabPFN in one paragraph (prior-fitted on synthetic tables, predicts a small table in one forward pass). When a foundation model is the wrong choice: data unlike the pretraining data, latency or cost limits, a small model already beats the baseline | |
| V65 | Practice B — Attention and pretraining | Notebooks 5.3 and 5.4 in class; 5.5 at home | |

## Closure (V66–V70)

| # | Slide | Content | Visual |
|---|---|---|---|
| V66 | Integrating exercise — Bike Sharing | See decision 2. Four TODOs, as in S02–S04 | |
| V67 | Choosing a sequence model | Table: RNN / LSTM-GRU / Transformer / pretrained model × what it needs (data, compute), how far it sees, cost at inference | |
| V68 | What each choice assumes | Format of S03 V69 and S04 V64: "assumes" and "pays with". Recurrence assumes the past fits in a vector; attention assumes the whole context fits in memory; a foundation model assumes your data resemble its pretraining data | |
| V69 | *(figure)* | **S05-F11** | F11 |
| V70 | Exit question | A hospital has two years of hourly bed occupancy and wants a 48 h forecast. Choose two and justify: commonsense baseline; LSTM trained on the hospital data; Chronos zero-shot; a Transformer trained from scratch; check first whether the series has the same weekly pattern as the training years | |

---

## Figures

Same routes as S03 and S04: **Generator** (prompts in `figure_prompts.md`), **Script** (`make_s05_*.py`, fixed seeds, numbers taken from the notebooks) and **Manim** (`escenas_s05.py`). All figures drawn from scratch.

| ID | Slide | Purpose | Planned filename | Route |
|---|---:|---|---|---|
| **S05-F01** | V08 | Four task types on sequences (many-to-one, aligned many-to-many, seq-to-seq, one-to-many) | `s05_f01_sequence_tasks.png` | Generator |
| **S05-F02** | V10 | Jena: sliding windows over the series, target 24 h ahead, train/val/test cut with no overlap | `s05_f02_windows_split.png` | Script (notebook 5.1) |
| **S05-F03** | V20 | LSTM cell with $\mathbf c_t$ path highlighted as the additive line, gates as valves | `s05_f03_lstm_cell.png` | Generator |
| **S05-F04** | V24 | Gradient norm $\lVert\partial L/\partial\mathbf x_t\rVert$ against $t$ for RNN, LSTM and GRU on Jena (log scale) | `s05_f04_gradient_through_time.png` | Script (notebooks 5.1–5.2) |
| **S05-F05** | V34 | Multi-head attention: $X$ → $H$ sets of $Q,K,V$ → $H$ matrices $A$ → concatenation → $W_O$ | `s05_f05_multihead.png` | Generator |
| **S05-F06** | V39 | Pre-norm Transformer block with each piece tagged with the session where it first appeared (S02, S03) | `s05_f06_transformer_block.png` | Generator |
| **S05-F07** | V41 | Time and memory of one attention layer and one LSTM layer against $T$ = 64…4096 | `s05_f07_cost_vs_length.png` | Script (notebook 5.3) |
| **S05-F08** | V44 | Encoder-only / decoder-only / encoder-decoder: attention masks side by side, with one model name each | `s05_f08_three_families.png` | Script (masks) + labels |
| **S05-F09** | V55 | Validation loss vs parameters, log–log, from the GPT sweep | `s05_f09_scaling_sweep.png` | Script (notebook 5.4) |
| **S05-F10** | V62 | Chronos tokenization: a series, mean scaling, quantization into bins, sequence of token ids | `s05_f10_chronos_tokens.png` | Script |
| **S05-F11** | V69 | Summary: the path from S01 to S05 (MLP → CNN → encoder → Transformer → foundation model) with what each step added | `s05_f11_summary.png` | Generator |

## Animations (Manim)

Only where motion changes understanding. Conventions of `escenas_s02.py`–`escenas_s04.py`: slide palette, on-screen text in English, `self.pausa()` before each reveal, poster from the last frame.

| ID | Slide | What it shows | Why animate | Rendered file |
|---|---:|---|---|---|
| **S05-A01** | V14 | The RNN cell rolled up, then unrolled over 8 steps with the same $W_{hh}$ highlighted at every step; then the gradient flows back from step 8 and its arrow thins step by step | Weight sharing in time, and the fading, are both temporal | `s05_a01_unrolling_bptt.mp4` |
| **S05-A02** | V22 | One value written into $\mathbf c_t$ at step 2, kept by $\mathbf f_t\approx 1$, read by $\mathbf o_t$ at step 7; the same input in a plain RNN fades | Shows the gates as control of a flow, which a still cannot | `s05_a02_lstm_memory.mp4` |
| **S05-A03** | V36 | Matrix $QK^\top$ filled cell by cell, then the mask turns the upper triangle to $-\infty$, then the row-wise softmax; finally one row read as "token 5 looks at tokens 1–5" | Order of operations of masked attention | `s05_a03_causal_mask.mp4` |
| **S05-A04** | V49 | Autoregressive generation: context → distribution over characters → sample → append → shift; then the same distribution with $\tau$ small and large | Generation is a loop; temperature is a deformation | `s05_a04_generation.mp4` |

## Notebooks

Short, self-contained, one change per cell, Chollet DLwP style in PyTorch, comments that link code and slide notation, open-in badges for Colab, Kaggle and Lightning.

| # | Notebook | Question | Data | Time |
|---|---|---|---|---|
| 5.1 | `sesion_05_1_secuencias_rnn.ipynb` | Windows and temporal split; commonsense baseline; flattened MLP; an RNN cell written as a loop and checked against `nn.RNN`; gradient norm vs $t$ (F02, F04) | Jena Climate | GPU, ~10 min |
| 5.2 | `sesion_05_2_lstm_gru.ipynb` | LSTM and GRU on the same windows; the same gradient plot; window of 20 h vs 5 days; recurrent dropout | Jena Climate | GPU, ~10 min |
| 5.3 | `sesion_05_3_atencion_transformer.ipynb` | Multi-head attention from the S03 function, checked against `nn.MultiheadAttention`; causal mask; cost vs $T$ (F07); a small Transformer on Jena against the LSTM; without position and with shuffled steps | Jena Climate | GPU, ~15 min |
| 5.4 | `sesion_05_4_preentrenamiento_gpt.ipynb` | Character tokenizer and a BPE demo cell; a small GPT from the block of 5.3; samples before and after ~5 min of training; temperature; homework section: the scaling sweep (F09) | *Don Quijote* (Gutenberg) | GPU, ~15 min |
| 5.5 | `sesion_05_5_modelos_fundacionales.ipynb` | Chronos-Bolt zero-shot on the Jena test windows against the LSTM of 5.2, at 1 h and 24 h; CLIP zero-shot on CIFAR-10 against the S04 probes; optional TabPFN on a Covertype subset | Jena, CIFAR-10, Covertype | GPU, ~15 min (homework) |
| 5.6 | `sesion_05_6_integrador_bike_sharing.ipynb` | Integrating exercise (decision 2) with MLflow | Bike Sharing | GPU |

Proposed TODOs for 5.6:

1. temporal split and windows without leakage (the scaler fit on training only; no window crosses the cut);
2. the recurrent model (LSTM or GRU) and its training run;
3. Chronos-Bolt zero-shot on **the same** test windows, the same horizon and the same metric, logged as a run;
4. `forecast_raw`, which loads the chosen model from MLflow, takes raw hourly records, rejects those that break the contract (gaps, wrong columns, out-of-range values) and returns the forecast.

Data loading follows the S03–S04 pattern: Kaggle dataset with *Add Input*, `kagglehub` with `KAGGLE_API_TOKEN` on Colab, local `data/` on Lightning. Kaggle handles still to choose for Jena, Bike Sharing and the text. The Chronos and CLIP weights come from Hugging Face, so *Internet* must be on.

---

## Reusable material from jdmartinev/si7011-deeplearning

| Material | Path | Use in S05 | Changes needed |
|---|---|---|---|
| Jena Climate RNN notebooks | `Lecture05/notebooks/rnns/part-1-data` … `part-6-gru` | Base for 5.1–5.2: loading, 50/25/25 temporal split, `StandardScaler` fit on train, `TimeseriesDataset` with `sampling_rate=6`, `sequence_length=120`, 24 h delay (Chollet DLwP ch. 10), commonsense baseline, RNN/LSTM/GRU and stacked/bidirectional variants | Five near-identical notebooks repeat the dataset, loop and callbacks: merge into two and reuse the S02–S04 loop. Data downloaded with `wget` from S3 → Kaggle pattern. TensorBoard → plots in the notebook, MLflow only in 5.6. Each notebook depends on `data/processed` from part 1; make each one self-contained. Add the gradient-vs-time measurement, which is not there |
| RNN lecture | `Lecture05/Lecture05.ipynb` and `Lecture05.slides.html` | Narrative order for Part 1: bag of words, 1-D CNN, why sequences, task types | Figures `Figs/RNN*.PNG` come from external material (Raschka's lecture slides); redraw them (F01, F03, A01). Text in Spanish → the deck is in English |
| Bike Sharing exercise | `Lecture05/notebooks/excercise/part-1-data.ipynb`, `part-2-model.ipynb` | Base for 5.6: cyclical encoding of hour and month, one-hot features, lags, temporal split, commonsense baseline | Lagged features are for scikit-learn; the window replaces them for the RNN. Add the data contract and the Chronos comparison. Part 2 is almost empty (only tips): the TODOs are new |
| Text and Transformer lectures | `Lecture06/Clase06a - Preprocesamiento de texto.pdf`, `Clase06b - Transformers.pdf` | Narrative for V43, V47 and Part 3 | Text only; diagrams redrawn (F05, F06, F08) |
| Tokenization comparison | `Lecture06/notebooks/excercise/tweeteval-part-1-data.ipynb` | Demo cell of 5.4: the same sentence through several tokenizers, token counts | Take only the comparison cell |
| Fine-tuning, LoRA, deployment | `Lecture06/notebooks/rnns/part-2` … `part-6`, `excercise/tweeteval-part-3` … `part-6` | **Not in S05.** Belongs to S06 (fine-tuning vs feature extraction vs LoRA, FastAPI) | |

Not in the old repo: attention from scratch with masks, cost measurements, a GPT from scratch, scaling sweeps, foundation models used zero-shot. The instructor's TinyGPT and GPT-from-scratch materials outside this repo can be the base for 5.4.

## Link to Evaluation Event 3 (S05–S06)

Concepts S05 adds, phrased as decisions to justify:

- reading a gradient-vs-time plot and explaining why the LSTM keeps signal where the RNN does not;
- predicting what shuffling the steps does to a model with and without position;
- explaining the $T^2$ cost and when it decides the architecture;
- distinguishing a Transformer from a foundation model, and zero-shot from fine-tuning;
- deciding with evidence whether a zero-shot foundation model is worth it against a small trained model (the integrating exercise).

## Open questions for the instructor

1. Text corpus: *Don Quijote* at character level, or something closer to the students' work (for example a corpus of Spanish news or the course's own notes)?
2. Practice B: 5.3 + 5.4 in class and 5.5 at home (proposal), or 5.3 + 5.5 in class and the GPT at home?
3. Integrating exercise: Bike Sharing (reuses the old exercise) or Jena with a different target, so the students do not change dataset?
4. Evaluation Event 2 (S03–S04) is not yet in `evaluations/`. Does it need to be ready before S05?
