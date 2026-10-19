# SI7011 — Session 05 · Sequences

## Slide-by-slide plan (draft)

> **How does a network remember the past?**

Instructor planning document, same format as `S04_plan.md`. It fixes the deck structure, the content of each slide, the figures and animations, and the notebooks before anything is built. Conventions of Sessions 02–04: slides in English, notebooks in Spanish, five parts opened by divider slides that recap the previous part, one prediction-before-experiment slide per part, two practice blocks and an exit question. Folder: `sessions/05_sequences/`.

**Status: built.** Deck, figures, animations and notebooks 5.1–5.6 are in `sessions/05_sequences/`. Decisions below are adopted; measured results are in the notebook table.

## Course change (adopted)

S05 covers **sequences only**: recurrence, LSTM/GRU, text as a sequence, seq2seq and the first form of attention. Transformers, pretraining, scaling and foundation models move to **S06**, which closes with adaptation (frozen features, fine-tuning, and LoRA mentioned). Distillation, quantization, distribution shift, calibration and deployment are no longer covered.

| | Before | Now |
|---|---|---|
| S05 | RNN, LSTM/GRU, attention, Transformers, pretraining, scaling, foundation models | **Sequences**: RNN, BPTT, LSTM/GRU, text and embeddings, seq2seq, Bahdanau attention |
| S06 | Fine-tuning, PEFT, distillation, quantization, OOD, calibration, deployment | **Transformers and foundation models**: full self-attention, the block, pretraining (GPT/BERT), scaling, foundation models (language, vision, time series), adaptation; LoRA mentioned |

Forward references updated in this branch: S02 slides (GELU, pre-norm block), notebooks 2.3 and 2.4, S03 slides and notebook 3.5 (heads, masks, cost), S03 `figures.md`, S04 slides and notebook 4.3 (BERT, CLIP). The S02, S03 and S04 PDFs must be re-rendered.

## Sources

| Source | What it gives this session |
|---|---|
| `course/course_structure.md` (EAFIT-IA) | Updated S05 topic list (this branch) |
| `course/evaluation_strategy.md` (EAFIT-IA) | Event 3 covers S05–S06: sequence modeling, attention, Transformers, pretrained and foundation models. Events evaluate decisions and evidence, not implementation |
| S02–S04 | S02: depth as a product of Jacobians, residuals, gradient clipping, regularization. S03: weight sharing, 1-D thinking about receptive fields, the permutation test of notebook 3.5. S04: `nn.Embedding` as lookup (V08), self-supervision |
| jdmartinev/si7011-deeplearning, `Lecture05` and `Lecture06` | Jena Climate notebooks (RNN, LSTM, GRU), Bike Sharing exercise, RNN lecture, text-preprocessing lecture. Details in [Reusable material](#reusable-material-from-jdmartinevsi7011-deeplearning) |

## Decisions (adopted)

1. **Two datasets, one per half.** Parts 1–3 use **Jena Climate** (temperature 24 h ahead from 14 variables every 10 min), already prepared in the old repo. Part 4 uses **IMDB** (50 000 English reviews, binary sentiment), because the sentiment of a review mostly depends on which words appear, which makes the word-order experiment informative. Part 5 uses a small synthetic task: Spanish sentences that contain a date in words ("el contrato se firmó el veintinueve de septiembre de mil novecientos noventa y ocho sin falta") converted to ISO format (`1998-09-29`). It trains in about a minute per model on CPU and its attention map is readable. Measured in notebook 5.5: without attention the decoder needs $d \approx 100$ (0.4 % exact at $d = 24$, 93 % at $d = 128$; between 48 and 96 the result depends on the seed); with attention $d = 24$ gives 100 %. Length is not what limits it here: trained on all lengths, the plain model does not degrade with length up to 27 words, and trained on short sentences only, the attention model extrapolates worse to longer ones.
2. **Integrating exercise: Bike Sharing.** *How many hours ahead can you forecast bike demand better than the obvious baseline?* Hourly demand (UCI), the old exercise. Data contract, temporal split, windows without leakage, LSTM/GRU, multi-step horizon, MLflow, `forecast_raw`. Four TODOs as in S02–S04. S06 can come back to the same series with a zero-shot time-series foundation model (Chronos) and compare.
3. **Attention ends the session, not the Transformer.** Part 5 introduces attention as the fix for the seq2seq bottleneck (Bahdanau, 2015). Self-attention, heads, masks and cost stay for S06. The last slide is the bridge: "what if we keep the attention and drop the recurrence?"
4. **Homework.** Notebook 5.5 (seq2seq with attention) is an optional homework; the slides show its results. Practice B is notebook 5.4.

## Notation

Consistent with S01–S04 (φ is the activation, σ only the sigmoid, *L* the loss, *J* the objective). S01 and S04 already call a hidden representation **h**; the hidden state keeps that letter.

| Symbol | Meaning |
|---|---|
| $\mathbf x_t$ | input at time step $t$, $t = 1,\dots,T$ |
| $\mathbf h_t$ | hidden state; $\mathbf h_t = \varphi(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b)$ |
| $\mathbf c_t$ | LSTM cell state |
| $\mathbf f_t, \mathbf i_t, \mathbf o_t$ | forget, input and output gates (σ, values in $(0,1)$) |
| $\odot$ | element-wise product |
| $T$ | sequence length (window); $T'$ output length in seq2seq |
| $H_{\text{fut}}$ | forecast horizon in steps (avoids clashing with $H$ heads in S06) |
| $\mathbf e_w$ | embedding of token $w$ (row of the embedding matrix $E$) |
| $\mathbf s_\tau$ | decoder state at output step $\tau$ |
| $\alpha_{\tau t}$ | attention weight of output step $\tau$ on input step $t$; $\sum_t \alpha_{\tau t} = 1$ |
| $\mathbf a_\tau = \sum_t \alpha_{\tau t}\mathbf h_t$ | context vector |

## Timing

| Time | Block | Slides / notebooks |
|---|---|---|
| 00:00–00:55 | Theory A | Opening, Parts 1–3 (V01–V33) |
| 00:55–01:45 | Practice A | V34 · notebooks 5.1–5.3 |
| 01:45–01:55 | Break | |
| 01:55–02:50 | Theory B | Parts 4–5 (V35–V58) |
| 02:50–03:45 | Practice B | V59 · notebook 5.4 (5.5 optional, at home) |
| 03:45–04:00 | Closure | V60–V64 |

Theory A has more slides than in S04, but Part 1 is light (data handling the students already know from S02) and moves fast.

## Deck structure

64 slides (S02 78, S03 75, S04 66).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Sequences as data | V05–V13 | |
| 2 · Recurrence and its gradients | V14–V24 | |
| 3 · Memory that lasts: LSTM and GRU | V25–V34 | Practice A (V34): notebooks 5.1–5.3 |
| 4 · Text as a sequence | V35–V46 | |
| 5 · Sequence to sequence and attention | V47–V59 | Practice B (V59): notebook 5.4; 5.5 optional at home |
| Closure | V60–V64 | Integrating exercise (V60), exit question (V64) |

Prediction-before-experiment slides: V12, V21, V31, V44, V56.

---

## Opening (V01–V04)

| # | Slide | Content | Visual |
|---|---|---|---|
| V01 | How does a network remember the past? | Title, session question | |
| V02 | Roadmap | The five parts and the two practice blocks | |
| V03 | From fixed inputs to sequences | S03–S04 recap: architecture and objective shape the representation. So far every input had a fixed size. Now its length changes and its order matters | |
| V04 | The question for today | A network sees one step at a time. What should it keep from what came before, and how does the gradient reach a step that was long ago? | |

## Part 1 · Sequences as data (V05–V13)

| # | Slide | Content | Visual |
|---|---|---|---|
| V05 | Part 1 · Sequences as data | Divider. Recap of S03: architecture encodes assumptions about the input | |
| V06 | Order matters | Text, sensors, audio, prices, gait. Shuffling the steps can destroy the meaning. Callback to S03: shuffling pixels hurt the CNN, not the MLP | |
| V07 | Tasks on sequences | Many-to-one (sentiment, next value), aligned many-to-many (tagging), seq-to-seq (translation), one-to-many (generation) | |
| V08 | *(figure)* | **S05-F01** | F01 |
| V09 | From a series to examples | Sliding window: $T$ past steps → value $H_{\text{fut}}$ steps ahead. Jena: one reading every 10 min, sampled every hour, $T = 120$ hours, target 24 h ahead | |
| V10 | *(figure)* | **S05-F02** | F02 |
| V11 | Three ways to leak the future | Shuffling before the split; normalizing with statistics of the whole series; windows that cross the cut. Each one gives a validation score the test will not repeat. Data-contract callback (S02–S04 integrators) | |
| V12 | Prediction before experiment | Jena: commonsense baseline ("the temperature in 24 h equals the temperature now") against a flattened MLP over 120 h × 14 variables. Which one wins on validation MAE? Answer in notebook 5.1 | |
| V13 | Why not an MLP or a CNN? | Flattened MLP: parameters grow with $T$, no sharing along time, order is only a position in a vector. 1-D convolution: shares weights (S03) but sees a fixed receptive field. Wanted: same weights at each step and a summary of everything before | |

## Part 2 · Recurrence and its gradients (V14–V24)

| # | Slide | Content | Visual |
|---|---|---|---|
| V14 | Part 2 · Recurrence | Divider. Recap: a series becomes windows; the baseline is the bar | |
| V15 | The recurrent cell | $\mathbf h_t = \varphi(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b)$. The same weights at every step: weight sharing along time. $\mathbf h_t$ summarizes $\mathbf x_{1:t}$ in a fixed-size vector | |
| V16 | Unrolling through time | **S05-A01** (media) | A01 |
| V17 | Shapes in PyTorch | `nn.RNN(input_size, hidden_size, batch_first=True)`: input $(B, T, d_x)$, output $(B, T, d_h)$ and final state. Many-to-one takes `out[:, -1]`. Written as a Python loop in notebook 5.2 and checked against `nn.RNN` | |
| V18 | Backpropagation through time | The unrolled graph is a deep network with tied weights. $\dfrac{\partial \mathbf h_T}{\partial \mathbf h_t} = \prod_{k=t+1}^{T} \operatorname{diag}\!\big(\varphi'(\cdot)\big)\,W_{hh}$. The product of Jacobians of S02, with the **same** matrix repeated | |
| V19 | Vanishing and exploding | Largest singular value of $W_{hh}$ times $\max\varphi'$ below 1: the signal from step $t$ fades geometrically; above 1: it can explode. With tanh, $\varphi' \le 1$, so fading is the usual case | |
| V20 | *(figure)* | **S05-F03** | F03 |
| V21 | Prediction before experiment | Same Jena setup with a 16-unit RNN. Does it beat the baseline? And with a window of 1 day instead of 5: better, worse or the same? | |
| V22 | Gradient clipping | Rescale $\nabla_\theta L$ when its norm exceeds a threshold: `clip_grad_norm_`. It stops explosions; it does nothing for fading. Callback to S02 diagnostics | |
| V23 | Truncated BPTT | Backpropagate only $k$ steps; carry $\mathbf h$ forward detached. Cheaper, and it limits by design how far back the network can learn | |
| V24 | What we need | A path along time where the gradient is not multiplied by $W_{hh}$ at every step. S02 had the answer for depth: an additive (residual) path | |

## Part 3 · Memory that lasts: LSTM and GRU (V25–V34)

| # | Slide | Content | Visual |
|---|---|---|---|
| V25 | Part 3 · LSTM and GRU | Divider. Recap: the repeated product of Jacobians wipes out the past | |
| V26 | An additive path for memory | $\mathbf c_t = \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\tilde{\mathbf c}_t$. Addition instead of repeated multiplication by $W_{hh}$: the residual idea of S02 along time. With $\mathbf f_t\approx 1$, $\partial\mathbf c_t/\partial\mathbf c_{t-1}\approx I$ | |
| V27 | *(figure)* | **S05-F04** | F04 |
| V28 | Three gates | $\mathbf f_t, \mathbf i_t, \mathbf o_t = \sigma(\cdot)$ from $[\mathbf h_{t-1}, \mathbf x_t]$: what to forget, what to write, what to show. $\mathbf h_t = \mathbf o_t\odot\tanh(\mathbf c_t)$. A gate is a learned multiplier that depends on the input. Initializing the forget bias to 1 | |
| V29 | The memory in motion | **S05-A02** (media) | A02 |
| V30 | GRU | Two gates (update and reset), no separate $\mathbf c_t$, about 3/4 of the LSTM parameters, often similar results. Pick by validation, not by habit | |
| V31 | Prediction before experiment | Notebook 5.3: LSTM and GRU on the same windows. Do they beat the RNN and the baseline? Does the gradient-vs-time plot change shape? | |
| V32 | Stacked and bidirectional | Stacking: $\mathbf h_t$ of one layer is $\mathbf x_t$ of the next. Bidirectional: two passes, forward and backward; only valid when the whole sequence is available when you predict (text, yes; forecasting the future from the past, the window can be read both ways, but never beyond its end) | |
| V33 | Regularizing recurrent networks | Dropout on inputs and on the recurrent connection with the same mask at every step (Gal & Ghahramani, 2016), weight decay, early stopping. Callback to S02 | |
| V34 | Practice A — Recurrent networks on Jena | Notebooks 5.1, 5.2 and 5.3. What to compare, how long, what to write down | |

## Part 4 · Text as a sequence (V35–V46)

| # | Slide | Content | Visual |
|---|---|---|---|
| V35 | Part 4 · Text as a sequence | Divider. Recap: an LSTM keeps signal over long windows. Now the steps are words | |
| V36 | From text to integers | Tokenize, build a vocabulary from training only, map to ids, unknown token, special tokens. Words vs characters vs subwords in one line each (subwords in S06) | |
| V37 | *(figure)* | **S05-F05** | F05 |
| V38 | Embeddings | $\mathbf e_w = E[w]$: `nn.Embedding` is one-hot × matrix (S04 V08). Learned with the task, or initialized from word2vec/GloVe. Distance in $E$ comes to mean similarity | |
| V39 | Batches of different lengths | Pad to the longest in the batch, mask the padding (or `pack_padded_sequence`), sort or bucket by length to waste less compute. Truncate long reviews: a design choice to report | |
| V40 | Three classifiers | Bag of embeddings (average $\mathbf e_w$, then a linear layer: no order), 1-D CNN over embeddings (local order), LSTM (full order). Same embedding size, same budget | |
| V41 | *(figure)* | **S05-F06** | F06 |
| V42 | What the embedding learned | Nearest neighbors of a few words in $E$ after training on sentiment: neighbors by polarity, not by meaning. An embedding is useful *for its task* (S04 V07) | |
| V43 | *(figure)* | **S05-F07** | F07 |
| V44 | Prediction before experiment | IMDB: shuffle the words of every test review. What happens to the bag of embeddings (nothing, by construction)? To the LSTM? Does the gap say order matters for sentiment? | |
| V45 | When order matters | Sentiment is close to bag of words; negation ("not good"), irony and long-range agreement are not. The task decides whether the order-aware model is worth it | |
| V46 | The limit of a single vector | Many-to-one keeps only $\mathbf h_T$ (or a pooling of all $\mathbf h_t$). For seq-to-seq that one vector must carry the whole input. Bridge to Part 5 | |

## Part 5 · Sequence to sequence and attention (V47–V59)

| # | Slide | Content | Visual |
|---|---|---|---|
| V47 | Part 5 · Sequence to sequence | Divider. Recap: one vector summarizes the sequence | |
| V48 | Encoder and decoder | Encoder RNN reads $\mathbf x_{1:T}$ → $\mathbf h_T$; decoder RNN starts from it and emits $y_1,\dots,y_{T'}$ one token at a time (Sutskever et al., 2014). Encoder-decoder as in S04, now in time | |
| V49 | *(figure)* | **S05-F08** | F08 |
| V50 | Teacher forcing | Training: the decoder gets the true previous token. Inference: it gets its own prediction, so errors accumulate (exposure bias). Greedy decoding; beam search in one line | |
| V51 | The bottleneck | Everything the decoder knows passes through $\mathbf h_T$. Performance falls with input length (Cho et al., 2014) | |
| V52 | Look back instead of remembering | At each output step, score every encoder state, normalize with softmax, average: $\alpha_{\tau t} = \operatorname{softmax}_t\big(\operatorname{score}(\mathbf s_{\tau-1}, \mathbf h_t)\big)$, $\mathbf a_\tau = \sum_t \alpha_{\tau t}\mathbf h_t$ (Bahdanau et al., 2015) | |
| V53 | Attention in motion | **S05-A03** (media) | A03 |
| V54 | Score functions | Additive (a small MLP, Bahdanau), dot product $\mathbf s^\top\mathbf h$ and scaled dot product (Luong, 2015). The dot product is the one that survives in S06 | |
| V55 | *(figure)* | **S05-F09** | F09 |
| V56 | Prediction before experiment | Dates in sentences: seq2seq without attention at $d = 128$ and $d = 24$, then with attention at $d = 24$. Which part of the date is lost first without attention? Does attention fix it at the same $d$? (notebook 5.5) | |
| V57 | What attention changed | The decoder reads any input position directly: path length 1 instead of $T$. The weights are interpretable as an alignment, with care (S03 F11) | |
| V58 | Keep the attention, drop the recurrence? | Recurrence is still sequential: step $t$ waits for $t-1$, training does not parallelize along $T$. If each step can look at every other step, is the recurrence needed? Bridge to S06 | |
| V59 | Practice B — Text and seq2seq | Notebook 5.4 in class; 5.5 optional at home | |

## Closure (V60–V64)

| # | Slide | Content | Visual |
|---|---|---|---|
| V60 | Integrating exercise — Bike Sharing | See decision 2. Four TODOs, as in S02–S04 | |
| V61 | Choosing a sequence model | Table: baseline / MLP on the window / 1-D CNN / RNN / LSTM-GRU / seq2seq with attention × how far it sees, parameters, when it is enough | |
| V62 | What each choice assumes | Format of S03 V69 and S04 V64: "assumes" and "pays with". The window assumes the past beyond $T$ does not matter; the RNN assumes the past fits in $\mathbf h_t$; bag of embeddings assumes order does not matter | |
| V63 | *(figure)* | **S05-F10** | F10 |
| V64 | Exit question | A hospital has two years of hourly bed occupancy and wants a 48 h forecast. Choose two and justify: commonsense baseline with weekly seasonality; LSTM with a 48 h window; LSTM with a 2-week window; a bidirectional LSTM over the window; check first that the split leaves the last months for test | |

---

## Figures

Same routes as S03 and S04: **Generator** (prompts in `figure_prompts.md`), **Script** (`make_s05_*.py`, fixed seeds, numbers taken from the notebooks) and **Manim** (`escenas_s05.py`). All figures drawn from scratch.

| ID | Slide | Purpose | Planned filename | Route |
|---|---:|---|---|---|
| **S05-F01** | V08 | Four task types on sequences | `s05_f01_sequence_tasks.png` | Generator |
| **S05-F02** | V10 | Jena: sliding windows, 24 h target, train/val/test cut with no window crossing it | `s05_f02_windows_split.png` | Script (notebook 5.1) |
| **S05-F03** | V20 | Gradient norm $\lVert\partial L/\partial\mathbf x_t\rVert$ against $t$ for the RNN (log scale) | `s05_f03_rnn_gradient_time.png` | Script (notebook 5.2) |
| **S05-F04** | V27 | LSTM cell with the $\mathbf c_t$ path drawn as the additive line, gates as valves | `s05_f04_lstm_cell.png` | Generator |
| **S05-F05** | V37 | Text pipeline: sentence → tokens → ids → padded batch with mask → embeddings | `s05_f05_text_pipeline.png` | Generator |
| **S05-F06** | V41 | Bag of embeddings / 1-D CNN / LSTM over the same sentence, what each one sees | `s05_f06_three_text_models.png` | Generator |
| **S05-F07** | V43 | Nearest neighbors in the learned embedding for 6 words | `s05_f07_embedding_neighbors.png` | Script (notebook 5.4) |
| **S05-F08** | V49 | Encoder-decoder RNN with the single vector $\mathbf h_T$ highlighted | `s05_f08_seq2seq.png` | Generator |
| **S05-F09** | V55 | Attention matrix $\alpha_{\tau t}$ for three dates: output characters × input words | `s05_f09_attention_alignment.png` | Script (notebook 5.5) |
| **S05-F10** | V63 | Summary: memory over time in each model (window, RNN, LSTM, attention) | `s05_f10_summary.png` | Generator |

Also a comparison figure for Jena (baseline, MLP, RNN, LSTM, GRU validation MAE, same axes) drawn from notebooks 5.1–5.3, if a slide needs it.

## Animations (Manim)

Only where motion changes understanding. Conventions of `escenas_s02.py`–`escenas_s04.py`: slide palette, on-screen text in English, `self.pausa()` before each reveal, poster from the last frame.

| ID | Slide | What it shows | Why animate | Rendered file |
|---|---:|---|---|---|
| **S05-A01** | V16 | The RNN cell rolled up, then unrolled over 8 steps with the same $W_{hh}$ highlighted at every step; then the gradient flows back from step 8 and its arrow thins step by step | Weight sharing in time and the fading are both temporal | `s05_a01_unrolling_bptt.mp4` |
| **S05-A02** | V29 | One value written into $\mathbf c_t$ at step 2, kept by $\mathbf f_t\approx 1$, read by $\mathbf o_t$ at step 7; the same input in a plain RNN fades | The gates as control of a flow | `s05_a02_lstm_memory.mp4` |
| **S05-A03** | V53 | Decoding a date: at each output step the scores over the input words appear, the softmax turns them into $\alpha_{\tau t}$, the context vector forms, the character is emitted; the matrix of F09 fills row by row | Attention as a per-step lookup | `s05_a03_seq2seq_attention.mp4` |

## Notebooks

Short, self-contained, one change per cell, Chollet DLwP style in PyTorch, comments that link code and slide notation, open-in badges for Colab, Kaggle and Lightning.

| # | Notebook | Question | Data | Time |
|---|---|---|---|---|
| 5.1 | `sesion_05_1_series_ventanas.ipynb` | Windows, temporal split and the three leaks (shown, then fixed); commonsense baseline; flattened MLP; 1-D CNN (F02) | Jena Climate | CPU/GPU, ~8 min |
| 5.2 | `sesion_05_2_rnn_bptt.ipynb` | An RNN cell as a Python loop, checked against `nn.RNN`; training on Jena; gradient norm vs $t$ (F03); clipping; window of 1 day vs 5 days | Jena Climate | GPU, ~10 min |
| 5.3 | `sesion_05_3_lstm_gru.ipynb` | LSTM cell as a loop checked against `nn.LSTM`; LSTM and GRU on the same windows (val 2.32 and 2.30 °C vs RNN 2.33); gradient per hour for the three cells (F11: RNN falls to 0, LSTM and GRU flatten near 1e-3); LSTM with 1 day vs 5 days (same MAE); 2 layers with dropout between layers; recurrent dropout and the forget-gate bias in *Tu turno* | Jena Climate | CPU, ~4 min |
| 5.4 | `sesion_05_4_texto_embeddings.ipynb` | Tokenizer and vocabulary from train (20 000 words, 256-token truncation); padding and masks; bag of embeddings 85.4 %, 1-D CNN 84.1 %, LSTM 82.6 % (3 epochs) on IMDB test; shuffled words: 85.4 / 79.6 / 80.7 %; six hand-written negation sentences; neighbors in $E$ (F07); pretrained GloVe 100d frozen (75.3 %) and fine-tuned (85.9 %), with neighbors by meaning | IMDB (Hugging Face `stanfordnlp/imdb`) | GPU ~3 min (CPU ~7 min) |
| 5.5 | `sesion_05_5_seq2seq_atencion.ipynb` | Dates in sentences → ISO: without attention at $d = 128$ and $d = 24$, accuracy per output position; with additive attention at $d = 24$; attention map (F09); $d$ sweep 48–96 | synthetic (generated in the notebook) | CPU, ~7 min |
| 5.6 | `sesion_05_6_integrador_bike_sharing.ipynb` | Integrating exercise (decision 2) with MLflow: 72 h of demand, weather and calendar → demand at 1, 6, 12 and 24 h. Data contract with the 165 missing hours (75 gaps); split by date (train to 2012-06, validation 2012-07 to 09, test 2012-10 to 12). Reference solution (GRU 32, 12 epochs, not committed), validation MAE 37 / 51 / 52 / 58 bikes per hour vs best baseline 84 / 84 / 83 / 83 (improvement 56 % at 1 h, 30 % at 24 h); test 31 / 49 / 47 / 54 vs 76 / 76 / 76 / 77 | Bike Sharing (UCI `hour.csv`) | CPU, <1 min per run |

TODOs of 5.6, in the S03–S04 style (committed without solutions; the notebook has no outputs):

1. temporal split and windows without leakage (scaler fit on train only, no window crossing a cut);
2. the recurrent model (LSTM or GRU) and its training run, logged to MLflow;
3. evaluation by horizon: MAE at 1, 6, 12 and 24 h against the baseline; at which horizon does the model stop paying off?
4. `forecast_raw`, which loads the chosen model from MLflow, takes raw hourly records, rejects those that break the contract (gaps, wrong columns, out-of-range values) and returns the forecast.

Data loading follows the S03–S04 pattern: Kaggle dataset with *Add Input*, `kagglehub` with `KAGGLE_API_TOKEN` on Colab, local `data/` on Lightning. Jena: Kaggle `mnassrib/jena-climate`. IMDB: Hugging Face `stanfordnlp/imdb` parquet. Bike Sharing: UCI zip (downloaded by the notebook if `hour.csv` is not in `data/` or a Kaggle input); Kaggle handle still to confirm.

---

## Reusable material from jdmartinev/si7011-deeplearning

| Material | Path | Use in S05 | Changes needed |
|---|---|---|---|
| Jena Climate RNN notebooks | `Lecture05/notebooks/rnns/part-1-data` … `part-6-gru` | Base for 5.1–5.3: loading, 50/25/25 temporal split, `StandardScaler` fit on train, `TimeseriesDataset` with `sampling_rate=6`, `sequence_length=120`, 24 h delay (Chollet DLwP ch. 10), commonsense baseline, RNN/LSTM/GRU, stacked and bidirectional | Five near-identical notebooks repeat the dataset, loop and callbacks: one change per notebook with the S02–S04 loop. Data downloaded with `wget` from S3 → Kaggle pattern. TensorBoard → plots in the notebook; MLflow only in 5.6. Each part depends on `data/processed` from part 1; make each notebook self-contained. Add the explicit RNN loop, the gradient-vs-time measurement and the leakage demo, which are not there |
| RNN lecture | `Lecture05/Lecture05.ipynb` and `Lecture05.slides.html` | Narrative for Parts 1 and 4: bag of words, 1-D CNN for text, why sequences, task types | Figures `Figs/RNN*.PNG` come from external material; redraw (F01, F04, F06, A01). Spanish text → English deck |
| Bike Sharing exercise | `Lecture05/notebooks/excercise/part-1-data.ipynb`, `part-2-model.ipynb` | Base for 5.6: cyclical encoding of hour and month, one-hot features, temporal split, commonsense baseline | The lagged features are for scikit-learn; the window replaces them. Add the data contract and the per-horizon evaluation. Part 2 is almost empty: the TODOs are new |
| Text-preprocessing lecture | `Lecture06/Clase06a - Preprocesamiento de texto.pdf` | Narrative for V36–V39 | Text only; F05 redrawn |
| Transformer lecture, fine-tuning, LoRA, deployment | `Lecture06/Clase06b - Transformers.pdf`, `Lecture06/notebooks/*` | **Not in S05.** Transformers and fine-tuning go to S06; deployment is no longer covered | |

Not in the old repo: the leakage demo, the explicit RNN loop, gradient-vs-time plots, text classifiers with embeddings, seq2seq and attention. Notebooks 5.4 and 5.5 are new.

## Link to Evaluation Event 3 (S05–S06)

Concepts S05 adds, phrased as decisions to justify:

- spotting a temporal leak from how a split or a scaler was built;
- reading a gradient-vs-time plot and explaining why the LSTM keeps signal where the RNN does not;
- deciding from a shuffled-words experiment whether a task needs an order-aware model;
- explaining the seq2seq bottleneck and what attention changes, from an accuracy-vs-length curve or an attention map.

## Resolved and open questions

Resolved: 5.4 uses IMDB (English); 5.5 uses Spanish dates in words → ISO; 5.5 is an optional homework.

Open:

1. Kaggle handle for Bike Sharing (5.6 downloads the UCI zip meanwhile, so *Internet* must be on).
2. Evaluation Event 2 (S03–S04) is not yet in `evaluations/`.
