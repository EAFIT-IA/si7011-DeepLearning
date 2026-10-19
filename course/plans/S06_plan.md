# SI7011 — Session 06 · Transformers and Foundation Models

## Slide-by-slide plan (draft)

> **What if we keep the attention and drop the recurrence?**

Instructor planning document, same format as `S04_plan.md` and `S05_plan.md`. It fixes the deck structure, the content of each slide, the figures and animations, and the notebooks before anything is built. Conventions of Sessions 02–05: slides in English, notebooks in Spanish, five parts opened by divider slides that recap the previous part, one prediction-before-experiment slide per part, two practice blocks and an exit question. Folder: `sessions/06_foundation_models/` (the name already used in the README of branch `s05-plan`).

**Status: draft.** Scope adopted with the S05 restructure (see `S05_plan.md`, *Course change*); the decisions below are proposals.

## Scope

From `course/course_structure.md` (branch `s05-plan`):

- self-attention: heads, causal masks, cost
- the Transformer block
- pretraining: next-token and masked prediction
- scaling
- foundation-model concept
- vision, time-series and tabular foundation models
- using a pretrained model: zero-shot, frozen features, fine-tuning
- PEFT / LoRA (mentioned)

Not covered: distillation, quantization, distribution shift, calibration, serving and deployment.

## Promises made by earlier sessions

S06 must keep these. Each row is a forward reference already in the slides or notebooks.

| Where | What it promises | Kept in |
|---|---|---|
| S02 slides (GELU, line 360) | GELU is the default activation in Transformers | V22 |
| S02 slides (LayerNorm, line 827) | LayerNorm is the standard choice in Transformers | V21 |
| S02 slides (pre-norm residual, line 1110) | "The identity path stays clean; this is the Transformer block of S06" | V20–V21, A02 |
| S03 V-"Self-attention in one slide" | "The full mechanics — heads, masks, cost — come in Session 06" | Part 1 |
| S03 positional embedding, permutation test | Attention is permutation-equivariant; position must be added | V15–V16, notebook 6.1 |
| S04 V34 | BERT masks only 15 % of the words | V36 |
| S04 V51 | CLIP is a foundation model | V48, notebook 6.4 |
| S05 V36 (table) | Subwords (BPE): Session 06 | V33 |
| S05 V54 | The scaled dot product is the one that survives | V08 |
| S05 V58 | "Is the recurrence still needed?" | V03–V04 |
| S05 decision 2 | Return to Bike Sharing with a zero-shot time-series foundation model (Chronos) | V50, notebook 6.4 |

## Sources

| Source | What it gives this session |
|---|---|
| `course/course_structure.md` (branch `s05-plan`) | Topic list above |
| `course/evaluation_strategy.md` | Event 3 covers S05–S06. Project, S06: "evaluate whether a pretrained or foundation model is relevant to the problem" |
| S02–S05 | Pre-norm residual block, LayerNorm, GELU, AdamW + warmup + cosine (S02); patches, tokens, ViT, positional embedding, three transfer strategies (S03); masked prediction, InfoNCE, probes, label efficiency (S04); text pipeline, IMDB baselines, Bahdanau attention, the dates task (S05) |
| MIT 6.S191, Lab 3 *Fine-Tune an LLM* ([MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning/tree/master/lab3), MIT License) | Base for the integrating exercise: chat template, loss masked to the answer, LoRA on a ~1 B decoder with a hand-written loop, held-out log-likelihood. Details in [Reusable material](#reusable-material-from-mit-6s191-lab-3) |
| jdmartinev/si7011-deeplearning, `Lecture06` | Transformer lecture, IMDB DistilBERT notebooks (full fine-tuning, feature extraction, LoRA, comparison), TweetEval exercise (not used). Details in [Reusable material](#reusable-material-from-jdmartinevsi7011-deeplearning) |

## Decisions (proposals)

1. **Same tasks as before, new model.** Every experiment in S06 reruns a task the students already solved, so each number has a reference on the slide next to it:
   - notebook 6.2: the **dates task of S05** (notebook 5.5) with a small Transformer encoder-decoder instead of the RNN with attention;
   - notebook 6.4: **EuroSAT** (S03–S04) with CLIP zero-shot and CLIP probe, and **Bike Sharing** (S05 integrator) with Chronos zero-shot;
   - notebook 6.5: **IMDB** (S05 notebook 5.4: bag of embeddings, 1-D CNN, LSTM) with a pretrained DistilBERT.
2. **One small pretraining run in class.** Notebook 6.3 trains a tiny GPT (about 1–3 M parameters) on a Spanish public-domain text with a BPE tokenizer trained in the notebook. It trains for a few minutes live and then loads a longer checkpoint, as SimCLR did in S04 (Kaggle model with an `s06_entrenar_gpt.ipynb` companion, same procedure as `s04_entrenar_simclr.ipynb`).
3. **Scaling is shown, not trained.** Fitting a scaling law needs a sweep over model sizes that does not fit in class. The slides use the published power laws (Kaplan et al., 2020; Hoffmann et al., 2022), redrawn by script. Optional: the companion notebook of decision 2 also trains 3–4 sizes and saves a `resultados_escalado.json`, so one slide shows the students' own small curve next to the published one.
4. **Own training loop, Hugging Face only for weights and tokenizers.** S02–S05 use a plain PyTorch loop; the old repo used `Trainer`. Keeping the course loop makes the three adaptation strategies differ in one line (which parameters have `requires_grad`). `transformers` is used for `AutoModel`/`AutoTokenizer`, `peft` for the one LoRA cell, `open_clip` or `transformers` for CLIP, `chronos-forecasting` for Chronos.
5. **LoRA gets one slide and one cell.** $W = W_0 + BA$ with rank $r$; the cell counts trainable parameters and trains for the same budget as full fine-tuning. No further PEFT methods.
6. **Integrating exercise: fine-tune a language model to answer in a style (adopted).** Adapted from MIT 6.S191 Lab 3. The S06 question becomes: *is it enough to ask a pretrained LLM for the style in the prompt, or does it have to be fine-tuned?* Three conditions on the same test questions: base model, base model with the style requested in the prompt, LoRA. The training loop and the next-token loss are the ones of notebook 6.3; the new pieces are the chat template and the mask that keeps the loss on the answer. Changes from the MIT lab:
   - base model: a plain Transformer decoder that handles Spanish and fits a Kaggle T4 (proposal: Qwen2.5-0.5B-Instruct, to check license and time per step), instead of LFM2-1.2B, a hybrid of short convolutions and attention whose modules do not match the block of Part 2;
   - metric: held-out masked NLL, computed with the loss of V64, as the graded number; the LLM judge (Gemini through OpenRouter) and Opik tracing become one optional cell (NVIDIA NIM free API), since they need accounts and keys;
   - style: in Spanish, not Yoda (a protected character, and English). Proposal: the style of the corpus used in 6.3, so the tiny GPT trained from scratch and the adapted LLM can be read side by side. The styled answers are generated once offline (`course/s06_generar_estilo.ipynb`) and published as a Kaggle dataset, as MIT does with its `text_styles/*.txt`;
   - MLflow and a `responder_raw` function, as in the S02–S05 integrators.

## Notation

Consistent with S01–S05 (φ is the activation, σ only the sigmoid, *L* the loss, *J* the objective; $H_{\text{fut}}$ is the forecast horizon of S05, so $H$ is free for heads).

| Symbol | Meaning |
|---|---|
| $T$ | number of tokens in the sequence (context length) |
| $X\in\mathbb R^{T\times d}$ | token representations, one row per token (row $i$ is $\mathbf x_i$, as in S03) |
| $d$ | model width |
| $W_Q, W_K, W_V$ | projection matrices; $Q = XW_Q$, $K = XW_K$, $V = XW_V$ |
| $d_k$ | width of queries and keys per head |
| $S = QK^\top/\sqrt{d_k}$ | attention scores |
| $A = \operatorname{softmax}(S + M)$ | attention weights, row-wise; $A_{ij}$ is how much token $i$ reads from token $j$ (the $\alpha_{\tau t}$ of S05) |
| $M$ | mask: $0$ where allowed, $-\infty$ where not (causal or padding) |
| $H$ | number of heads |
| $\mathbf p_i$ | positional embedding of position $i$ |
| $W_O$ | output projection after concatenating heads |
| $Z\in\mathbb R^{T\times |V|}$ | logits of a language model, one row per position; $\mathbf z_t$ predicts token $w_{t+1}$ |
| $L_t$ | next-token loss at position $t$, $-\log\operatorname{softmax}(\mathbf z_t)_{w_{t+1}}$ |
| $m_t$ | loss mask: $1$ if token $w_{t+1}$ belongs to the answer, $0$ otherwise; $J = \sum_t m_t L_t / \sum_t m_t$ |
| $\exp(J)$ | perplexity |
| $N$ | number of parameters |
| $D$ | number of training tokens |
| $C \approx 6ND$ | training compute in FLOPs |
| $W_0, B, A_r, r$ | LoRA: frozen weight, low-rank factors $B\in\mathbb R^{d\times r}$, $A_r\in\mathbb R^{r\times d}$ (subscript to avoid clashing with the attention matrix $A$) |

## Timing

| Time | Block | Slides / notebooks |
|---|---|---|
| 00:00–00:55 | Theory A | Opening, Parts 1–2 (V01–V30) |
| 00:55–01:45 | Practice A | V31 · notebooks 6.1–6.2 |
| 01:45–01:55 | Break | |
| 01:55–02:50 | Theory B | Parts 3–5 (V32–V69) |
| 02:50–03:45 | Practice B | V70 · notebooks 6.3 and 6.5 (6.4 at home) |
| 03:45–04:00 | Closure | V71–V75 |

Theory A is dense but much of it is review: S03 showed the attention formula, S02 the block's parts, S05 the attention weights. Part 1 states what is new (heads, masks, cost) and moves on.

Theory B now has 38 slides (V32–V69) because of the fine-tuning block V61–V66, which prepares the integrating exercise. If it runs long, V50 (time series and tables) and V53 (cost of size) can be cut to one line each, and V67 (comparing fairly) can move to the notebook.

## Deck structure

75 slides (S02 78, S03 75, S04 66, S05 64).

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · Self-attention in full | V05–V18 | |
| 2 · The Transformer block | V19–V31 | Practice A (V31): notebooks 6.1–6.2 |
| 3 · Pretraining | V32–V42 | |
| 4 · Scale and foundation models | V43–V53 | |
| 5 · Using a pretrained model | V54–V70 | Practice B (V70): notebooks 6.3–6.5 |
| Closure | V71–V75 | Integrating exercise (V71), exit question (V75) |

Prediction-before-experiment slides: V14, V28, V40, V52, V68.

---

## Opening (V01–V04)

| # | Slide | Content | Visual |
|---|---|---|---|
| V01 | What if we keep the attention and drop the recurrence? | Title, session question (the last slide of S05) | |
| V02 | Roadmap | The five parts and the two practice blocks | |
| V03 | Where S05 left us | Seq2seq with attention: the decoder reads every encoder state directly. But the encoder is still an RNN: step $t$ waits for $t-1$, training does not parallelize along $T$, and the path between distant tokens is still long inside the encoder | |
| V04 | The question for today | If every token can read every other token, do we need the recurrence at all? And once we have such a model, what happens when we train it on a lot of text? | |

## Part 1 · Self-attention in full (V05–V18)

| # | Slide | Content | Visual |
|---|---|---|---|
| V05 | Part 1 · Self-attention | Divider. Recap of S05: attention as a per-step lookup from the decoder into the encoder | |
| V06 | From cross-attention to self-attention | In S05 the decoder state asked and the encoder states answered. Now every token asks every token of the **same** sequence. Output $i$ is a weighted average of all tokens, with weights that depend on content | |
| V07 | Query, key, value | Three roles for each token, three learned projections: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. What I am looking for, what I offer to be found by, what I hand over. A soft dictionary lookup | |
| V08 | Scaled dot-product attention | $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V$. The dot product of S05 V54, now in matrix form: one matrix product gives all $T^2$ scores at once | |
| V09 | Self-attention in motion | **S06-A01** (media) | A01 |
| V10 | Why divide by $\sqrt{d_k}$ | With unit-variance entries, $\mathbf q^\top\mathbf k$ has variance $d_k$. Large scores saturate the softmax: one weight near 1, gradients near 0. The same variance argument as initialization in S02 | |
| V11 | Multi-head attention | $H$ heads of width $d_k = d/H$, each with its own $W_Q, W_K, W_V$; concatenate and project with $W_O$. Same cost as one wide head. Several relations at once, like several kernels in a conv layer (S03) | |
| V12 | *(figure)* | **S06-F01** | F01 |
| V13 | Masks | $A = \operatorname{softmax}(S + M)$ with $M_{ij} = -\infty$ where $i$ must not read $j$. **Padding mask**: ignore `[PAD]` tokens (S05 V39). **Causal mask**: token $i$ reads only $j \le i$, so the model cannot look at the future | |
| V14 | Prediction before experiment | Shuffle the tokens of a sentence and feed them to a self-attention layer **without** positional information. What happens to the outputs? And with the causal mask on? Answer in notebook 6.1 | |
| V15 | Order has to be added | Without position, attention is permutation-equivariant (S03 permutation test, S05 V44 shuffled words). Add $\mathbf p_i$ to each token: learned (S03, ViT) or sinusoidal (Vaswani et al., 2017). Modern LLMs rotate $Q$ and $K$ by position instead (RoPE): one line | |
| V16 | *(figure)* | **S06-F02** | F02 |
| V17 | The cost of attention | $T\times T$ scores per head per layer: time and memory grow as $T^2$. Doubling the context quadruples the attention cost. The price of path length 1. Measured in notebook 6.1 | |
| V18 | Recurrence vs attention | Table: per-layer cost ($T d^2$ vs $T^2 d$), sequential steps ($T$ vs 1), longest path between two tokens ($T$ vs 1). Attention wins on parallelism and path; loses on long $T$ | |

## Part 2 · The Transformer block (V19–V31)

| # | Slide | Content | Visual |
|---|---|---|---|
| V19 | Part 2 · The Transformer block | Divider. Recap: attention mixes tokens; by itself it is a weighted average, close to linear in $V$ | |
| V20 | Two sublayers | $X \leftarrow X + \operatorname{MHA}(\operatorname{LN}(X))$, then $X \leftarrow X + \operatorname{MLP}(\operatorname{LN}(X))$. Attention mixes **across** tokens; the MLP transforms **each** token on its own. Everything else is from S02 | |
| V21 | Pre-norm and the clean residual path | LayerNorm (S02: normalizes each token, not the batch) before each sublayer. The identity path stays clean from input to output, as promised in S02. Post-norm (the original paper) needs warmup to train deep | |
| V22 | The MLP and GELU | $\operatorname{MLP}(\mathbf x) = W_2\,\varphi(W_1\mathbf x)$ with $W_1\in\mathbb R^{4d\times d}$ and φ = GELU (S02). Two thirds of the parameters of a block are here | |
| V23 | *(figure)* | **S06-F03** | F03 |
| V24 | The block in motion | **S06-A02** (media) | A02 |
| V25 | Counting parameters | Per block: $4d^2$ (attention) $+ 8d^2$ (MLP) $= 12d^2$. Embedding: $|V|\cdot d$. GPT-2 small: $d = 768$, 12 blocks → about 85 M in blocks + 38 M embeddings. Computed in notebook 6.2 and checked with `sum(p.numel())` | |
| V26 | Three architectures | **Encoder** (no mask; BERT, ViT): understand a whole input. **Decoder** (causal mask; GPT): generate. **Encoder-decoder** (cross-attention; original Transformer, T5): input → output sequence, the S05 seq2seq without recurrence | |
| V27 | *(figure)* | **S06-F04** | F04 |
| V28 | Prediction before experiment | The dates task of S05: a 2-layer Transformer encoder-decoder with $d = 64$ against the S05 RNN with attention. Which trains faster per epoch? Which reaches 100 % first? And without positional embeddings? (notebook 6.2) | |
| V29 | Training a Transformer | AdamW, warmup then cosine decay (S02), dropout on attention weights and residuals, gradient clipping (S05). Warmup matters more here: early Adam steps with large updates can push attention into saturation | |
| V30 | What the Transformer changed | No recurrence: all positions in parallel during training. Path length 1 between any two tokens. One block type for text, images (S03 ViT), audio, time series. What it costs: $T^2$ and no built-in notion of order or locality | |
| V31 | Practice A — Attention and the block | Notebooks 6.1 and 6.2. What to compare, how long, what to write down | |

## Part 3 · Pretraining (V32–V42)

| # | Slide | Content | Visual |
|---|---|---|---|
| V32 | Part 3 · Pretraining | Divider. Recap: a block that scales well and fits any sequence. Recap of S04: unlabeled data can supervise itself | |
| V33 | Subword tokens | Words: huge vocabulary, unknown words. Characters: long sequences, and $T^2$. **BPE**: start from bytes, merge the most frequent pair, repeat until $|V|$ merges. Frequent words are one token; rare words split into pieces. Trained in notebook 6.3 | |
| V34 | *(figure)* | **S06-F05** | F05 |
| V35 | Next-token prediction | $J(\theta) = -\sum_t \log p_\theta(w_t \mid w_{<t})$: a cross-entropy over the vocabulary at every position (S01). The labels are the text itself shifted by one: self-supervision (S04) | |
| V36 | Masked prediction for text | BERT: mask 15 % of the tokens, predict them from both sides (S04 V34: images need 75 %, text 15 %). An encoder, not a generator. Good representations for classification; cannot generate left to right | |
| V37 | The causal mask makes training parallel | One forward pass of a $T$-token sequence gives $T$ predictions and $T$ losses at once, each seeing only its past. Training is parallel; generation is still one token at a time | |
| V38 | Generating text | **S06-A03** (media) | A03 |
| V39 | Sampling | Greedy (argmax), temperature $\tau$ on the logits (S04 temperature, same role), top-$k$ / top-$p$. Low τ: repetitive; high τ: incoherent | |
| V40 | Prediction before experiment | Tiny GPT trained on a Spanish novel. After 200 steps, after 2 000 steps, after the long checkpoint: what does a sample look like at each stage? Letters, words, syntax, meaning: which appears first? (notebook 6.3) | |
| V41 | *(figure)* | **S06-F06** | F06 |
| V42 | What pretraining learns | To predict the next token well, the model has to learn spelling, grammar, facts and some reasoning patterns present in the text. Nobody labels any of it. GPT-2 (2019): one pretrained model, many tasks without task-specific training | |

## Part 4 · Scale and foundation models (V43–V53)

| # | Slide | Content | Visual |
|---|---|---|---|
| V43 | Part 4 · Scale and foundation models | Divider. Recap: next-token prediction turns any text into supervision | |
| V44 | Scaling laws | Test loss falls as a power law in parameters $N$, data $D$ and compute $C$, over many orders of magnitude (Kaplan et al., 2020). Bigger is predictably better, not suddenly better | |
| V45 | *(figure)* | **S06-F07** | F07 |
| V46 | Compute-optimal training | $C \approx 6ND$. For a fixed $C$, grow $N$ and $D$ together: about 20 tokens per parameter (Hoffmann et al., 2022, Chinchilla). Many earlier large models were undertrained | |
| V47 | What is a foundation model? | Trained on broad data at scale with self-supervision, then adapted to many downstream tasks (Bommasani et al., 2021). A training and use pattern, not an architecture: a Transformer is not a foundation model by itself | |
| V48 | Beyond text: vision | CLIP (S04 V51): image and caption as a positive pair, 400 M pairs; **zero-shot classification** by comparing an image with the texts "a photo of a {class}". DINO and MAE (S04): vision backbones without labels | |
| V49 | *(figure)* | **S06-F08** | F08 |
| V50 | Beyond text: time series and tables | **Chronos**: scale and quantize a series into tokens, train a language model on many series, forecast by sampling. **TabPFN**: a Transformer pretrained on synthetic tables that predicts for a new small table in one forward pass. Both zero-shot | |
| V51 | Zero-shot is a claim to check | A foundation model may never have seen data like yours: satellite bands, a hospital's bed occupancy, a factory's sensors. The baseline still has to be beaten (S05 commonsense baseline) | |
| V52 | Prediction before experiment | EuroSAT: CLIP zero-shot vs CLIP features + linear probe vs the S04 SimCLR probe at 10 % of labels. Bike Sharing: Chronos zero-shot vs the S05 LSTM at 24 h. Which comes out ahead in each case? (notebook 6.4) | |
| V53 | The cost of size | Training compute only the largest labs can pay; inference memory: parameters × bytes (a 7 B model in 16-bit is 14 GB). Most practitioners use foundation models, they do not train them. Bridge to Part 5 | |

## Part 5 · Using a pretrained model (V54–V70)

| # | Slide | Content | Visual |
|---|---|---|---|
| V54 | Part 5 · Using a pretrained model | Divider. Recap: pretrained models exist for text, images, series and tables; the question is how to use one | |
| V55 | Four ways to use it | Zero-shot (no training), frozen features + probe (train the head), fine-tuning (train everything), PEFT (train a few added parameters). The three transfer strategies of S03 V47 and S04 V53, plus one | |
| V56 | *(figure)* | **S06-F09** | F09 |
| V57 | Frozen features | Run the encoder once, cache $\mathbf h$ (the `[CLS]` token or the mean of the tokens), train a linear layer in seconds. Cheap, robust with few labels; limited by what pretraining put in $\mathbf h$ | |
| V58 | Fine-tuning | All weights trainable, small learning rate ($10^{-5}$ to $5\cdot10^{-5}$), warmup, few epochs (the S03 fine-tuning recipe). Best when there are enough labels or the domain is far; costs memory for gradients and optimizer state of every weight | |
| V59 | LoRA, in one slide | Freeze $W_0$, learn $\Delta W = BA_r$ with rank $r \ll d$: $2dr$ trainable values instead of $d^2$. For $d = 768$, $r = 8$: 2 % of the matrix. Merge after training, so inference costs nothing extra (Hu et al., 2021) | |
| V60 | *(figure)* | **S06-F10** | F10 |
| V61 | From a language model to an assistant | A pretrained LM continues text; it does not answer. Three stages: pretraining (Part 3), **supervised fine-tuning** on (question, answer) pairs, preference tuning (RLHF, DPO: one line). The **chat template** turns a pair into one token sequence with role markers. Fine-tuning a style is the SFT stage in miniature | |
| V62 | The next-token loss, token by token | Logits $Z\in\mathbb R^{T\times|V|}$. Position $t$ predicts token $t+1$: compare `logits[:, :-1]` with `tokens[:, 1:]`. $L_t = -\log \operatorname{softmax}(\mathbf z_t)_{w_{t+1}}$, a cross-entropy over the vocabulary (S01). Worked example with 6 tokens: target, $p(\text{target})$, $L_t$ | |
| V63 | The loss in motion | **S06-A04** (media) | A04 |
| V64 | Loss only on the answer | Mask $m_t = 1$ if token $t+1$ belongs to the answer, $0$ for template and question: $J = \sum_t m_t L_t \big/ \sum_t m_t$. Without the mask the model also learns to predict the user's question, and the template tokens (nearly free to predict) dilute the gradient. Code: `F.cross_entropy(..., reduction="none")`, then `loss[mask].mean()` | |
| V65 | *(figure)* | **S06-F12** | F12 |
| V66 | Measuring a style | Held-out NLL: the same masked $J$ on test pairs, without gradients. $\exp(J)$ is the perplexity: "among how many tokens is the model choosing". Compare base, base with the style asked in the prompt, and LoRA, with the **same tokenizer** (NLL across tokenizers is not comparable). Optional: an LLM judge, which is cheaper to read and harder to trust | |
| V67 | Comparing fairly | Same data split, same number of steps or the same compute, several seeds where cheap. Report trainable parameters, training time and memory next to accuracy. Data contract for text: tokenizer and maximum length are part of the model | |
| V68 | Prediction before experiment | IMDB with 2 000 training reviews: S05 LSTM, frozen DistilBERT + probe, full fine-tuning, LoRA. Rank them by accuracy. Rank them by trainable parameters. Do the two rankings agree? (notebook 6.5) | |
| V69 | When to use what | Few labels and a domain close to pretraining → frozen features. Enough labels or a distant domain → fine-tuning. Limited memory or many tasks → LoRA. No labels at all → zero-shot, checked against a baseline | |
| V70 | Practice B — Pretraining and adaptation | Notebooks 6.3 and 6.5 in class; 6.4 at home | |

## Closure (V71–V75)

| # | Slide | Content | Visual |
|---|---|---|---|
| V71 | Integrating exercise — Is a foundation model worth it? | See decision 6. Four TODOs, as in S02–S05 | |
| V72 | The course in six questions | learn → train → structure → represent → remember → pretrain / adapt. One line per session: the idea each one added to the loop $x \rightarrow f_\theta(x) \rightarrow L \rightarrow \nabla_\theta L \rightarrow \theta'$ | |
| V73 | What each choice assumes | Format of S03 V69, S04 V64, S05 V62: "assumes" and "pays with". Attention assumes any token may matter (pays with $T^2$); pretraining assumes the pretraining data resembles yours; LoRA assumes the update is low rank | |
| V74 | *(figure)* | **S06-F11** | F11 |
| **S06-F12** | V65 | One chat example as tokens, colored by role (template, question, answer); bars with the per-token loss $L_t$ of the base model; masked positions greyed; the masked mean $J$ against the unmasked mean | `s06_f12_masked_loss.png` | Script (notebook 6.6, base model, before training) |
| V75 | Exit question | A clinic has 800 labeled Spanish clinical notes (3 classes) and one 16 GB GPU. Choose two and justify: an LSTM trained from scratch; a multilingual encoder frozen + probe; full fine-tuning of the same encoder; LoRA on a 7 B decoder; ask a chat LLM zero-shot and compare with a baseline on a held-out set | |

---

## Figures

Same routes as S03–S05: **Generator** (prompts in `figure_prompts.md`), **Script** (`make_s06_*.py`, fixed seeds, numbers taken from the notebooks) and **Manim** (`escenas_s06.py`). All figures drawn from scratch.

| ID | Slide | Purpose | Planned filename | Route |
|---|---:|---|---|---|
| **S06-F01** | V12 | Multi-head attention: $X$ → $H$ heads with their own $Q, K, V$ → concatenate → $W_O$; shapes $(T, d) \to H\times(T, d_k) \to (T, d)$ | `s06_f01_multihead.png` | Generator |
| **S06-F02** | V16 | Attention matrices of one sentence: no mask, padding mask, causal mask; plus outputs before and after shuffling the tokens without position | `s06_f02_masks_permutation.png` | Script (notebook 6.1) |
| **S06-F03** | V23 | The pre-norm Transformer block with the residual path drawn as the straight line of S02 | `s06_f03_block.png` | Generator |
| **S06-F04** | V27 | Encoder, decoder, encoder-decoder side by side, with the mask each one uses | `s06_f04_three_architectures.png` | Generator |
| **S06-F05** | V34 | BPE on one Spanish sentence: characters → merges → tokens; vocabulary sizes 256 / 2 000 / 8 000 vs sequence length | `s06_f05_bpe.png` | Script (notebook 6.3) |
| **S06-F06** | V41 | Training loss of the tiny GPT with three samples printed at their steps | `s06_f06_gpt_samples.png` | Script (notebook 6.3) |
| **S06-F07** | V45 | Test loss vs compute on log-log axes from the published power-law fits; optional points from `resultados_escalado.json` | `s06_f07_scaling.png` | Script |
| **S06-F08** | V49 | CLIP zero-shot: one EuroSAT image against the text embeddings of the 10 class prompts, similarities as bars | `s06_f08_clip_zeroshot.png` | Script (notebook 6.4) |
| **S06-F09** | V56 | Four ways to use a pretrained model: which parameters are frozen, trained or added | `s06_f09_four_ways.png` | Generator |
| **S06-F10** | V60 | LoRA: frozen $W_0$ beside $B$ and $A_r$, with the parameter counts | `s06_f10_lora.png` | Generator |
| **S06-F11** | V74 | Summary: accuracy vs trainable parameters on IMDB for the S05 models and the four strategies of 6.5 | `s06_f11_summary.png` | Script (notebook 6.5: `resultados_adaptacion.json`) |

## Animations (Manim)

Only where motion changes understanding. Conventions of `escenas_s02.py`–`escenas_s05.py`: slide palette, on-screen text in English, `self.pausa()` before each reveal, poster from the last frame.

| ID | Slide | What it shows | Why animate | Rendered file |
|---|---:|---|---|---|
| **S06-A01** | V09 | A 5-token sentence. Each token emits $\mathbf q, \mathbf k, \mathbf v$; token 3 compares its $\mathbf q$ with every $\mathbf k$, scores become a softmax row, the row averages the $\mathbf v$'s; then all rows at once as $QK^\top$, and the causal mask greys out the upper triangle | Moves from one token's lookup (S05-A03) to the whole matrix in one product | `s06_a01_self_attention.mp4` |
| **S06-A02** | V24 | Tokens flow through one block: LN → attention (tokens exchange information, arrows between columns) → add; LN → MLP (each column on its own) → add. The residual stream stays as a straight band | Separates "across tokens" from "per token" | `s06_a02_block.mp4` |
| **S06-A03** | V38 | Autoregressive generation: the prompt goes in, the last position's logits become a distribution over the vocabulary, a token is sampled, appended, and the window shifts; then the same with low and high temperature | Generation is a loop over time; training is not | `s06_a03_generation.mp4` |
| **S06-A04** | V63 | A short chat example as a row of tokens. The model outputs one distribution per position; each is shifted one step to face the next token; the probability of the true token is picked and becomes $L_t$ (a bar). Then the mask turns off the template and question bars, and the average is taken over the answer only. Last, the same sequence after LoRA: the answer bars shrink | Shows where the shift, the cross-entropy and the mask act, which the code hides in two lines | `s06_a04_masked_loss.mp4` |

## Notebooks

Short, self-contained, one change per cell, Chollet DLwP style in PyTorch, comments that link code and slide notation, open-in badges for Colab, Kaggle and Lightning.

| # | Notebook | Question | Data | Time |
|---|---|---|---|---|
| 6.1 | `sesion_06_1_autoatencion.ipynb` | Self-attention as five lines of tensor code, checked against `F.scaled_dot_product_attention`; the effect of $\sqrt{d_k}$ on softmax entropy; heads by reshaping; padding and causal masks; permutation with and without position (F02); time and memory vs $T$ | synthetic and a few sentences | CPU, ~8 min |
| 6.2 | `sesion_06_2_bloque_transformer.ipynb` | Build the pre-norm block; count parameters against $12d^2$; encoder-decoder Transformer on the dates task of 5.5 (same data generator, same metrics); against the S05 result; without positional embeddings; post-norm without warmup at 6 layers | synthetic (generator of 5.5) | CPU/GPU, ~10 min |
| 6.3 | `sesion_06_3_preentrenamiento_gpt.ipynb` | Train a BPE tokenizer (`tokenizers`) and compare vocabulary sizes (F05); tiny GPT with the block of 6.2 and the causal mask; train a few minutes live, then load the long checkpoint; samples at each stage (F06); greedy vs temperature vs top-$k$ | Spanish public-domain novel | GPU, ~15 min |
| 6.4 | `sesion_06_4_modelos_fundacionales.ipynb` | CLIP zero-shot on EuroSAT with prompt templates (F08); CLIP image features + probe at 1 % and 10 % of the labels next to the S04 SimCLR and S03 ImageNet numbers; Chronos zero-shot on Bike Sharing at 1, 6, 12, 24 h against the commonsense baseline and the S05 LSTM | EuroSAT, Bike Sharing | GPU, ~15 min |
| 6.5 | `sesion_06_5_adaptacion.ipynb` | DistilBERT on a 2 000-review IMDB subset, `max_len = 256`: frozen + probe on cached features, full fine-tuning, LoRA with `peft` ($r = 8$); same steps for all three; accuracy, trainable parameters, time and peak memory; table with the S05 5.4 models (F11) | IMDB | GPU, ~15 min |
| 6.6 | `sesion_06_6_integrador_estilo_llm.ipynb` | Integrating exercise (decision 6): does a pretrained LLM need fine-tuning to answer in a style, or is the prompt enough? With MLflow; F12 from its first cells | styled Q&A pairs (Kaggle dataset, decision 6) | GPU (T4), ~25 min |

Proposed TODOs for 6.6, in the S03–S05 style. Sections marked "(listo)": loading the model and tokenizer, generating answers with `chat`, the LoRA wrapper, the training loop (copied from 6.3).

1. **Template and mask** · 10 min: apply the chat template to a (question, answer) pair and build $m_t$ from the position where the answer starts; check it by decoding only the tokens with $m_t = 1$;
2. **The masked loss** · 5 min: complete `perdida_respuesta(logits, tokens, mask)` from V62 and V64 (shift by one, `reduction="none"`, masked mean); the notebook checks it against the unmasked loss of `model(..., labels=...)` on an all-answer sequence;
3. **Train and compare** · 15 min: LoRA run of fixed steps logged to MLflow (rank, trainable parameters, loss curve); held-out $J$ and perplexity for base, base + style in the prompt, LoRA; three generated answers per condition;
4. **`responder_raw`** · 5 min: loads the LoRA adapter from MLflow, applies the template and the contract (non-empty question, maximum length in tokens) and returns the answer.

**Predice** cells before TODO 3: will the held-out $J$ of the base model with the style in the prompt be closer to the base model or to LoRA?

Data loading follows the S03–S05 pattern: Kaggle dataset with *Add Input*, `kagglehub` with `KAGGLE_API_TOKEN` on Colab, local `data/` on Lightning. Pretrained weights (DistilBERT, CLIP, Chronos, the 6.6 base model) come from the Hugging Face Hub; on Kaggle that needs *Internet on*, which must be stated in the README. Kaggle handles to choose for the novel, IMDB and the styled Q&A pairs of 6.6.

---

## Reusable material from jdmartinev/si7011-deeplearning

| Material | Path | Use in S06 | Changes needed |
|---|---|---|---|
| Transformer lecture | `Lecture06/Clase06b - Transformers.pdf` (37 pp.) | Narrative order for Parts 1–3 and V55–V59: attention from translation (pp. 7–8), self-attention without and with learned weights (pp. 9–14), QKV as retrieval (p. 12), multi-head as several conv filters (pp. 15–19), masked attention (pp. 20–25), positional encoding (p. 26), GPT (pp. 27–28), BERT (pp. 29–32), feature-based vs fine-tuning (p. 33), PEFT and LoRA (pp. 34–37) | Text only; Spanish → English deck. Diagrams come from external material (the original paper, blog figures): redrawn in F01, F03, F04, F09, F10 and A01. Add what is not there: $\sqrt{d_k}$, cost, pre-norm, parameter count, BPE, scaling, non-text foundation models |
| IMDB DistilBERT notebooks | `Lecture06/notebooks/rnns/part-1-data` … `part-5-comparison` | Base for 6.5: tokenization and saving the tokenized dataset (part 1), freezing the encoder (part 3), `LoraConfig` and module inspection (part 4), comparison table, confusion matrices and examples where the models disagree (part 5) | Five notebooks with the same loading and `Trainer` code → one notebook, course loop, one change per section. Each part pushes to and reloads from the Hub: keep models in memory. Full IMDB at `max_len` 512 does not fit the class time: subset and 256 tokens. Add trainable parameters, time and peak memory to the table. Folder name `rnns` is misleading |
| TweetEval exercise | `Lecture06/notebooks/excercise/tweeteval-part-1-data` … `part-5-lora` | **Not used** now that the integrating exercise is the LLM one. Its tokenizer comparison (TODO 2.1) can become a *Tu turno* in 6.3 | |
| Deployment notebooks | `Lecture06/notebooks/*/part-6-deployment.ipynb`, `client.py` | **Not used.** Serving is no longer covered | |
| Hugging Face tutorial | `Lecture06/hf_transformers_tutorial.pdf` | Optional student reference for `AutoModel` / `AutoTokenizer` | Link only |

Not in the old repo: self-attention from scratch, cost measurement, the block and parameter count, Transformer on the dates task, BPE training, a GPT trained from scratch, scaling, CLIP, Chronos. Notebooks 6.1–6.4 are new.

## Reusable material from MIT 6.S191 Lab 3

`lab3/LLM_Finetuning.ipynb`, its solution and `mitdeeplearning/lab3.py` in [MITDeepLearning/introtodeeplearning](https://github.com/MITDeepLearning/introtodeeplearning) (MIT License: keep the copyright notice in the adapted notebook). Data: 2 048 questions of `databricks/databricks-dolly-15k` (CC BY-SA 3.0) with answers rewritten in a style, stored as `data/text_styles/{style}.txt`; the next 2 048 questions are the test set.

| Piece | Use in 6.6 | Changes needed |
|---|---|---|
| Templating and tokenization (§1.1) | Template with and without answer; inspecting BPE tokens of a real LLM (link to V33) | Use the tokenizer's own `apply_chat_template` instead of a hand-written string |
| `chat` with sampling (§1.3.1) | "(listo)" helper, temperature as in V39 | None |
| LoRA with `peft` (§1.3.2) | "(listo)", plus the trainable-parameter count against $2dr$ (V59) | `target_modules` for the chosen model |
| `forward_and_compute_loss` (§1.3.3) | Becomes TODO 2 | Shift by one made explicit (the lab hides it) and notation of V62–V64 |
| Training loop (§1.3.4): Lion, 200 steps, batch 1, context 512 | Replaced by the loop of 6.3 with AdamW | Batch > 1 with padding mask, MLflow logging |
| LLM judge and Opik (§2.2–2.5) | One optional cell | NIM instead of OpenRouter; no Opik |
| Final log-likelihood on a fixed text (§2.6) | Generalized to held-out $J$ on the test pairs | Same metric for the three conditions |

## Link to Evaluation Event 3 (S05–S06)

Concepts S06 adds, phrased as decisions to justify:

- explaining from an attention matrix which mask was used, and why a model without position cannot tell two orderings apart;
- estimating the parameters and the attention cost of a configuration, and the effect of doubling $T$ or $d$;
- choosing between zero-shot, frozen features, fine-tuning and LoRA from the number of labels, the domain and the memory available, and backing the choice with a baseline;
- reading a comparison table where the most accurate strategy is not the cheapest, and deciding which to use;
- explaining which tokens a fine-tuning loss counts and why, and why NLL from two different tokenizers cannot be compared.

## Open questions for the instructor

1. **Style and base model for 6.6.** Which style in Spanish (proposal: the 6.3 corpus), and which base model after measuring time per step on a T4 (proposal: Qwen2.5-0.5B-Instruct).
2. **Event 3 Part B.** Event 1 used the S02 integrator. Does Event 3 use the S05 integrator, the S06 one, or one combined deliverable?
3. **Corpus for 6.3.** A Spanish novel from Project Gutenberg (*Don Quijote*, about 2 M characters) or a Colombian public-domain text? Needs a Kaggle dataset handle.
4. **6.4 in class or at home.** The proposal keeps 6.3 and 6.5 in class. If 6.4 is in class instead of 6.3, students see foundation models used but no pretraining run.
5. **Project.** `evaluation_strategy.md` still asks the project for an "inference pipeline" and "deployment or executable demonstration" in S06. With deployment out of the course, should that become the `*_raw` function pattern the integrators already use?
