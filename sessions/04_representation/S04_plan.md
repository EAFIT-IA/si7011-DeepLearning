# SI7011 — Session 04 · Representation Learning

## Slide-by-slide plan

> **How can a network learn useful representations without explicit labels?**

Instructor planning document. It fixes the deck structure, the content of each slide, the figures and animations, and the notebooks before anything is built. It follows the same conventions as Sessions 02 and 03: slides in English, notebooks in Spanish, five parts opened by divider slides that recap the previous part, one prediction-before-experiment slide per part, two practice blocks and an exit question.

## Sources

| Source | What it gives this session |
|---|---|
| `course/course_structure.md` (EAFIT-IA) | Topic list for S04: encoders and decoders, autoencoders, denoising, masked prediction, embeddings, contrastive learning, InfoNCE, self-supervised learning, linear probing |
| `course/evaluation_strategy.md` (EAFIT-IA) | Event 2 covers S03–S04: "embeddings, reconstruction and self-supervised learning"; students must justify representation, architecture and experimental comparisons |
| S03 deck, branch `s03-esqueleto` | Bridges already in place: V49 says probe accuracy measures representation quality "which we use again in Session 04"; V41 augmentation encodes invariances; V56–V58 patches and tokens; V47 three transfer strategies |
| jdmartinev/si7011-deeplearning, `Lecture03-04` | SimCLR notebooks and the self-supervised section of `3_1_CNNs.pdf` (pp. 238–273). Details in [Reusable material](#reusable-material-from-jdmartinevsi7011-deeplearning) |

## Decisions (taken)

All four proposals below were adopted when the notebooks were built.

1. **Integrating exercise.** Proposal: continue EuroSAT from S03. Pretrain on EuroSAT images without labels, then probe with 1 % and 10 % of the labels and compare with the ImageNet backbone from the S03 exercise. It answers the question S03 left open (does ImageNet help with satellite images?) with a new tool, and it can be the Part B deliverable of Event 2.
2. **SimCLR training time.** In class, notebook 4.4 trains for 3 epochs to see the loss fall, then loads a checkpoint trained beforehand (30 epochs by default, with `course/s04_entrenar_simclr.ipynb`, published as a Kaggle dataset). Training from scratch in class does not fit in Practice B.
3. **Masked prediction notebook (4.3).** A tiny MAE on Fashion-MNIST (7×7 patches, 16 tokens) reusing the small ViT from notebook 3.5. It fits in 10 minutes on GPU, but Practice B is already full, so 4.3 is listed as homework.
4. **VAE and generative models.** Not in the S04 topic list. One line in V26 points out that a VAE is the generative version of the autoencoder; nothing more.

## Notation

Consistent with S01–S03 (φ is the activation, *g* the backbone, *L* the loss, *J* the objective).

| Symbol | Meaning |
|---|---|
| $g$ | encoder (the backbone of S03) |
| $d$ | decoder |
| $\mathbf h = g(\mathbf x)$ | representation, code or embedding, $\mathbf h\in\mathbb R^k$ (S01 already called the hidden representation *h*) |
| $\hat{\mathbf x} = d(\mathbf h)$ | reconstruction |
| $\tilde{\mathbf x}$ | corrupted input (noise or mask) |
| $p$ | projection head |
| $\mathbf u = p(\mathbf h)$ | projected embedding, used only by the contrastive loss |
| $s_{ij}$ | cosine similarity between $\mathbf u_i$ and $\mathbf u_j$ |
| $\tau$ | temperature |
| $N$ | images per batch; $2N$ views |
| $L_i$, $J$ | per-example loss and dataset objective, as in S01 |

*z* is not used for the code, because in S01 it is the pre-activation.

## Timing

| Time | Block | Slides / notebooks |
|---|---|---|
| 00:00–00:55 | Theory A | Opening, Part 1, Part 2 (V01–V26) |
| 00:55–01:45 | Practice A | V27 · notebooks 4.1–4.2 |
| 01:45–01:55 | Break | |
| 01:55–02:50 | Theory B | Parts 3–5 (V28–V60) |
| 02:50–03:45 | Practice B | V61 · notebooks 4.4–4.5 (4.3 as homework) |
| 03:45–04:00 | Closure | V62–V66 |

## Deck structure

66 slides: 51 content slides, 11 figures and 4 animations. S02 has 78 and S03 75.

| Part | Slides | Practice |
|---|---|---|
| Opening | V01–V04 | |
| 1 · What is a representation? | V05–V13 | |
| 2 · Autoencoders | V14–V27 | Practice A (V27): notebooks 4.1–4.2 |
| 3 · Masked prediction | V28–V36 | |
| 4 · Contrastive learning | V37–V51 | |
| 5 · Evaluating representations | V52–V61 | Practice B (V61): notebooks 4.3–4.5 |
| Closure | V62–V66 | Integrating exercise (V62), exit question (V66) |

Prediction-before-experiment slides: V12, V19, V35, V47, V60.

---

## Opening (V01–V04)

| # | Slide | Content | Visual |
|---|---|---|---|
| V01 | How can a network learn useful representations without labels? | Title, session question | |
| V02 | Roadmap | The five parts and the two practice blocks | |
| V03 | From architecture to representation | S03 recap: a pretrained backbone $g$ is reused across tasks. But that backbone needed 1.2 M labeled ImageNet images. What did it actually learn? A representation $\mathbf h=g(\mathbf x)$ | |
| V04 | The question for today | Labels are scarce and expensive (S03 exit question: 300 X-rays). Unlabeled images are almost free. Can the data supervise itself? | |

## Part 1 · What is a representation? (V05–V13)

| # | Slide | Content | Visual |
|---|---|---|---|
| V05 | Part 1 · What is a representation? | Divider. Recap of S03: architecture encodes assumptions about the input | |
| V06 | A representation is a change of coordinates | $\mathbf x\mapsto\mathbf h=g(\mathbf x)$. Link to S01 (E6): the hidden layer bends space until a line separates the classes. A good $\mathbf h$ makes the task easy for a simple model | Still from S01-E6 |
| V07 | What makes a representation useful | Four properties: keeps the information the task needs; discards nuisances (lighting, position); is compact; is linearly decodable. "Useful" always means useful *for some task* | |
| V08 | Embeddings | One vector per item; distance means similarity. `nn.Embedding` is one-hot × matrix (lookup). For images, the embedding is the output of $g$ (512 values for ResNet-18) | |
| V09 | *(figure)* | **S04-F01** | F01 |
| V10 | Labels from the data itself | Pretext task: hide or alter part of $\mathbf x$, predict it from the rest. Two families: **self-prediction** (reconstruct, denoise, fill in) and **contrastive** (decide which inputs belong together). The pretext task is discarded; $g$ is kept | |
| V11 | Measuring a representation | Freeze $g$, then: (1) linear probe, (2) $k$-NN in embedding space, (3) accuracy as a function of the number of labels. Callback to S03 V49 | |
| V12 | Prediction before experiment | A ResNet-18 with **random** weights, frozen, plus a linear probe on CIFAR-10. Chance is 10 %. What accuracy? Answer in notebook 4.5 (expected: well above chance, around 30–40 %) | |
| V13 | Architecture versus objective | S03 chose the architecture (what $g$ can represent). S04 chooses the objective (what $g$ is pushed to represent). Same ResNet, different $J$ → different $\mathbf h$ | |

## Part 2 · Autoencoders (V14–V27)

| # | Slide | Content | Visual |
|---|---|---|---|
| V14 | Part 2 · Autoencoders | Divider. Recap: a representation is judged by what a simple model can do with it | |
| V15 | Encoder and decoder | $\hat{\mathbf x}=d(g(\mathbf x))$, $L_i=\lVert\mathbf x_i-\hat{\mathbf x}_i\rVert^2$. The trap: $g=d=\text{identity}$ gives zero loss and learns nothing | |
| V16 | *(figure)* | **S04-F02** | F02 |
| V17 | The bottleneck forces a choice | Undercomplete: $k\ll D$ (784 → 32). The network must decide what to keep. Compression is the supervision | |
| V18 | A linear autoencoder is PCA | Linear $g,d$ with MSE learn the subspace of the top-$k$ principal components. The only new ingredient of a deep autoencoder is φ | |
| V19 | Prediction before experiment | Fashion-MNIST, $k=2$ and $k=32$. Which reconstructs better? In which latent space are the classes more separated, judged by a linear probe? | |
| V20 | The latent space organizes itself | **S04-A01** (media) | A01 |
| V21 | *(figure)* | **S04-F03** | F03 |
| V22 | Reconstruction is not the goal | Pixel MSE spends capacity on texture and gives blurry outputs. Measured in notebook 4.1 on Fashion-MNIST: with $k=32$ the ReLU autoencoder has 28 % less reconstruction error than PCA, and its probe changes by +0.6 points (0.807 → 0.812). In notebook 4.2 the denoising autoencoder reconstructs clean images worse (0.0114 → 0.0126) and has a better probe (0.813 → 0.819) | |
| V23 | Convolutional autoencoder | Conv encoder from S03; decoder with `ConvTranspose2d` or upsample + conv. Shapes 1×28×28 → 64×7×7 → $k$ → back | |
| V24 | Denoising autoencoder | $\tilde{\mathbf x}=\mathbf x+\boldsymbol\varepsilon$ or masked; $L_i=\lVert\mathbf x_i-d(g(\tilde{\mathbf x}_i))\rVert^2$. Copying is no longer possible, so the bottleneck can be wide | |
| V25 | Denoising pulls points back to the data | **S04-A02** (media) | A02 |
| V26 | Other ways to prevent copying | Sparse codes; noise; and the generative version, the VAE (beyond this course). Common idea: limit the information that passes through $\mathbf h$ | |
| V27 | Practice A — Autoencoders | Notebooks 4.1 and 4.2. What to compare, how long, what to write down | |

## Part 3 · Masked prediction (V28–V36)

| # | Slide | Content | Visual |
|---|---|---|---|
| V28 | Part 3 · Masked prediction | Divider. Recap: denoising predicts the clean input from a corrupted one. Masking is the extreme case of corruption | |
| V29 | From noise to masks | Remove instead of perturb; predict only what was hidden | |
| V30 | Hand-designed pretext tasks | Rotation prediction, jigsaw puzzles, colorization, inpainting. Each one worked a little and each one had shortcuts (for example, chromatic aberration gives away patch position) | |
| V31 | Masked autoencoders | Patches as in S03 V56; mask 75 %; the encoder sees only the visible patches; a small decoder fills in the rest; the loss counts only masked patches | |
| V32 | *(figure)* | **S04-F04** | F04 |
| V33 | Masking in motion | **S04-A03** (media) | A03 |
| V34 | Why mask so much? | Images are redundant: with 25 % masked, neighbors can be interpolated. Text needs only 15 % (BERT, S05). Encoding only 25 % of the patches also cuts compute by about 4× | |
| V35 | Prediction before experiment | Tiny MAE on Fashion-MNIST with 25 % and 75 % masked. Which one gives the better probe? (notebook 4.3) | |
| V36 | What masked prediction learns | Probe below fine-tuning for MAE features: the information is there, but not linearly arranged. Cite the MAE paper (ViT-L on ImageNet: about 10 points between probe and fine-tuning). In notebook 4.3 on Fashion-MNIST the gap is small (100 labels per class: probe 0.76, fine-tuning 0.78), and the notebook says so. Bridge to Part 4 | |

## Part 4 · Contrastive learning (V37–V51)

| # | Slide | Content | Visual |
|---|---|---|---|
| V37 | Part 4 · Contrastive learning | Divider. Recap: self-prediction spends effort on pixels. Alternative: no decoder at all | |
| V38 | Learning relations, not pixels | Similar inputs close, dissimilar inputs far, in embedding space. No reconstruction | |
| V39 | Augmentations define "the same" | Two random views of one image are a positive pair. The augmentations chosen are the invariances learned. Callback to S03 V41 | |
| V40 | *(figure)* | **S04-F05** | F05 |
| V41 | SimCLR | $\mathbf x\to$ two views $\to$ shared $g\to\mathbf h\to p\to\mathbf u$. The other $2N-2$ views in the batch are negatives | |
| V42 | *(figure)* | **S04-F06** | F06 |
| V43 | Similarity and temperature | $s_{ij}=\dfrac{\mathbf u_i^\top\mathbf u_j}{\tau\,\lVert\mathbf u_i\rVert\,\lVert\mathbf u_j\rVert}$. Small τ sharpens the comparison and focuses on the hardest negatives | |
| V44 | InfoNCE is a cross-entropy | $L_i=-\log\dfrac{\exp(s_{i,j(i)})}{\sum_{k\ne i}\exp(s_{ik})}$: a classification over $2N-1$ candidates where the correct "class" is the other view. Same softmax + cross-entropy as S01 | |
| V45 | *(figure)* | **S04-F07** | F07 |
| V46 | Pull and push on the sphere | **S04-A04** (media) | A04 |
| V47 | Prediction before experiment | Train SimCLR without color jitter. What does the network learn to use instead? (Answer: color histograms are enough to match two crops; the probe drops.) Second question: τ = 0.05 vs 1.0 | |
| V48 | Why a projection head? | The loss makes $\mathbf u$ invariant to the augmentations, so $\mathbf u$ throws information away. $\mathbf h$, one layer earlier, keeps more. Downstream we use $\mathbf h$ and discard $p$ | |
| V49 | Negatives and batch size | More negatives, harder task, better features. SimCLR needs large batches; MoCo keeps a queue of past embeddings instead | |
| V50 | Collapse | If every $\mathbf u$ is the same, two views always match. Negatives prevent this. Methods without negatives (BYOL, DINO) prevent it with asymmetry: a second network updated slowly and stop-gradient | |
| V51 | The same idea, other pairs | MoCo, BYOL, DINO in one line each. CLIP: positives are an image and its caption. Bridge to S05 (foundation models) | |

## Part 5 · Evaluating representations (V52–V61)

| # | Slide | Content | Visual |
|---|---|---|---|
| V52 | Part 5 · Evaluating representations | Divider. Recap: three objectives (reconstruct, fill in, contrast), one encoder $g$ | |
| V53 | Three ways to read a representation | $k$-NN (no training), linear probe (one layer), fine-tuning (everything). The same three strategies as S03 V47, now applied to our own $g$ | |
| V54 | $k$-NN in embedding space | Cosine similarity, majority vote of the $k$ nearest training embeddings. It is also retrieval: "show me images like this one" | |
| V55 | *(figure)* | **S04-F08** | F08 |
| V56 | Label efficiency | The question that justifies self-supervision: accuracy as a function of labels per class (1, 10, 100, all), for random $g$, autoencoder, SimCLR and ImageNet | |
| V57 | *(figure)* | **S04-F09** | F09 |
| V58 | Looking at embeddings | t-SNE and UMAP keep neighbors, not distances: cluster sizes and the gaps between them mean nothing. Use them to look, and use the probe to measure | |
| V59 | *(figure)* | **S04-F10** | F10 |
| V60 | Prediction before experiment | 10 labels per class with the SimCLR backbone: $k$-NN, linear probe or fine-tuning? Which wins and why? | |
| V61 | Practice B — Contrastive learning and evaluation | Notebooks 4.4 and 4.5 in class; 4.3 at home | |

## Closure (V62–V66)

| # | Slide | Content | Visual |
|---|---|---|---|
| V62 | Integrating exercise — EuroSAT without labels | See decision 1. Four TODOs, as in S02 and S03 | |
| V63 | Choosing a pretext task | Table: autoencoder / denoising / masked / contrastive × what it learns, what it needs (decoder, augmentations, large batch), what it is good for (probe vs fine-tune) | |
| V64 | What each objective assumes | Table in the format of S03 V69: "assumes" and "pays with". Reconstruction assumes every pixel matters; contrastive assumes the augmentations do not change meaning | |
| V65 | *(figure)* | **S04-F11** | F11 |
| V66 | Exit question | 50,000 unlabeled dermatology images, 200 labeled. Choose two and justify: SimCLR with the default augmentations; MAE; ImageNet backbone + probe; denoising autoencoder; check what color jitter does to a lesion before pretraining | |

---

## Figures

Same production routes as S03: **Generator** (prompts in `figure_prompts.md`), **Script** (`make_s04_fnn_*.py`, fixed seeds, numbers from the notebook so slide and notebook agree) and **Manim** (`escenas_s04.py`). All figures are drawn from scratch.

| ID | Slide | Purpose | Planned filename | Route |
|---|---:|---|---|---|
| **S04-F01** | V09 | Supervised vs self-supervised pipeline: label from an annotator vs label generated from the input; pretext head discarded, $g$ kept | `s04_f01_supervised_vs_selfsupervised.png` | Generator |
| **S04-F02** | V16 | Autoencoder with shapes: 784 → 256 → $k$ → 256 → 784, bottleneck highlighted | `s04_f02_autoencoder.png` | Generator |
| **S04-F03** | V21 | Reconstructions of the same 8 test images for $k$ = 2, 8, 32 and PCA with $k$ = 32 | `s04_f03_reconstructions_k.png` | Script (notebook 4.1) |
| **S04-F04** | V32 | MAE pipeline: patches, 75 % masked, encoder on visible tokens, mask tokens, decoder, loss on masked patches only | `s04_f04_mae_pipeline.png` | Generator |
| **S04-F05** | V40 | Eight CIFAR-10 images, each with two SimCLR views | `s04_f05_augmented_views.png` | Script (notebook 4.4) |
| **S04-F06** | V42 | SimCLR: two views, shared $g$, $\mathbf h$, $p$, $\mathbf u$, similarity; downstream path uses $\mathbf h$ | `s04_f06_simclr.png` | Generator |
| **S04-F07** | V45 | 2N × 2N similarity matrix for N = 8: diagonal masked, positives marked, one row read as softmax | `s04_f07_similarity_matrix.png` | Script (trained checkpoint from 4.4) |
| **S04-F08** | V55 | Retrieval: 5 query images and their 6 nearest neighbors with SimCLR $\mathbf h$ | `s04_f08_knn_retrieval.png` | Script (notebook 4.5) |
| **S04-F09** | V57 | Probe accuracy vs labels per class (1, 10, 100, 5000) for random, AE, SimCLR, ImageNet | `s04_f09_label_efficiency.png` | Script (notebook 4.5: `resultados_eficiencia.json`) |
| **S04-F10** | V59 | t-SNE of CIFAR-10 test embeddings: random $g$, autoencoder, SimCLR | `s04_f10_tsne_embeddings.png` | Script (notebook 4.5) |
| **S04-F11** | V65 | Summary: the four objectives around one encoder $g$ (reconstruct, denoise, fill in, contrast) and what each one keeps | `s04_f11_objectives_summary.png` | Generator |

## Animations (Manim)

Only where motion changes understanding. Conventions of `escenas_s02.py` and `escenas_s03.py`: palette of the slides, on-screen text in English like the deck, `self.pausa()` before each reveal, poster extracted from the last frame.

| ID | Slide | What it shows | Why animate | Rendered file |
|---|---:|---|---|---|
| **S04-A01** | V20 | 2-D latent space of a Fashion-MNIST autoencoder through training (snapshots saved in notebook 4.1): a shapeless cloud separates into groups. Colors by class are added only at the end, with the reminder "labels were never used" | The point is that structure *appears*; a still shows only the result | `s04_a01_latent_space.mp4` |
| **S04-A02** | V25 | Points on a 1-D curve in 2-D; noise pushes them off; the denoising map sends each one back to the curve (arrows), then the learned vector field | Shows the manifold intuition behind denoising | `s04_a02_denoising_manifold.mp4` |
| **S04-A03** | V33 | Image → 16 patches (reuses S03-A04) → 75 % masked → encoder takes only the 4 visible tokens → decoder receives mask tokens → reconstruction, loss highlighted on masked patches | Order of operations and the encoder/decoder asymmetry | `s04_a03_masking.mp4` |
| **S04-A04** | V46 | 2-D circle (unit sphere): 6 images, 2 views each. Gradient steps pull positives together and push negatives apart; then the same scene with τ small vs large (only the nearest negatives move vs all of them) | InfoNCE as forces; temperature as reach | `s04_a04_contrastive_sphere.mp4` |

## Notebooks

Short, self-contained, one change per cell, Chollet DLwP style in PyTorch, with comments that link code to the slide notation. Open-in badges for Colab, Kaggle and Lightning.

| # | Notebook | Question | Data | Time |
|---|---|---|---|---|
| 4.1 | `sesion_04_1_autoencoders.ipynb` | What does a bottleneck keep? $k$ = 2, 8, 32; PCA vs linear AE vs deep AE; probe on $\mathbf h$; saves latent snapshots for A01 and figure F03 | Fashion-MNIST | CPU, ~10 min |
| 4.2 | `sesion_04_2_denoising_conv.ipynb` | Conv AE, then denoising with three noise levels; reconstruction error and probe side by side | Fashion-MNIST | CPU/GPU, ~10 min |
| 4.3 | `sesion_04_3_masked_prediction.ipynb` | Tiny MAE with the small ViT of notebook 3.5; mask ratio 25 / 50 / 75 %; probe and fine-tune (homework) | Fashion-MNIST | GPU, ~15 min |
| 4.4 | `sesion_04_4_simclr.ipynb` | Two views (GPU augmentations), InfoNCE checked against the formula, projection head; 3 epochs live, then the long checkpoint (trained with `course/s04_entrenar_simclr.ipynb`); similarity matrix (F07); ablation without color jitter from the second checkpoint; saves `simclr_g.pt` | CIFAR-10 | GPU, ~15 min |
| 4.5 | `sesion_04_5_evaluar_representaciones.ipynb` | Four ResNet-18 encoders (random, autoencoder trained in the notebook, SimCLR, ImageNet): $k$-NN, retrieval (F08), linear probe, label efficiency at 1/10/100/1000/5000 per class (F09), t-SNE (F10), and with 10 labels per class $k$-NN vs probe vs fine-tuning | CIFAR-10 | GPU, ~15 min |
| 4.6 | `sesion_04_6_integrador_eurosat_ssl.ipynb` | Integrating exercise (decision 1) with MLflow | EuroSAT | GPU |

Proposed TODOs for 4.6, in the S03 style:
1. write the two-view augmentation for satellite images (which augmentations keep the class: rotations yes, color jitter?);
2. complete the InfoNCE loss from the slide formula;
3. evaluate with $k$-NN and probe at 1 % and 10 % of the labels and compare with the ImageNet probe from the S03 exercise;
4. log the encoder to MLflow and write `embed_raw`, which returns $\mathbf h$ for new raw images and rejects those that break the data contract.

---

## Reusable material from jdmartinev/si7011-deeplearning

| Material | Path | Use in S04 | Changes needed |
|---|---|---|---|
| SimCLR pretraining notebook | `Lecture03-04/notebooks/4_self-supervised/part-4-SimCLR.ipynb` | Base for notebook 4.4: `AugmentedImages` (two views), the paired-view grid (→ F05), `info_nce_loss` with its ranking metric (position of the positive among $2N-1$; good for showing the pretext task is learned) | Upscales 32 px to 128 px with `RandomResizedCrop(128)`: 16× more pixels; use 32 px and a CIFAR stem (3×3 conv, no max pool). Blur with kernel 9 makes no sense at 32 px. Validation and test loaders also augment. Separate `g` and `p` into two modules so only $g$ is saved. Adam 5e-4 for 50 epochs without schedule → LARS/AdamW + cosine, as in S02. Rename variables to the slide notation (`h`, `u`, `tau`) |
| SimCLR fine-tuning notebook | `Lecture03-04/notebooks/4_self-supervised/part-5-resnet18-finetune.ipynb` | Base for the fine-tuning section of 4.5 (load the checkpoint, swap the head) | Only fine-tunes all layers with SGD lr 0.1, high for a pretrained backbone. Add frozen probe, $k$-NN, random baseline and label budgets. Same 128 px upscaling |
| Self-supervised section of the CNN lecture | `Lecture03-04/3_1_CNNs.pdf`, pp. 238–273 | Narrative order for Parts 1, 3, 4 and 5: labels generated from the features (p. 239), two families self-prediction / contrastive (p. 243), SimCLR summary and why the extra projection (pp. 257–261), similarity with temperature (pp. 262–267), three options after pretraining (pp. 269–272) | Text only. The diagrams come from external material (the SimCLR paper figure on p. 268, among others); they are redrawn in F01, F04, F06 |
| Tiny ImageNet competition | `Lecture03-04/README.md` | Not used. Possible alternative dataset for the integrating exercise if EuroSAT is not chosen | |
| Frozen-feature extraction for text | `Lecture06/notebooks/rnns/part-3-feature-extraction.ipynb` | Pattern of computing features once and caching them (used in 4.5 to train probes in seconds) | Belongs to S06 for text; take only the caching pattern |

Not in the repo: autoencoders, denoising, masked prediction, $k$-NN evaluation, label-efficiency experiments. Notebooks 4.1–4.3 and most of 4.5 are new.

## Link to Evaluation Event 2 (S03–S04)

Concepts S04 adds to the event, phrased as decisions to justify rather than code to write:

- choosing a pretext task for a domain and justifying the augmentations;
- reading a label-efficiency curve (when does self-supervision pay off?);
- explaining why a probe and fine-tuning can rank two encoders differently (MAE vs SimCLR);
- diagnosing collapse from the training curves (loss stops falling, $\mathbf h$ has near-zero variance).
