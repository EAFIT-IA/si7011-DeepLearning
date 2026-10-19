# SI7011 — Session 05

## Figure and animation inventory

Session 05 focuses on the question:

> **How does a network remember the past?**

Slide numbers are those of the current deck source, [`../slides/S05_slides.marp.md`](../slides/S05_slides.marp.md), which has 65 slides: the 64 of [`S05_plan.md`](../../../course/plans/S05_plan.md) plus F11 after the Part 4 divider (so slides from V36 on are one number higher than in the plan). Pending figures and animations are shown in the deck as dashed placeholder boxes with their ID and planned filename. To integrate a figure, replace the placeholder `<div class="placeholder">…</div>` with:

```markdown
<!-- _class: figure -->
<!-- _paginate: false -->
<!-- _footer: "" -->

![bg contain <alt text>](../figures/s05_fnn_<name>.png)
```

Animations use the `media` class with a poster image and a print poster, as in Sessions 02–04.

Figures keep white backgrounds, academic visual style, preserved proportions and lowercase filenames with underscores. All figures are drawn from scratch; none reproduces a figure from a paper, a blog or another course (in particular, not the `Figs/RNN*.PNG` of the previous course, the classic LSTM diagram with yellow and pink blocks, or the seq2seq and attention figures of the original papers).

## Production routes

- **Generator** — conceptual diagrams with exact labels: prompts in [figure_prompts.md](figure_prompts.md).
- **Script** — figures that show real data or a trained model: a `make_s05_*.py` script in this folder (matplotlib, fixed seeds), reading the JSON or `.npz` that the notebook saves, so the slide and the notebook always agree.
- **Manim** — animations in `escenas_s05.py`, with the conventions of `escenas_s02.py`–`escenas_s04.py`.

## Figures

| ID | Slide | Purpose | Filename | Route | Status |
|---|---:|---|---|---|---|
| **S05-F01** | V08 | Four task types on sequences | `s05_f01_sequence_tasks.png` | Generator | Included |
| **S05-F02** | V10 | Jena: temperature series, three windows with their 24 h targets, train/val/test blocks, no window crossing a cut | `s05_f02_windows_split.png` | Script `make_s05_figuras_notebooks.py` (notebook 5.1, section 3: `particion_jena.json`) | Included |
| **S05-F03** | V20 | $\lVert\partial L/\partial\mathbf x_t\rVert$ against $t$ for the trained RNN, log scale, mean over a validation batch | `s05_f03_rnn_gradient_time.png` | Script `make_s05_figuras_notebooks.py` (notebook 5.2, section 3: `gradiente_tiempo_rnn.npz`) | Included |
| **S05-F04** | V27 | The LSTM cell with the additive memory line | `s05_f04_lstm_cell.png` | Generator + script: `fix_s05_f04_lstm_cell.py` (removes an $\mathbf h_t$ drawn out of the $\mathbf c_t$ line) | Included |
| **S05-F05** | V37 | Text pipeline: text → tokens → ids → padded batch and mask → embeddings | `s05_f05_text_pipeline.png` | Generator | Included |
| **S05-F06** | V41 | Bag of embeddings, 1-D CNN and LSTM over the same sentence | `s05_f06_three_text_models.png` | Script `make_s05_f06_three_text_models.py` (drawn, no data) | Included |
| **S05-F07** | V43 | Six query words and their five nearest neighbors in the learned $E$ | `s05_f07_embedding_neighbors.png` | Script `make_s05_figuras_notebooks.py` (notebook 5.4, section 8: `vecinos_embedding.json`) | Included |
| **S05-F08** | V49 | Seq2seq with the single vector $\mathbf h_T$, on a date | `s05_f08_seq2seq.png` | Generator + script: `fix_s05_f08_seq2seq.py` (one input per decoder step) | Included |
| **S05-F09** | V55 | Attention weights $\alpha_{\tau t}$ for three dates | `s05_f09_attention_alignment.png` | Script `make_s05_figuras_notebooks.py` (notebook 5.5, section 6: `atencion_fechas.npz`) | Included |
| **S05-F10** | V63 | Window, RNN, LSTM and attention: how much of step 1 reaches the prediction | `s05_f10_summary.png` | Generator + script: `fix_s05_f10_summary.py` (removes "T = 8"; RNN ribbon thin at step 1, thick at step 8, as F03) | Included |
| **S05-F11** | V36 | $\lVert\partial L/\partial\mathbf x_t\rVert$ against $t$ for RNN, LSTM and GRU on Jena, log scale; hours where the RNN gradient is exactly 0 shaded | `s05_f11_gradient_rnn_lstm_gru.png` | Script `make_s05_figuras_notebooks.py` (notebook 5.3, section 4: `gradiente_tiempo_lstm_gru.npz`) | Included |

### Script specifications

- **F02.** Jena temperature, full record in light gray, the three blocks (train 50 %, validation 25 %, test 25 %) as colored bands. Three example windows of 120 h drawn as brackets on a zoomed inset with their target 24 h later marked as an orange dot; one forbidden window crossing the train/validation cut drawn in red with a cross. Cut indices from the notebook.
- **F03.** One line, log-scale $y$, $t$ from 1 to 120 on $x$ (hours before the prediction on a secondary axis). Notebook 5.3 saves the same array for LSTM and GRU; if the slide in Part 3 needs the comparison, add a second panel with the three curves on the same axes rather than a new figure.
- **F07.** Grid of 6 rows: the query word, then its 5 neighbors with their cosine similarity, colored by the sign of each word's weight in the final linear layer (positive / negative). Words chosen in the notebook, not by hand afterwards.
- **F09.** Three heatmaps side by side, input words on $x$, output characters on $y$, same color scale, the three sentences fixed in notebook 5.5 (one with "primero", one with the year in words, one with filler after the date). Rebuild after re-running the notebook: `python make_s05_figuras_notebooks.py ../notebooks .`

## Animations (Manim)

Conventions of `escenas_s02.py`–`escenas_s04.py`: slide palette, on-screen text in English, `self.pausa()` before each reveal, poster from the last frame.

| ID | Slide | What it shows | Rendered file | Status |
|---|---:|---|---|---|
| **S05-A01** | V16 | The RNN cell rolled up, then unrolled over 8 steps with the same $W_{hh}$ highlighted at every step; the gradient flows back from step 8 and its arrow thins step by step | `s05_a01_unrolling_bptt.mp4` | Pending |
| **S05-A02** | V29 | A value written into $\mathbf c_t$ at step 2, kept by $\mathbf f_t\approx 1$, read by $\mathbf o_t$ at step 7; the same input in a plain RNN fades | `s05_a02_lstm_memory.mp4` | Pending |
| **S05-A03** | V53 | Decoding a date: scores over the input words, softmax into $\alpha_{\tau t}$, context $\mathbf a_\tau$, emitted character; the weight matrix of F09 fills row by row | `s05_a03_seq2seq_attention.mp4` | Pending |

The A03 weights should come from the notebook 5.5 run (`atencion_fechas.npz`), so the animation and F09 show the same matrix.
