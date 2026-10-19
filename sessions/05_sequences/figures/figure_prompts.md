# S05 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. For each figure, build the full prompt as: **common style block** + the figure's section.
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section. Do not deliver JPG.
4. Before delivering, check the image against the **review checklist** and the figure's own "Check" line. If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the current deck, `../slides/S05_slides.marp.md`. Figures that show real data or a trained model (S05-F02, F03, F07, F09, F11, F12, F13), and the drawn F06 and F14, are **not** in this file: they are produced by scripts, see `figures.md`.

## Status

| Figure | Concept | Slide | Status |
|---|---|---:|---|
| S05-F01 | Four task types on sequences | V08 | done |
| S05-F04 | The LSTM cell | V28 | done, patched by script |
| S05-F05 | The text pipeline | V40 | done |
| S05-F06 | What each text model sees | V44 | replaced by script `make_s05_f06_three_text_models.py` |
| S05-F08 | Sequence to sequence with one vector | V52 | done, patched by script |
| S05-F10 | How each model reaches the past | V68 | done, patched by script |
| S05-F15 | The GRU cell | V32 | **pending** |

## Common style block (prepend to every prompt)

> Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. **No title at the top: the slide provides the title.** Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish example sentences given in quotes, which must be copied letter by letter, accents included.
>
> Notation (use it exactly): input at step $t$: $\mathbf x_t$; hidden state $\mathbf h_t$; LSTM cell state $\mathbf c_t$; forget, input and output gates $\mathbf f_t$, $\mathbf i_t$, $\mathbf o_t$; candidate $\tilde{\mathbf c}_t$; element-wise product $\odot$; sequence length $T$; embedding of token $w$: $\mathbf e_w$; embedding matrix $E$; decoder state $\mathbf s_\tau$; attention weights $\alpha_{\tau t}$; context vector $\mathbf a_\tau$; prediction $\hat y$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Time always runs **left to right**.
>
> Palette: blue #1f6fb4 (recurrent cells and encoders: the same blue box for the same weights at every step), teal #167d78 (decoders, gates, output heads), orange #d9631e (memory: the cell state $\mathbf c_t$, the single summary vector, attention), red #c0392b (errors, lost information), gray #8a94a3 (padding, inactive or discarded parts), navy text #14213d.

## Review checklist (every image)

- Opens correctly and is a 16:9 PNG.
- No title inside the image; no production codes such as "S05-F04".
- Time runs left to right; the same recurrent box is drawn identically at every step.
- The cell state $\mathbf c_t$ and the summary vector are orange; the hidden state is written $\mathbf h_t$, never $z$.
- Spanish strings are copied exactly, with accents.
- No logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the well-known LSTM diagram with yellow boxes and pink circles, and not the seq2seq figure of Sutskever et al. or Bahdanau et al.).

---

## S05-F01 · Four task types on sequences (done, slide 8)

File: `s05_f01_sequence_tasks.png`

Four panels in a 2 × 2 grid, each with a teal tag at the top-left corner, a small diagram and one example line below. In every panel the inputs are a row of small light-blue squares at the bottom, the recurrent steps are identical blue rounded boxes in a middle row, connected left to right by short arrows, and the outputs are small teal squares at the top.

1. **Top left, tag "many-to-one":** 5 inputs, 5 blue boxes, **one** output above the **last** box only. Example: "sentiment of a review · temperature in 24 h".
2. **Top right, tag "aligned many-to-many":** 5 inputs, 5 blue boxes, **one output above every box** (5 outputs). Example: "a tag for every word · an anomaly flag for every reading".
3. **Bottom left, tag "sequence-to-sequence":** 4 inputs into 4 blue boxes (the encoder), then an orange arrow, then 3 teal boxes (the decoder) with no inputs below them and **one output above each teal box** (3 outputs). Example: "translation · date in words → ISO date".
4. **Bottom right, tag "one-to-many":** **one** input below the first box only, 4 blue boxes, one output above every box (4 outputs). Example: "generating text from a start".

Check: the counts of inputs and outputs are exactly as listed; the boxes in the middle row look the same in all panels; in panel 3 the decoder boxes are teal and the encoder boxes blue.

---

## S05-F04 · The LSTM cell (done, patched by script, slide 28)

File: `s05_f04_lstm_cell.png`

One LSTM cell drawn as a large rounded rectangle, light-blue fill, in the center of the image. Original design: **do not** reproduce the common diagram with yellow layer boxes and pink pointwise circles.

1. **The memory line:** a thick, perfectly straight **orange horizontal line** crossing the top part of the cell from left to right. It enters at the left edge labeled $\mathbf c_{t-1}$ and leaves at the right edge labeled $\mathbf c_t$. On this line, in order from left to right, exactly two operations: a small circle with $\odot$ and then a small circle with $+$. Above the line, a small orange label: "additive path: $\mathbf c_t = \mathbf f_t \odot \mathbf c_{t-1} + \mathbf i_t \odot \tilde{\mathbf c}_t$".
2. **The gates:** three teal valve icons (a small rounded box with a slider inside), labeled $\mathbf f_t$, $\mathbf i_t$, $\mathbf o_t$, each with the small text "$\sigma(\cdot)$" under its label.
   - $\mathbf f_t$ connects by a thin teal arrow up to the $\odot$ circle on the memory line, with the small label "forget".
   - $\mathbf i_t$ and a fourth small blue box labeled $\tilde{\mathbf c}_t = \tanh(\cdot)$ meet at a small $\odot$ circle, whose result goes up by a thin arrow to the $+$ circle on the memory line, with the small label "write".
   - $\mathbf o_t$ sits at the right; a thin arrow comes down from the memory line after the $+$ through a small box "$\tanh$" and meets $\mathbf o_t$ at a small $\odot$ circle, with the small label "read". The result of that circle is $\mathbf h_t$.
3. **Inputs:** at the bottom left, $\mathbf h_{t-1}$ enters from the left edge and $\mathbf x_t$ enters from below. They join into one small bracket labeled $[\mathbf h_{t-1}, \mathbf x_t]$, which branches by thin gray arrows to $\mathbf f_t$, $\mathbf i_t$, $\tilde{\mathbf c}_t$ and $\mathbf o_t$.
4. **Outputs:** $\mathbf h_t$ leaves at the right edge (lower half) and also goes up out of the top of the cell as a second arrow, both labeled $\mathbf h_t$.

**Below the cell, a gray rounded strip:** "with $\mathbf f_t \approx \mathbf 1$: $\partial \mathbf c_t / \partial \mathbf c_{t-1} \approx I$ — the residual of Session 02, along time".

Check: the orange line is straight, unbroken and carries exactly one $\odot$ and one $+$; there are exactly three teal gates; no $W_{hh}$ multiplication appears on the orange line; $\mathbf h_t$ comes from $\mathbf o_t \odot \tanh(\mathbf c_t)$.

---

## S05-F05 · The text pipeline (done, slide 40)

File: `s05_f05_text_pipeline.png`

Five stages left to right, connected by arrows, each stage with a small gray caption below.

1. **Text:** a light panel with two short reviews written in a typewriter font, one under the other: "a slow but moving film" and "terrible". Caption: "raw text".
2. **Tokens:** the same two reviews split into word chips: [a] [slow] [but] [moving] [film] and [terrible]. Caption: "tokenize".
3. **Ids:** the chips replaced by integers: [7] [431] [22] [1] [58] and [912]. The chip with **1** has a small gray tag "[UNK]" above it (the word "moving" is not in the vocabulary). Caption: "vocabulary from the training set only".
4. **Padded batch:** a 2 × 5 grid of integer cells. Row 1: 7, 431, 22, 1, 58. Row 2: 912, 0, 0, 0, 0, where the four 0 cells are gray and labeled "[PAD]" once with a brace. To the right of the grid, a second small 2 × 5 grid of 1 and 0 values labeled "mask": row 1 all 1; row 2: 1, 0, 0, 0, 0. Caption: "pad to the longest, keep a mask".
5. **Embeddings:** a 3-D block (a stack of 2 slabs, each made of 5 colored columns) labeled $(B, T, d_e) = (2, 5, d_e)$. A small box above the arrow into this stage: "$\mathbf e_w = E[w]$, `nn.Embedding`". The padding columns in the second slab are gray. Caption: "lookup".

Check: the integers match between stages 3 and 4; exactly four padding cells; the mask matches the padding; the dimension label is exactly $(B, T, d_e) = (2, 5, d_e)$.

---

## S05-F06 · What each text model sees (done, slide 44)

File: `s05_f06_three_text_models.png`

Three horizontal rows, one per model, each inside a rounded panel with a tag on the left. In every row the same sentence appears at the bottom as six word chips: [not] [good] [,] [but] [very] [fun], and above each chip a small light-blue column for its embedding $\mathbf e_w$.

1. **Tag "bag of embeddings":** the six embedding columns all go by thin gray arrows into one teal circle labeled "mean", then to a small teal box "linear" and $\hat y$. To the right, a small gray note: "same output for any order of the words".
2. **Tag "1-D CNN":** overlapping brackets of width 3 drawn **directly on top of the embedding columns** (each bracket touches the three columns it covers, and the filter box sits above its bracket) (words 1–3, 2–4, 3–5, 4–6 shown as four brackets), each into a small blue box with the same filter icon, then a teal box "max over time" and $\hat y$. Gray note: "sees order within 3 words".
3. **Tag "LSTM":** six identical blue boxes in a chain left to right, one above each embedding, connected by arrows; the last box outputs $\mathbf h_T$ in orange, then a small teal box "linear" and $\hat y$. Gray note: "sees the whole order, step by step".

Check: the sentence and its punctuation are identical in the three rows; exactly four brackets in the CNN row; exactly six LSTM boxes; only the LSTM row has an orange $\mathbf h_T$.

---

## S05-F08 · Sequence to sequence with one vector (done, patched by script, slide 52)

File: `s05_f08_seq2seq.png`

Left half: the encoder. Right half: the decoder. One orange vector between them.

1. **Encoder:** **five** identical blue boxes left to right, labeled "encoder" with a brace below. Under them, one input word chip each: [diecinueve] [de] [octubre] [de] [2026]. Arrows between the boxes carry $\mathbf h_1, \dots, \mathbf h_5$ (label only the first and the last: $\mathbf h_1$ and $\mathbf h_T$).
2. **The single vector:** from the last encoder box, a thick orange arrow to a tall narrow orange bar labeled $\mathbf h_T = \mathbf s_0$, with the small text "everything the decoder knows". This bar is the visual center of the figure.
3. **Decoder:** ten identical teal boxes left to right, labeled "decoder" with a brace below, starting from the orange bar. Above them, one output character chip each: [2] [0] [2] [6] [-] [1] [0] [-] [1] [9]. Under each decoder box except the first, a thin gray arrow brings the previous output character back as input; under the first, a chip "[START]" (the start token). Label the first decoder state $\mathbf s_1$.
4. **Bottom strip, gray, full width:** "the whole input must fit in one fixed-size vector, whatever its length".

Check: the input is exactly "diecinueve de octubre de 2026" as five chips; the output is exactly 2026-10-19 as ten character chips; there is exactly one orange bar and no other connection from the encoder to the decoder (no attention lines in this figure).

---

## S05-F10 · How each model reaches the past (done, patched by script, slide 68)

File: `s05_f10_summary.png`

Four horizontal rows, one per model, aligned on the same time axis of 8 steps, numbered 1 to 8 (step 1 at the left, step 8 at the right; do not write "$T = 8$": in this deck $T$ is the window length). In every row a small teal square at the far right labeled $\hat y$ is the prediction made at step 8. The question each row answers: how much of **step 1** is still available at the prediction.

1. **Tag "window":** only steps 4–8 drawn as light-blue squares inside a bracket labeled "window of $T$ steps"; steps 1–3 drawn in gray with a small red cross, labeled "outside the window: lost".
2. **Tag "RNN":** 8 identical blue boxes in a chain. An orange ribbon runs along the chain whose thickness is the weight of each step in $\hat y$: a hairline at step 1, growing at every step, thickest at step 8. Small label: "recent steps dominate".
3. **Tag "LSTM":** 8 identical blue boxes in a chain, with an orange straight line above them (the cell state) running from step 1 to step 8 with **constant thickness**, and small teal gate marks on it at each step. Small label: "kept by the gates".
4. **Tag "attention":** 8 light-blue squares for the encoder states; from $\hat y$ (on the right), 8 thin orange curved lines go back directly to each of the 8 states, with different thicknesses (the line to step 1 clearly visible). Small label: "looks back directly, one weight per step".

Bottom strip, gray: "the window forgets, the RNN fades, the LSTM keeps, attention looks back".

Check: all four rows share the same 8-step axis; the RNN ribbon grows monotonically from step 1 to step 8; the LSTM line has constant thickness; the attention row has exactly 8 lines, all ending at $\hat y$.

---

## S05-F15 · The GRU cell (pending, slide 32)

File: `s05_f15_gru_cell.png`

One GRU cell drawn as a large rounded rectangle, light-blue fill, in the center of the image, **in the same visual style as the LSTM cell of this deck** (same valve icons for the gates, same small circles for operations, same orange line). Original design: do not reproduce the common diagram with yellow layer boxes and pink pointwise circles. Gates in this figure: update gate $\mathbf u_t$ and reset gate $\mathbf r_t$; candidate $\tilde{\mathbf h}_t$. There is **no** $\mathbf c_t$ in this figure.

1. **The memory line is the hidden state itself:** a thick, perfectly straight **orange horizontal line** crossing the top part of the cell from left to right. It enters at the left edge labeled $\mathbf h_{t-1}$ and leaves at the right edge labeled $\mathbf h_t$. On this line, in order from left to right, exactly two operations: a small circle with $\odot$ and then a small circle with $+$. Above the line, a small orange label: "additive path: $\mathbf h_t = (1 - \mathbf u_t) \odot \mathbf h_{t-1} + \mathbf u_t \odot \tilde{\mathbf h}_t$".
2. **The gates:** two teal valve icons (a small rounded box with a slider inside), labeled $\mathbf r_t$ (left) and $\mathbf u_t$ (right), each with the small text "$\sigma(\cdot)$" under its label.
   - $\mathbf u_t$ goes up through a small box "$1 - \cdot$" to the $\odot$ circle on the orange line, with the small label "keep".
   - $\mathbf u_t$ also goes to a second small $\odot$ circle, where it meets the candidate box; the result goes up by a thin arrow to the $+$ circle on the orange line, with the small label "write".
3. **The candidate:** a small blue box labeled $\tilde{\mathbf h}_t = \tanh(\cdot)$, between the two gates. Its input is a small bracket labeled $[\mathbf r_t \odot \mathbf h_{t-1}, \mathbf x_t]$: a thin gray arrow comes down from the orange line **before** the first $\odot$ circle into a small $\odot$ circle where it meets $\mathbf r_t$ (small label "reset"), and from there into that bracket.
4. **Inputs:** $\mathbf x_t$ enters from below at the bottom left. A small bracket labeled $[\mathbf h_{t-1}, \mathbf x_t]$ (fed by a thin gray arrow from the orange line and by $\mathbf x_t$) branches by thin gray arrows to $\mathbf r_t$ and $\mathbf u_t$; $\mathbf x_t$ also goes to the bracket of the candidate.
5. **Output:** $\mathbf h_t$ leaves at the right edge on the orange line and also goes up out of the top of the cell as a second arrow labeled $\mathbf h_t$.

**Below the cell, a gray rounded strip:** "two gates, no separate cell: the additive path is $\mathbf h_t$ itself · three weight blocks instead of four, about 3/4 of the LSTM parameters".

Check: the orange line is straight, unbroken and carries exactly one $\odot$ and one $+$; exactly two teal gates, labeled $\mathbf r_t$ and $\mathbf u_t$; no $\mathbf c_t$ and no output gate anywhere; the "$1 - \cdot$" box sits between $\mathbf u_t$ and the $\odot$ on the line; $\mathbf r_t$ acts on $\mathbf h_{t-1}$ only inside the candidate.
