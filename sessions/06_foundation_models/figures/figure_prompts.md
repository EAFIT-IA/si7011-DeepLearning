# S06 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. For each figure, build the full prompt as: **common style block** + the figure's section.
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section. Do not deliver JPG.
4. Before delivering, check the image against the **review checklist** and the figure's own "Check" line. If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the current deck, `../slides/S06_slides.marp.md`. Figures that show real data or a trained model (S06-F02, F05, F06, F07, F08, F11, F12) are **not** in this file: they are produced by scripts, see `figures.md`.

## Status

| Figure | Concept | Slide | Status |
|---|---|---:|---|
| S06-F01 | Multi-head attention with shapes | V12 | pending |
| S06-F03 | The pre-norm Transformer block | V23 | pending |
| S06-F04 | Encoder, decoder, encoder-decoder | V27 | pending |
| S06-F09 | Four ways to use a pretrained model | V56 | pending |
| S06-F10 | LoRA | V60 | pending |

## Common style block (prepend to every prompt)

> Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. **No title at the top: the slide provides the title.** Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.
>
> Notation (use it exactly): token representations $X$, with one **row per token**; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows **bottom to top** inside a Transformer block (input at the bottom, output at the top), and **left to right** everywhere else.
>
> Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are **trained**), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

## Review checklist (every image)

- Opens correctly and is a 16:9 PNG.
- No title inside the image; no production codes such as "S06-F03".
- Every matrix carries its shape label (for example "$T \times d$"); the exact proportions of the rectangles do not matter.
- Attention parts are orange; trained parameters teal; frozen parameters gray with a small padlock.
- No logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, **not** the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids).

---

## S06-F01 · Multi-head attention with shapes (pending, slide 12)

File: `s06_f01_multihead.png`

Left to right, four stages.

1. **Left:** one blue rectangle labeled $X$ with the shape label "$T \times d$" under it.
2. **Center-left:** three arrows fan out from $X$ to three stacked lanes labeled "head 1", "head 2", "head $H$" (with a vertical "⋮" between head 2 and head $H$). In each lane: three small orange rectangles side by side labeled $Q$, $K$, $V$ with the shape "$T \times d_k$" under them, then a small square orange heat map labeled $A$ with shape "$T \times T$", then one small orange rectangle labeled "head output" with shape "$T \times d_k$". Next to the arrow that enters each lane write "$W_Q, W_K, W_V$" in small type, and over the heat map the formula $\operatorname{softmax}\!\big(QK^\top/\sqrt{d_k}\big)$ (only in head 1, to avoid clutter). The three heat maps must show **different** patterns (one diagonal, one with a bright column, one diffuse) so the heads look different.
3. **Center-right:** the three head outputs join into one wide rectangle labeled "concatenate" with shape "$T \times H d_k$".
4. **Right:** an arrow labeled $W_O$ to a blue rectangle labeled "output" with shape "$T \times d$".

Bottom strip, gray, full width: "$d_k = d / H$: same parameters and cost as one head of width $d$".

Check: exactly three lanes drawn (head 1, head 2, head $H$); every rectangle carries its shape label; input and output are both $T \times d$; the three heat maps differ.

---

## S06-F03 · The pre-norm Transformer block (pending, slide 23)

File: `s06_f03_block.png`

One tall block in the center, drawn **bottom to top**, with a thick straight blue vertical band running the full height on its left side: the residual stream. Labels on the band: $X$ at the bottom, "residual stream" along it, "output" at the top.

From the band, two side branches go out to the right and come back, each through these boxes from bottom to top:

1. **Lower branch:** a light gray box "LN" → an orange box "multi-head attention" (small subtitle "mixes tokens") → the branch returns to the band at a circle with a plus sign "+".
2. **Upper branch:** a light gray box "LN" → a teal box "MLP" with a small subtitle "$d \rightarrow 4d \rightarrow d$, GELU" and a second subtitle "each token on its own" → returns to the band at a second "+".

To the right of the block, two short annotations aligned with each branch:
- next to the attention branch: "$X \leftarrow X + \operatorname{MHA}(\operatorname{LN}(X))$";
- next to the MLP branch: "$X \leftarrow X + \operatorname{MLP}(\operatorname{LN}(X))$".

At the far left of the band, a vertical bracket covering the whole block labeled "$\times$ number of blocks" in small type.

Small inset at the bottom right, gray outline, titled "post-norm (original)": a mini version where "LN" sits **after** each "+" on the band itself, so the band is interrupted twice; caption under the inset: "LN on the residual path".

Check: in the main block the blue band is never interrupted (no box sits on it, only the two "+" joins); LN is before attention and before the MLP; attention is orange and the MLP teal; the inset is clearly smaller and grayer than the main block.

---

## S06-F04 · Encoder, decoder, encoder-decoder (pending, slide 27)

File: `s06_f04_three_architectures.png`

Three columns of equal width, each with a teal tag at the top-left, a small diagram and two lines of text below.

1. **Tag "encoder":** a stack of two blue blocks labeled "block"; under them a row of 5 blue token squares labeled "the movie was not bad"; above them a row of 5 output squares. To the right, a small 5 × 5 attention mask grid with **all cells filled orange** and the caption "no mask: every token sees every token". Text below: "understand an input" · "BERT, ViT".
2. **Tag "decoder":** the same two-block stack; under it 5 token squares "the movie was not bad"; above it 5 output squares labeled with the **next** token, shifted by one: "movie", "was", "not", "bad", "!". To the right, a 5 × 5 mask grid where the **lower triangle and the diagonal are orange** and the upper triangle is gray with small red crosses; caption "causal mask: token $i$ sees $j \le i$". Text below: "generate, one token at a time" · "GPT".
3. **Tag "encoder-decoder":** a blue encoder stack on the left reading 5 squares, one word each: "la", "película", "no", "está", "mal"; a teal decoder stack on the right reading 5 squares "<s>", "the", "movie", "is", "not" and producing 5 squares "the", "movie", "is", "not", "bad"; an orange arrow labeled "cross-attention" goes from the top of the encoder stack into the decoder stack. Text below: "input sequence → output sequence" · "original Transformer, T5".

Check: the decoder mask is lower-triangular including the diagonal; the encoder mask is all orange; the decoder outputs are the inputs shifted left by one; cross-attention appears only in column 3; the Spanish words are copied exactly with the accent in "película" and "está".

---

## S06-F09 · Four ways to use a pretrained model (pending, slide 56)

File: `s06_f09_four_ways.png`

Four panels side by side, same layout in each: a tall rounded rectangle for the pretrained network (labeled "pretrained network", divided into four horizontal layers) and a small box on top for the task head. A short label at the top of each panel and two lines of text at the bottom.

Color code, shown once in a legend strip along the bottom of the figure: gray with a padlock = "frozen"; teal = "trained"; orange outline = "added".

1. **"zero-shot":** all four layers gray with padlocks; **no** task head; instead a small speech bubble "a photo of a {class}" feeding a gray "text encoder" box beside the network. Bottom text: "trains nothing" · "only a prompt".
2. **"frozen features + probe":** four layers gray with padlocks; head box teal labeled "linear". Bottom text: "trains the head" · "features computed once".
3. **"fine-tuning":** all four layers teal; head teal. Bottom text: "trains everything" · "small learning rate".
4. **"PEFT (LoRA)":** four layers gray with padlocks, but each layer has a **thin orange-outlined side box** attached labeled "$BA_r$" (label only on the top one); head teal. Bottom text: "trains a few added parameters" · "base weights unchanged".

Above all panels, a thin gray arrow pointing right labeled "more parameters trained", starting over panel 1 and ending over panel 3, with panel 4 placed slightly lower so the arrow does not pass over it.

Check: padlocks only on gray parts; panel 1 has no head; panel 4 has orange-outlined side boxes on every layer; the legend shows exactly three entries.

---

## S06-F10 · LoRA (pending, slide 60)

File: `s06_f10_lora.png`

Left to right.

1. **Left:** a blue input vector labeled $\mathbf x$ (a thin vertical bar, label "$d$" next to it).
2. **Center, two parallel paths from $\mathbf x$ that join at a circle with "+":**
   - **upper path:** a large gray square labeled $W_0$ with a padlock and the shape "$d \times d$"; below it, in small type: "frozen · $d^2$ values".
   - **lower path:** a thin, tall teal rectangle labeled $A_r$ with shape "$r \times d$" (drawn wide and short: height $r$, width $d$), followed by a narrow tall teal rectangle labeled $B$ with shape "$d \times r$" (height $d$, width $r$); between them a tiny vector labeled "$r$"; below the pair, in small type: "trained · $2dr$ values". Over the lower path a small label: "$\Delta W = BA_r$".
3. **Right:** after the "+", a blue output vector labeled $\mathbf y$ with the formula above it: $\mathbf y = W_0\mathbf x + BA_r\mathbf x$.

Bottom strip, two boxes side by side:
- left box, gray: "$d = 768,\ r = 8$: $\ 2dr = 12\,288$ vs $d^2 = 589\,824$ $\;\rightarrow\;$ 2 %";
- right box, teal outline: "after training: $W = W_0 + BA_r$, no extra cost at inference".

Check: $A_r$ is drawn short and wide, $B$ tall and narrow, both much thinner than $W_0$; only $W_0$ has a padlock; the numbers 12 288, 589 824 and 2 % are exactly as written; the letter for the low-rank factor is $A_r$ with the subscript (not plain $A$, which in this deck is the attention matrix).
