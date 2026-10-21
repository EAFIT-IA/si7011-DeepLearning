# S06 — Image-generator prompts for pending figures

## Instructions for the image-generation agent

1. Work on **one figure at a time**, in the order of the status table below. Only generate figures marked **pending**.
2. Copy the **whole code block** of the figure and paste it into the generator as is. Each block already contains the common style, the figure and its check, in plain text (do not copy from the rendered page outside the code blocks: GitHub renders the formulas and they paste broken).
3. Deliver one image per figure: **PNG, 16:9, 1672 × 941 px**, named exactly as the `File:` line of its section. Do not deliver JPG.
4. Before delivering, check the image against the "CHECK" part of its block and the review checklist below. If any label, number or symbol differs from the prompt, regenerate.
5. Do not edit the slides or `figures.md`; the course maintainer integrates each image after review.

Slide numbers refer to the current deck, `../slides/S06_slides.marp.md`. This file holds **only** the figures made with the image generator. Not here: the animations (Manim, `escenas_s06.py`); the figures drawn from the notebooks' results (F06, F10, F13, F14, F15, F18, F19); and the token-column series of Part 1 (F01, F02, F03, F05), drawn by `make_s06_f01_f05_tokens.py` because it needs identical columns and exact numbers. See `figures.md`.

## Status

| Figure | Concept | Slide | Status |
|---|---|---:|---|
| S06-F04 | Multi-head attention with shapes | V15 | done, patched by script |
| S06-F07 | The pre-norm Transformer block | V27 | done, patched by script |
| S06-F08 | A whole GPT, from token ids to probabilities | V29 | done, patched by script |
| S06-F09 | Encoder, decoder, encoder-decoder | V32 | pending |
| S06-F11 | How BERT hides words | V42 | pending |
| S06-F12 | Training in parallel, generating one token at a time | V44 | pending |
| S06-F16 | Four ways to use a pretrained model | V63 | pending |
| S06-F17 | From fine-tuning to LoRA, in three steps | V67 | pending |

## Review checklist (every image)

- Opens correctly and is a 16:9 PNG.
- No title inside the image; no production codes such as "S06-F07".
- Every matrix carries its shape label (for example "$T \times d$"); the exact proportions of the rectangles do not matter.
- Attention parts are orange; trained parameters teal; frozen parameters gray with a small padlock.
- No logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, **not** the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The **order of ideas** may follow a standard explanation; the drawing must be our own.

---

## S06-F04 · Multi-head attention with shapes (done, patched by script, slide 15)

File: `s06_f04_multihead.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Left to right, four stages.

1. Left: one blue rectangle labeled $X$ with the shape label "$T \times d$" under it.
2. Center-left: three arrows fan out from $X$ to three stacked lanes labeled "head 1", "head 2", "head $H$" (with a vertical "⋮" between head 2 and head $H$). In each lane: three small orange rectangles side by side labeled $Q$, $K$, $V$ with the shape "$T \times d_k$" under them, then a small square orange heat map labeled $A$ with shape "$T \times T$", then one small orange rectangle labeled "head output" with shape "$T \times d_k$". Next to the arrow that enters each lane write "$W_Q, W_K, W_V$" in small type, and over the heat map the formula $\mathrm{softmax}\!\big(QK^\top/\sqrt{d_k}\big)$ (only in head 1, to avoid clutter). The three heat maps must show different patterns (one diagonal, one with a bright column, one diffuse) so the heads look different.
3. Center-right: the three head outputs join into one rectangle labeled "concatenate" with shape "$T \times H d_k$", drawn as three vertical strips side by side labeled "1", "2", "$H$" (concatenation along the features, not stacked rows).
4. Right: an arrow labeled $W_O$ to a blue rectangle labeled "output" with shape "$T \times d$".

Bottom strip, gray, full width: "$d_k = d / H$: same parameters and cost as one head of width $d$".

CHECK BEFORE DELIVERING

The formula has the square root, $\sqrt{d_k}$; the concatenation is vertical strips side by side; exactly three lanes drawn (head 1, head 2, head $H$); every rectangle carries its shape label; input and output are both $T \times d$; the three heat maps differ.
```

---

## S06-F07 · The pre-norm Transformer block (done, patched by script, slide 27)

File: `s06_f07_block.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

One tall block in the center, drawn bottom to top, with a thick straight blue vertical band running the full height on its left side: the residual stream. Labels on the band: $X$ at the bottom, "residual stream" along it, "output" at the top.

From the band, two side branches go out to the right and come back, each through these boxes from bottom to top:

1. Lower branch: a light gray box "LN" → an orange box "multi-head attention" (small subtitle "mixes tokens") → the branch returns to the band at a circle with a plus sign "+".
2. Upper branch: a light gray box "LN" → a teal box "MLP" with a small subtitle "$d \rightarrow 4d \rightarrow d$, GELU" and a second subtitle "each token on its own" → returns to the band at a second "+".

To the right of the block, two short annotations aligned with each branch:
- next to the attention branch: "$X \leftarrow X + \mathrm{MHA}(\mathrm{LN}(X))$";
- next to the MLP branch: "$X \leftarrow X + \mathrm{MLP}(\mathrm{LN}(X))$".

At the far left of the band, a vertical bracket covering the whole block labeled "$\times$ number of blocks" in small type.

Small inset at the bottom right, gray outline, titled "post-norm (original)": a mini version of the same block, bottom to top: $X$ → the attention branch rejoins the band at a "+" → then an "LN" box sitting on the band → the MLP branch rejoins at a second "+" → then a second "LN" on the band → output. Attention and MLP stay in side branches, as in the main block; only the LN boxes move onto the band. Caption under the inset: "LN on the residual path".

CHECK BEFORE DELIVERING

The input is labeled $X$ (capital); the formulas contain only the symbols given; in the inset, attention and MLP are side branches and only the two LN boxes sit on the band; in the main block the blue band is never interrupted (no box sits on it, only the two "+" joins); LN is before attention and before the MLP; attention is orange and the MLP teal; the inset is clearly smaller and grayer than the main block.
```

---

## S06-F08 · A whole GPT, from token ids to probabilities (done, patched by script, slide 29)

File: `s06_f08_whole_gpt.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

One vertical stack in the center, drawn bottom to top, each stage a rounded box, with its output shape written to its right in a gray monospace label. From bottom to top:

1. a row of five small integer chips: "17", "903", "41", "8", "2210" with the words "me", "senté", "en", "el", "banco" under them in italics; shape label "$(B, T)$ token ids". ALL FIVE chips send their arrow into the token embedding box (next stage); none goes into the position embedding box;
2. blue box "token embedding $E$" with a small note "lookup: row $w$ of $E$, $\lvert V\rvert \times d$"; beside it a light purple-gray box "position embedding" with the small note "lookup: row $t$, one per position", joined with a "+" (it has no arrow from the chips: it depends on the position, not on the token); shape label "$(B, T, d)$";
3. a tall blue box containing three thin stacked copies of the pre-norm Transformer block (each one drawn as a mini blue band with an orange "attention" chip and a teal "MLP" chip beside it), with a vertical bracket "$\times D$ blocks"; shape label "$(B, T, d)$";
4. gray box "LN"; shape label "$(B, T, d)$";
5. teal box "linear $W_U$, $d \times \lvert V\rvert$"; shape label "$(B, T, \lvert V\rvert)$ logits";
6. teal box "softmax"; on top, for the last position only, a small horizontal bar chart of 5 candidate next tokens with their probabilities: "y" 0.31, "de" 0.22, "." 0.18, "a" 0.09, "," 0.06; label "$p(w_{T+1} \mid w_{\le T})$".

On the left side of the stack, a thin dashed gray curve from the "linear $W_U$" box down to the "token embedding $E$" box, labeled "often the same matrix (tied)".

CHECK BEFORE DELIVERING

Every stage has its shape label; all five token-id arrows go into the token embedding; the stack goes bottom to top; the block stack carries "$\times D$"; only one bar chart, for the last position; the numbers are exactly as written.
```

---

## S06-F09 · Encoder, decoder, encoder-decoder (pending, slide 32)

File: `s06_f09_three_architectures.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Three columns of equal width, each with a teal tag at the top-left, a small diagram and two lines of text below.

1. Tag "encoder": a stack of two blue blocks labeled "block"; under them a row of 5 blue token squares labeled "the movie was not bad"; above them a row of 5 output squares. To the right, a small 5 × 5 attention mask grid with all cells filled orange and the caption "no mask: every token sees every token". Text below: "understand an input" · "BERT, ViT".
2. Tag "decoder": the same two-block stack; under it 5 token squares "the movie was not bad"; above it 5 output squares labeled with the next token, shifted by one: "movie", "was", "not", "bad", "!". To the right, a 5 × 5 mask grid where the lower triangle and the diagonal are orange and the upper triangle is gray with small red crosses; caption "causal mask: token $i$ sees $j \le i$". Text below: "generate, one token at a time" · "GPT".
3. Tag "encoder-decoder": a blue encoder stack on the left reading 5 squares, one word each: "la", "película", "no", "está", "mal"; a teal decoder stack on the right reading 5 squares "<s>", "the", "movie", "is", "not" and producing 5 squares "the", "movie", "is", "not", "bad"; an orange arrow labeled "cross-attention" goes from the top of the encoder stack into the decoder stack. Text below: "input sequence → output sequence" · "original Transformer, T5".

CHECK BEFORE DELIVERING

The decoder mask is lower-triangular including the diagonal; the encoder mask is all orange; the decoder outputs are the inputs shifted left by one; cross-attention appears only in column 3; the Spanish words are copied exactly with the accent in "película" and "está".
```

---

## S06-F11 · How BERT hides words (pending, slide 42)

File: `s06_f11_bert_masking.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Left to right, four stages.

1. Input: the five token chips "me", "senté", "en", "el", "banco".
2. Choose 15 %: the same chips with "senté" outlined in orange; small label "15 % of the tokens are chosen (here, one)".
3. Replace, three branches fanning out from the chosen chip, each with its percentage on the branch:
   - "80 %" → the chip becomes a gray chip "[MASK]";
   - "10 %" → the chip becomes a red-outlined chip "perro" (a random word);
   - "10 %" → the chip stays "senté".
4. Predict: the 80 % branch continues into the row "me", "[MASK]", "en", "el", "banco" entering a blue box "encoder (no mask: reads both sides)". Two orange arrows converge on the "[MASK]" position inside the box: one from "me" (labeled "left context") and one from "en el banco" (labeled "right context"). Out of the box, at the "[MASK]" position only, a teal chip "senté ?" with the label "loss only on the chosen token".

Bottom strip, gray: "the model cannot tell which tokens were chosen, so it must keep a good representation of every token".

CHECK BEFORE DELIVERING

The three percentages are 80, 10, 10; the random word is "perro"; context arrives from both sides of "[MASK]"; the loss is marked only on the chosen position; "senté" keeps its accent.
```

---

## S06-F12 · Training in parallel, generating one token at a time (pending, slide 44)

File: `s06_f12_training_vs_generation.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Two panels side by side, with a teal tag each.

Left panel, tag "training: one forward pass". A row of six input chips: "<s>", "me", "senté", "en", "el", "banco". Above them a blue box "decoder (causal mask)" spanning the row. Above the box, a row of six target chips shifted one place to the left: "me", "senté", "en", "el", "banco", "<end>", each with a small teal "$L_t$" tag under it. To the right of the box, a small 6 × 6 mask grid, lower triangle and diagonal orange, upper triangle gray. Label under the panel: "6 predictions and 6 losses at once".

Right panel, tag "generation: one token per step". Three stacked rows, each one longer than the one above:
1. input chips "<s>" → decoder box → output chip "me";
2. input chips "<s>", "me" → decoder box → output chip "senté";
3. input chips "<s>", "me", "senté" → decoder box → output chip "en".
In each row, a dashed orange curved arrow takes the output chip back to the end of the input of the next row. Label under the panel: "the next input is the token just produced".

CHECK BEFORE DELIVERING

The left targets are the inputs shifted by one; the left mask is lower-triangular including the diagonal; the right panel has exactly three rows that grow by one token each; the words are copied exactly ("senté").
```

---

## S06-F16 · Four ways to use a pretrained model (pending, slide 63)

File: `s06_f16_four_ways.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Four panels side by side, same layout in each: a tall rounded rectangle for the pretrained network (labeled "pretrained network", divided into four horizontal layers) and a small box on top for the task head. A short label at the top of each panel and two lines of text at the bottom.

Color code, shown once in a legend strip along the bottom of the figure: gray with a padlock = "frozen"; teal = "trained"; orange outline = "added".

1. "zero-shot": all four layers gray with padlocks; no task head; instead a small speech bubble "a photo of a {class}" feeding a gray "text encoder" box beside the network. Bottom text: "trains nothing" · "only a prompt".
2. "frozen features + probe": four layers gray with padlocks; head box teal labeled "linear". Bottom text: "trains the head" · "features computed once".
3. "fine-tuning": all four layers teal; head teal. Bottom text: "trains everything" · "small learning rate".
4. "PEFT (LoRA)": four layers gray with padlocks, but each layer has a thin orange-outlined side box attached labeled "$BA_r$" (label only on the top one); head teal. Bottom text: "trains a few added parameters" · "base weights unchanged".

Above all panels, a thin gray arrow pointing right labeled "more parameters trained", starting over panel 1 and ending over panel 3, with panel 4 placed slightly lower so the arrow does not pass over it.

CHECK BEFORE DELIVERING

Padlocks only on gray parts; panel 1 has no head; panel 4 has orange-outlined side boxes on every layer; the legend shows exactly three entries.
```

---

## S06-F17 · From fine-tuning to LoRA, in three steps (pending, slide 67)

File: `s06_f17_lora.png`

```text
Make one image: PNG, 16:9, 1672 x 941 px.

STYLE

Clean academic figure for a deep learning lecture slide, 16:9, 1672 × 941 px, pure white background, flat vector style, thin rounded panels with soft pastel fills, dark navy text, LaTeX-style math typography. No title at the top: the slide provides the title. Use exactly the labels, symbols and numbers given; no extra text, no watermark, no spelling variations. All text horizontal and legible at slide size. English only, except the Spanish words given in quotes, which must be copied letter by letter, accents included.

Notation (use it exactly): token representations $X$, with one ROW PER TOKEN; number of tokens $T$; model width $d$; queries, keys, values $Q$, $K$, $V$; projections $W_Q$, $W_K$, $W_V$, $W_O$; width per head $d_k$; number of heads $H$; attention weights $A$; mask $M$; LayerNorm written "LN"; feed-forward network written "MLP"; activation written GELU; frozen pretrained weight $W_0$; LoRA factors $B$ and $A_r$; rank $r$. Never use $\sigma$ for anything except the sigmoid; the generic activation is $\varphi$. Write the loss as a plain italic $L$, never calligraphic $\mathcal L$. Data flows BOTTOM TO TOP inside a Transformer block (input at the bottom, output at the top), and LEFT TO RIGHT everywhere else.

Running example (for figures that show words): the Spanish sentence "me senté en el banco", five tokens "me", "senté", "en", "el", "banco". Numbers given in a prompt are the only numbers in the image.

Palette: blue #1f6fb4 (token representations, the residual stream, encoders), orange #d9631e (everything that belongs to attention: $Q$, $K$, $V$, attention weights, cross-attention), teal #167d78 (MLP, output heads, decoders, parameters that are TRAINED), gray #8a94a3 (frozen or inactive parts, masked positions), red #c0392b (errors, forbidden connections), navy text #14213d.

Originality: no logos, no photographs, no reproduction of a figure from a paper, a blog or another course (in particular, not the Transformer diagram of Vaswani et al. 2017 with its pink and blue stacked blocks and "N×" bracket, not Jay Alammar's illustrated Transformer figures, not Sebastian Raschka's self-attention, multi-head and LoRA figures (green token bars, pink $K$ and green $V$ blocks, purple stacked heads, the gray "forward pass with updated model" panels), and not the LoRA figure of Hu et al. 2021 with the blue $W$ and orange $A$, $B$ trapezoids). The order of ideas may follow a standard explanation; the drawing must be our own.

FIGURE

Three panels left to right, numbered ①, ②, ③, each with the same input vector $\mathbf x$ (blue, at the bottom) and output vector $\mathbf y$ (blue, at the top). Data flows bottom to top inside each panel.

1. Panel ① "fine-tuning": one large teal square "$W$" with shape "$d \times d$"; beside it a teal side box: "$W \leftarrow W + \Delta W$, with $\Delta W$ from backpropagation". Under the panel: "$d^2$ trained values".
2. Panel ② "the same, in two paths": $\mathbf x$ splits into two parallel paths that join at a "+" before $\mathbf y$: left, a large gray square "$W_0$" with a padlock ("frozen"); right, a large teal square "$\Delta W$" ("trained"), same size as $W_0$. Formula over the "+": $\mathbf y = W_0\mathbf x + \Delta W\mathbf x$. Under the panel: "still $d^2$ trained values".
3. Panel ③ "LoRA": the same two paths, but the right square is replaced by two thin teal rectangles in sequence: first $A_r$ drawn short and wide (shape "$r \times d$"), then $B$ drawn tall and narrow (shape "$d \times r$"), with a tiny vector labeled "$r$" between them; label "$\Delta W = BA_r$". Formula over the "+": $\mathbf y = W_0\mathbf x + BA_r\mathbf x$. Under the panel: "$2dr$ trained values".

Bottom strip, two boxes side by side:
- left box, gray: "$d = 768,\ r = 8$: $\ 2dr = 12\,288$ vs $d^2 = 589\,824$ $\;\rightarrow\;$ 2 %";
- right box, teal outline: "after training: $W = W_0 + BA_r$, no extra cost at inference".

CHECK BEFORE DELIVERING

Panels ② and ③ have the same two-path layout and differ only in the right path; $A_r$ is short and wide, $B$ tall and narrow; only $W_0$ has a padlock; the numbers 12 288, 589 824 and 2 % are exactly as written; the low-rank factor is $A_r$ with the subscript (plain $A$ is the attention matrix).
```
