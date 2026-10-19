# Session 05 slides · How does a network remember the past?

**[Download the slides (PDF)](S05_slides.pdf)**

The PDF cannot play videos, so the slides with animations show a still image. These are the animations; click to watch them:

| Animation | What it shows |
|---|---|
| [Unrolling through time](../figures/s05_a01_unrolling_bptt.mp4) | A recurrent cell unrolled over 8 steps with the same weights at every step; the gradient flowing back grows weaker step by step |
| [The memory in motion](../figures/s05_a02_lstm_memory.mp4) | An LSTM writes a value into its memory, keeps it for several steps and reads it later; in a plain recurrent network the same value fades |
| [Attention in motion](../figures/s05_a03_seq2seq_attention.mp4) | Converting a date written in words: at each output character, the decoder decides which input words to look at |

Practice for this session: [notebooks](../notebooks/README.md).
