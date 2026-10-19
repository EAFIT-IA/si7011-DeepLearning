# Session 06 slides · What if we keep the attention and drop the recurrence?

**[Download the slides (PDF)](S06_slides.pdf)**

The PDF cannot play videos, so the slides with animations show a still image. These are the animations; click to watch them:

| Animation | What it shows |
|---|---|
| [Self-attention](../figures/s06_a01_self_attention.mp4) | One token compares its query with every key, the scores become weights, and the weights average the values; then every token at once, and the causal mask |
| [The block in motion](../figures/s06_a02_block.mp4) | Tokens through one Transformer block: attention mixes information across tokens, the MLP transforms each token on its own, the residual stream runs straight through |
| [Generating text](../figures/s06_a03_generation.mp4) | A language model writes one token at a time: a distribution over the vocabulary, a sampled token, appended to the input; low and high temperature |
| [The loss in motion](../figures/s06_a04_masked_loss.mp4) | How the fine-tuning loss is computed: each position predicts the next token, the probability of the true token becomes a loss, and only the answer tokens count |

Practice for this session: [notebooks](../notebooks/README.md).
