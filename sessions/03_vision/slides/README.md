# Session 03 slides · How does data structure shape the network?

**[Download the slides (PDF)](S03_slides.pdf)**

The PDF cannot play videos, so the slides with animations show a still image. These are the animations; click to watch them:

| Animation | What it shows |
|---|---|
| [From dense to convolution](../figures/s03_a01_dense_to_conv.mp4) | A dense layer loses its connections to distant inputs, then its weights collapse into one shared kernel: from about a million parameters to 10 |
| [A kernel sliding over an image](../figures/s03_a02_kernel_sliding.mp4) | A vertical-edge kernel slides over a small image; the first position is worked out number by number, then the whole feature map fills in |
| [The receptive field grows with depth](../figures/s03_a03_receptive_field.mp4) | What one unit sees after one, two and three 3×3 layers: 3, 5, 7 pixels with stride 1, and 3, 7, 15 with stride 2 |
| [From image to tokens](../figures/s03_a04_patches_to_tokens.mp4) | An image is cut into patches, each patch is flattened and mapped by the same matrix to a token, and positions and the [CLS] token are added |

Practice for this session: [notebooks](../notebooks/README.md).
