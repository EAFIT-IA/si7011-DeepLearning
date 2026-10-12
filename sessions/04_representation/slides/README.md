# Session 04 slides · How can a network learn useful representations without labels?

**[Download the slides (PDF)](S04_slides.pdf)**

The PDF cannot play videos, so the slides with animations show a still image. These are the animations; click to watch them:

| Animation | What it shows |
|---|---|
| [The latent space organizes itself](../figures/s04_a01_latent_space.mp4) | The 2-D code of a Fashion-MNIST autoencoder during training: a shapeless cloud splits into groups, and the labels appear only at the end |
| [Denoising pulls points back to the data](../figures/s04_a02_denoising_manifold.mp4) | Noise pushes points off a curve; the best denoiser sends them back toward the data, and its field points to where the data live |
| [Masking in motion](../figures/s04_a03_masking.mp4) | A masked autoencoder: 4 of 16 patches go through a large encoder, a small decoder fills in the 12 hidden ones |
| [Pull and push on the sphere](../figures/s04_a04_contrastive_sphere.mp4) | InfoNCE pulls the two views of an image together and pushes other images apart; with a small temperature, only the nearest negatives push |

Practice for this session: [notebooks](../notebooks/README.md).
