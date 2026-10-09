# Session 03 notebooks

**How does data structure shape the network?** Go through notebooks 1 to 5 in order, then do the integrating exercise.

In notebooks 1–5 everything is already written and runs as is. Each cell changes **one thing** — the architecture, a layer, the training strategy — and you compare the new result with the previous one. Before running a cell, guess what will happen; after running it, check whether you were right. Each notebook takes about 30 minutes and ends with a short *Tu turno* section where you try your own changes.

Notebooks 1–3 and 5 use CIFAR-10 (60,000 color images of 32×32 pixels, 10 classes) and the same training loop from Session 02, so the only thing that changes between experiments is the idea being tested. Notebook 4 uses Oxford-IIIT Pets (37 breeds of cats and dogs), because transfer learning matters most when labeled images are few.

Notebooks 1–3 run on a laptop CPU. Notebooks 4, 5 and the integrating exercise run best with a GPU (Colab or Kaggle).

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Images and structure](sesion_03_1_imagenes_estructura.ipynb) | Why does shuffling the pixels not bother an MLP, but ruin a CNN? |
| 2 | [Convolution](sesion_03_2_convolucion.ipynb) | What does a convolution compute, and how do kernel, stride and padding change the output and the receptive field? |
| 3 | [CNNs and residuals](sesion_03_3_cnn_resnet.ipynb) | What does each piece of a modern CNN add: BatchNorm, residual blocks, augmentation? |
| 4 | [Transfer learning](sesion_03_4_transfer_learning.ipynb) | With few labeled images, is it better to train from scratch, train only a new head, or fine-tune? |
| 5 | [From patches to ViT](sesion_03_5_patches_vit.ipynb) | What happens when we drop the assumption of locality and let the model learn it? |
| 6 | [Integrating exercise · EuroSAT](sesion_03_6_integrador_eurosat.ipynb) | Does a model pretrained on ordinary photographs help with images taken from space? |

## The integrating exercise

This one is yours to complete. It uses EuroSAT: 27,000 Sentinel-2 satellite images of 64×64 pixels, labeled with 10 land-use classes (forest, river, highway, crops, residential...).

The data loading, the training loop and the MLflow logging are already written. You fill in **four TODOs**, and each one tells you which notebook shows the piece you need:

1. split the data and build the transforms: preprocessing that matches the backbone, and augmentation that makes sense for satellite images;
2. choose your strategy and build the model: CNN from scratch, linear probe or fine-tuning;
3. choose the optimizer, the learning rates (backbone vs head) and the schedule;
4. write `predict_raw`, which loads your saved model from MLflow and predicts on new image files.

You will compare three runs (a CNN from scratch as baseline, your strategy, and one change of your choice) and evaluate on the test set once.

## Open online
<!-- open-in-badges -->
