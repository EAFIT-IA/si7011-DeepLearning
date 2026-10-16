# Session 04 notebooks

**How can a network learn useful representations without explicit labels?** Go through notebooks 1, 2, 4 and 5 in class; notebook 3 is homework. Then do the integrating exercise.

In notebooks 1–5 everything is already written and runs as is. Each cell changes **one thing** — the objective, the corruption, the amount of labels — and you compare the new result with the previous one. Before running a cell, guess what will happen; after running it, check whether you were right. Each notebook ends with a short *Tu turno* section where you try your own changes.

No network in notebooks 1–5 sees a label while it learns its representation. Labels appear only at the end, to **measure** how useful the representation is: with a linear probe (one linear layer on top of the frozen representation) or with $k$ nearest neighbors.

Notebooks 1–3 use Fashion-MNIST (70,000 grayscale images of clothing, 28×28) and run on a laptop CPU. Notebooks 4, 5 and the integrating exercise use CIFAR-10 and EuroSAT and need a GPU (Colab or Kaggle).

**Before class, on Kaggle:** add each notebook's datasets with **Add Input**, so the data are read from disk instead of downloaded:

| Notebooks | Datasets to add |
|---|---|
| 1, 2, 3 | [`zalando-research/fashionmnist`](https://www.kaggle.com/datasets/zalando-research/fashionmnist) |
| 4, 5 | [`pankrzysiu/cifar10-python`](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python), and the model [`juanmartinezv4399/si7011-s04-simclr`](https://www.kaggle.com/models/juanmartinezv4399/si7011-s04-simclr) (the trained SimCLR encoders; in *Add Input*, tab **Models**) |
| 6 | [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) |

Turn on the GPU (*Settings → Accelerator*) for notebooks 4, 5 and 6, and keep *Internet* on: notebooks 5 and 6 download the ImageNet weights of ResNet-18. On Colab, the notebooks download the datasets with your `KAGGLE_API_TOKEN` secret, the same one from Session 01. Each notebook prints the source it used: check it.

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Autoencoders](sesion_04_1_autoencoders.ipynb) | What does a bottleneck keep, and is a better reconstruction a better representation? |
| 2 | [Denoising and convolutional autoencoders](sesion_04_2_denoising_conv.ipynb) | What changes when the network cannot simply copy its input? |
| 3 | [Masked prediction](sesion_04_3_masked_prediction.ipynb) (homework) | How much of an image can you hide and still learn from what is left? |
| 4 | [SimCLR](sesion_04_4_simclr.ipynb) | Can a network learn to recognize objects only by matching two altered views of the same image? |
| 5 | [Evaluating representations](sesion_04_5_evaluar_representaciones.ipynb) | With 1, 10 or 100 labels per class, which representation is worth the most? |
| 6 | [Integrating exercise · EuroSAT without labels](sesion_04_6_integrador_eurosat_ssl.ipynb) | With few labels, is it better to pretrain on ordinary photos with labels or on satellite images without them? |

## The SimCLR checkpoint

Training SimCLR long enough takes hours on a GPU, so notebooks 4 and 5 load a model that is already trained, from the Kaggle model [`juanmartinezv4399/si7011-s04-simclr`](https://www.kaggle.com/models/juanmartinezv4399/si7011-s04-simclr). On Colab it is downloaded with your token; on your computer or on Lightning, put `simclr_resnet18_cifar10.pt` next to the notebook or in `data/`. Notebook 4 trains for 3 epochs in front of you first, so you see the loss go down before loading the long run.

## The integrating exercise

This one is yours to complete. It continues the EuroSAT exercise of Session 03: same images, same contract, same split. This time you pretrain a SimCLR encoder on the satellite images **without their labels**, and then measure it with only 1 % and 10 % of the labels, against an encoder pretrained on ImageNet.

The data loading, the pretraining loop and the MLflow logging are already written. You fill in **four TODOs**, and each one tells you which notebook shows the piece you need:

1. write the two-view augmentation for satellite images (which rotations, flips and colors keep what matters?);
2. write the InfoNCE loss from the formula on the slides;
3. evaluate each encoder with $k$-NN and a linear probe using only a fraction of the labels;
4. write `embed_raw`, which loads your encoder from MLflow and returns embeddings for new raw images, rejecting the ones that break the data contract.

You will compare three encoders (random, ImageNet and yours), change one thing in your recipe, and evaluate on the test set once.

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_04_1_autoencoders.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_1_autoencoders.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_1_autoencoders.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_1_autoencoders.ipynb) |
| `sesion_04_2_denoising_conv.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_2_denoising_conv.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_2_denoising_conv.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_2_denoising_conv.ipynb) |
| `sesion_04_3_masked_prediction.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_3_masked_prediction.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_3_masked_prediction.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_3_masked_prediction.ipynb) |
| `sesion_04_4_simclr.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_4_simclr.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_4_simclr.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_4_simclr.ipynb) |
| `sesion_04_5_evaluar_representaciones.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_5_evaluar_representaciones.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_5_evaluar_representaciones.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_5_evaluar_representaciones.ipynb) |
| `sesion_04_6_integrador_eurosat_ssl.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_6_integrador_eurosat_ssl.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/04_representation/notebooks/sesion_04_6_integrador_eurosat_ssl.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F04_representation%2Fnotebooks%2Fsesion_04_6_integrador_eurosat_ssl.ipynb) |
