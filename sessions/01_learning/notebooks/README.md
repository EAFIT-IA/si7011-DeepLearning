# Session 01 notebooks

## Guided practice — 60 minutes

[sesion_01_pytorch_60min.ipynb](sesion_01_pytorch_60min.ipynb)

Tensors, affine transformations, scalar autograd, parameter updates,
`TensorDataset`, `DataLoader` and nonlinear classification with an MLP.
The notebook includes executed examples, loss curves and decision boundaries.

## Exercise — Multiclass MLP with tabular data

[sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb](sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb)

Kaggle Dry Bean dataset: 16 numerical features and 7 classes.
Students complete seven TODO sections covering preprocessing, data loaders,
the MLP, loss and optimizer, training, prediction and deployment. The notebook
follows the three pipeline stages used throughout the course, each with an artifact:
**data** (a data contract that validates columns and types, plus a data card with
the file hash), **training** (state selected on validation) and **deployment**
(TODO 7: one `torch.save` bundle with weights, scaler statistics, contract and
class names; `predict_raw` reloads it and predicts on raw rows, with a consistency
check against the test predictions and a rejected malformed row). The evaluation
reports accuracy, macro F1 and a confusion matrix.

The former names `S01_N01_gradient_descent.ipynb` and
`S01_N02_pytorch_autograd.ipynb` were planning placeholders.
The presentation now links to the available notebooks above.

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F01_learning%2Fnotebooks%2Fsesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) |
| `sesion_01_pytorch_60min.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_pytorch_60min.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_pytorch_60min.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F01_learning%2Fnotebooks%2Fsesion_01_pytorch_60min.ipynb) |
