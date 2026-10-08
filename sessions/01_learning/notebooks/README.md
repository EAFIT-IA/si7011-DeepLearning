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
