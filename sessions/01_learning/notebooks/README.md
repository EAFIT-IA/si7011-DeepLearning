# Session 01 notebooks

**How does a neural network learn?** Start with the guided practice, then do the exercise.

| # | Notebook | What you do |
|---|---|---|
| 1 | [Guided practice: PyTorch in 60 minutes](sesion_01_pytorch_60min.ipynb) | Everything is written and runs as is. You go from tensors and automatic differentiation to training a small network that separates two classes that a straight line cannot. Run each cell and read the result before moving on. |
| 2 | [Exercise: a multiclass MLP on tabular data](sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) | Your turn. You classify dry beans into 7 varieties from 16 measurements of each bean, filling in **seven TODOs**. |

## The exercise

The data is the [Dry Bean dataset](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset). **Before class:** on Kaggle, add the dataset to your notebook with *Add Input*; on Colab, save your Kaggle token as the secret `KAGGLE_API_TOKEN` (section 1 of the notebook explains how). A CPU is enough. The notebook walks through the three stages you will use for the whole course:

1. **Data:** check that the file has the columns and types you expect, and split and scale it correctly.
2. **Training:** build the network, choose the loss and the optimizer, train, and keep the version that does best on validation.
3. **Deployment:** save everything the model needs in one file, reload it, and predict on new raw rows, just as you would in a real application.

At the end you report accuracy, macro F1 and a confusion matrix, and you will see why accuracy alone can be misleading when some classes are rarer than others.

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_ejercicio_mlp_multiclase_kaggle.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/jdmartinev-org/vision-model/studios/si7011-sesion01/code) |
| `sesion_01_pytorch_60min.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/01_learning/notebooks/sesion_01_pytorch_60min.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/juanmartinezv4399/si7011-sesion-01-pytorch-60min-ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/jdmartinev-org/vision-model/studios/si7011-sesion01/code) |
