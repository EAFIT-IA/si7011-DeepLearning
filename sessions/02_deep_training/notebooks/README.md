# Session 02 notebooks

**Why can we train deep networks?** Go through notebooks 1 to 5 in order, then do the integrating exercise.

In notebooks 1–5 everything is already written and runs as is. Each cell changes **one thing** — the initialization, the optimizer, a layer — and you compare the new curve with the previous one. Before running a cell, guess what will happen; after running it, check whether you were right. Each notebook takes about 30 minutes and runs on a laptop CPU, and ends with a short *Tu turno* section where you try your own changes.

All five use the same images (Fashion-MNIST) and the same training loop, so the only thing that changes between experiments is the idea being tested.

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Signal and initialization](sesion_02_1_senal_inicializacion.ipynb) | Why does a 20-layer network sometimes learn nothing at all, and how can we tell **before** training? |
| 2 | [Optimization](sesion_02_2_optimizacion.ipynb) | How do the learning rate, momentum, Adam, the batch size and a schedule change the way the loss goes down? |
| 3 | [Normalization and regularization](sesion_02_3_normalizacion_regularizacion.ipynb) | How do we stop a model from memorizing, and what does BatchNorm actually fix? |
| 4 | [Residuals and diagnostics](sesion_02_4_residual_diagnosticos.ipynb) | Why are more layers not always better, and which quick checks catch bugs early? |
| 5 | [Hyperparameter search with Optuna and MLflow](sesion_02_5_optuna_mlflow.ipynb) | How do we search for a good configuration and keep track of every run? |
| 6 | [Integrating exercise · Covertype](sesion_02_6_integrador_covertype.ipynb) | Can you put all the pieces together on new data, from raw rows to a deployable model? |

## The integrating exercise

This one is yours to complete. It uses Forest Covertype: 581,012 forest plots described by 54 variables, where the task is to predict which of 7 types of tree cover dominates each plot.

The data loading, the training loop and the MLflow logging are already written. You fill in **four TODOs**, and each one tells you which notebook shows the piece you need:

1. split and scale the data without leaking information from validation or test;
2. build your model;
3. choose your optimizer and learning-rate schedule;
4. write `predict_raw`, which loads your saved model from MLflow and predicts on new raw rows.

You will compare three runs (a baseline, your recipe, and one change of your choice) and evaluate on the test set once. It is also the deliverable for Part B of [Evaluation Event 1](../../../evaluations/event_1/README.md).

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_02_1_senal_inicializacion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_1_senal_inicializacion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_1_senal_inicializacion.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_1_senal_inicializacion.ipynb) |
| `sesion_02_2_optimizacion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_2_optimizacion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_2_optimizacion.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_2_optimizacion.ipynb) |
| `sesion_02_3_normalizacion_regularizacion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_3_normalizacion_regularizacion.ipynb) |
| `sesion_02_4_residual_diagnosticos.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_4_residual_diagnosticos.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_4_residual_diagnosticos.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_4_residual_diagnosticos.ipynb) |
| `sesion_02_5_optuna_mlflow.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_5_optuna_mlflow.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_5_optuna_mlflow.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_5_optuna_mlflow.ipynb) |
| `sesion_02_6_integrador_covertype.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_6_integrador_covertype.ipynb) |
