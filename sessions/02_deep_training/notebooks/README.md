# Session 02 notebooks

## New structure (in progress)

One notebook per part of the slides. Every notebook is fully worked and executed, uses Fashion-MNIST as
784-dimensional vectors, and changes **one thing per cell** before overlaying the curves (format taken from
Chollet's *Deep Learning with Python*, ch. 5). The theory stays in the slides. Each notebook ends with a short
*Tu turno* list of experiments (about 10 min). Covertype + MLflow becomes the integrating exercise.

| # | Notebook | Status |
|---|---|---|
| 1 | Signal and initialization (from Practice A) | pending |
| 2 | Optimization: SGD, momentum, Adam, batch size, schedule, clipping | pending |
| 3 | [Normalization and regularization](sesion_02_3_normalizacion_regularizacion.ipynb) | **pilot** |
| 4 | Residual connections, LayerNorm and diagnostics | pending |
| — | Integrating exercise: Covertype + MLflow | pending (from Practice B) |

**3 · Normalization and regularization** (about 1 minute on CPU). Baseline MLP 784–512–512–10 trained on 10,000
examples overfits (validation loss 0.39 at epoch 6 → 0.52 at epoch 20). One change at a time: AdamW with
weight decay 1.0 (final validation loss 0.41, ‖W‖ halved), Dropout 0.5 (0.40), early stopping with
`patience = 3` (restores epoch 6). A 20-layer MLP with SGD stays at ln 10 ≈ 2.303, and with BatchNorm it reaches 84%.
It also shows BatchNorm in `train()` versus `eval()` (batch of one, running statistics) and evaluates the test set once.

## Practice A — Inspect a deep MLP · in class, 30 min

[sesion_02_practica_a_senal_profundidad.ipynb](sesion_02_practica_a_senal_profundidad.ipynb)

Executed demo, CPU only, no downloads (about 1.5 minutes).
Forward hooks measure `mean(a_l)`, `std(a_l)`, `std(δ_{a_l})` and `‖∇_{W_l}J‖` layer by layer
for {sigmoid, tanh, ReLU} × {PyTorch default, N(0, 0.01²), N(0, 1), Xavier, He}.
It also reproduces the A01 variance sweep (`c · 2/n_in`) against the theory line, shows dead ReLUs,
checks the initial loss against ln C, and ends with a short training run showing that the
inspection predicts which initializations will train. Questions A1–A7.

## Worked example — training recipe and MLflow · in class, 20 min

[sesion_02_ejemplo_receta_mlflow.ipynb](sesion_02_ejemplo_receta_mlflow.ipynb)

Fully implemented and executed, on synthetic tabular data (`make_classification`: 30,000 rows, 40 features with
very different scales, 5 classes; no downloads, about 1.5 minutes on CPU). It shows every piece Practice B asks
students to build: data contract and data card, split and scaler fitted on training, configurable MLP (He,
BatchNorm, Dropout), AdamW, warmup + cosine, gradient-norm measurement and clipping, early stopping, MLflow
logging with a deployable `best.pt` (weights + scaler + contract), comparison with `search_runs`, and `predict_raw`
from the MLflow artifact on raw rows. A deep plain baseline stays at ln 5 while the full recipe reaches a
validation macro F1 of about 0.94.

## Practice B — A robust training recipe with MLflow · student work, 90 min

[sesion_02_practica_b_receta_mlflow.ipynb](sesion_02_practica_b_receta_mlflow.ipynb)

Forest Covertype (UCI / Kaggle `uciml/forest-cover-type-dataset`): 581,012 rows, 54 features,
7 imbalanced classes. Plain PyTorch with one `cfg` dict per run. Eight TODOs:
data split and scaling, initialization, configurable MLP (BatchNorm/Dropout), optimizer
(SGD/Adam/AdamW), warmup + cosine schedule, training step with gradient-norm measurement and clipping,
and the experiment ladder R0–R5 (baseline → He → AdamW → BatchNorm → weight decay + Dropout →
schedule + clipping + early stopping).

It follows the data → training → deployment pipeline with an artifact per stage.
**Data:** a contract (54 columns, 10 numeric, 44 binary indicators) validated on load and a
data card logged with every run. **Training:** every run is logged to MLflow (local
`sqlite:///mlflow.db` + `mlruns/`): parameters, per-epoch metrics, learning rate, gradient norm,
initial-loss checks and a `best.pt` artifact that carries the weights **and** the scaler statistics
and data contract. **Deployment (section 11, TODO 8):** `predict_raw` loads the chosen run from MLflow
and predicts on raw rows; checks cover consistency with the test predictions, contract rejection,
batch inference from a CSV with per-row and per-batch latency, and export with `torch.export` (`.pt2`).
Serving behind an API is left for a later session.
Runs are compared in the notebook with `mlflow.search_runs` and `get_metric_history`;
the MLflow UI is optional (Colab cell provided; on Kaggle download `mlflow.db` and `mlruns/`).
The test set is evaluated once, for the recipe chosen on validation.

With the default `TRAIN_SIZE = 100_000`, the six runs take about 13 minutes on CPU and a few
minutes on a GPU. The instructor solution is kept outside the repository.

## Hyperparameter search with Optuna and MLflow · 60 min

[sesion_02_busqueda_hiperparametros.ipynb](sesion_02_busqueda_hiperparametros.ipynb)

Same Covertype pipeline as Practice B (contract, 60/20/20 split, scaler fitted on training; 30,000 training
examples so the search fits in class). A logged baseline, then an Optuna study (TPE sampler, median pruner)
over learning rate and weight decay (log scale), Dropout, depth, width, batch size and normalization.
Two TODOs: the search space and the pruning callback. In MLflow, the study is a parent run and every trial
a nested child run with its parameters, per-epoch curves and state (`COMPLETE` / `PRUNED`); the parent stores
the best parameters and the optimization-history, importance and slice plots. Students compare trials with
MLflow's *Parallel Coordinates Plot*, retrain the best configuration, evaluate on test once and log a deployable
artifact (weights + scaler + contract) compatible with Practice B's `predict_raw`.

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_02_3_normalizacion_regularizacion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_3_normalizacion_regularizacion.ipynb) |
| `sesion_02_busqueda_hiperparametros.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_busqueda_hiperparametros.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_busqueda_hiperparametros.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_busqueda_hiperparametros.ipynb) |
| `sesion_02_ejemplo_receta_mlflow.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_ejemplo_receta_mlflow.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_ejemplo_receta_mlflow.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_ejemplo_receta_mlflow.ipynb) |
| `sesion_02_practica_a_senal_profundidad.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_practica_a_senal_profundidad.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_practica_a_senal_profundidad.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_practica_a_senal_profundidad.ipynb) |
| `sesion_02_practica_b_receta_mlflow.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_practica_b_receta_mlflow.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_practica_b_receta_mlflow.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F02_deep_training%2Fnotebooks%2Fsesion_02_practica_b_receta_mlflow.ipynb) |
