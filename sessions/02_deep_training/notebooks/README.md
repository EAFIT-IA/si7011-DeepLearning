# Session 02 notebooks

## Practice A — Inspect a deep MLP · in class, 30 min

[sesion_02_practica_a_senal_profundidad.ipynb](sesion_02_practica_a_senal_profundidad.ipynb)

Executed demo, CPU only, no downloads (about 1.5 minutes).
Forward hooks measure `mean(a_l)`, `std(a_l)`, `std(δ_{a_l})` and `‖∇_{W_l}J‖` layer by layer
for {sigmoid, tanh, ReLU} × {PyTorch default, N(0, 0.01²), N(0, 1), Xavier, He}.
It also reproduces the A01 variance sweep (`c · 2/n_in`) against the theory line, shows dead ReLUs,
checks the initial loss against ln C, and ends with a short training run showing that the
inspection predicts which initializations will train. Questions A1–A7.

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
