# Session 05 notebooks

**How does a network remember the past?** Each notebook changes **one thing** at a time and compares the result with the previous one: before running a cell, guess what will happen; after running it, check whether you were right. Everything is already written and runs as is; each notebook ends with a short *Tu turno* section.

Notebooks 1 and 2 use **Jena Climate**: 14 weather variables every 10 minutes, 2009–2016, from the Max Planck Institute for Biogeochemistry. The task is the same in both: 5 days of hourly readings → temperature 24 hours later. Notebook 5 generates its own data (Spanish sentences that contain a date). All three run on a laptop CPU in a few minutes.

**Before class, on Kaggle:** add the dataset [`mnassrib/jena-climate`](https://www.kaggle.com/datasets/mnassrib/jena-climate) with **Add Input** (notebooks 1 and 2). On Colab, the notebooks download it with your `KAGGLE_API_TOKEN` secret, the same one from Session 01; on your computer or on Lightning, put `jena_climate_2009_2016.csv` in a `data/` folder next to the notebook. Each notebook prints the source it used: check it.

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Series y ventanas](sesion_05_1_series_ventanas.ipynb) | ¿Qué es un ejemplo en una serie de tiempo, y qué número hay que superar? |
| 2 | [RNN y retropropagación en el tiempo](sesion_05_2_rnn_bptt.ipynb) | ¿Hasta dónde llega el gradiente en una red recurrente? |
| 5 | [Seq2seq y atención](sesion_05_5_seq2seq_atencion.ipynb) | ¿Qué cabe en un solo vector, y qué cambia cuando el decodificador puede mirar atrás? |

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_05_1_series_ventanas.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_1_series_ventanas.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_1_series_ventanas.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F05_sequences%2Fnotebooks%2Fsesion_05_1_series_ventanas.ipynb) |
| `sesion_05_2_rnn_bptt.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_2_rnn_bptt.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_2_rnn_bptt.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F05_sequences%2Fnotebooks%2Fsesion_05_2_rnn_bptt.ipynb) |
| `sesion_05_5_seq2seq_atencion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_5_seq2seq_atencion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/05_sequences/notebooks/sesion_05_5_seq2seq_atencion.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F05_sequences%2Fnotebooks%2Fsesion_05_5_seq2seq_atencion.ipynb) |
