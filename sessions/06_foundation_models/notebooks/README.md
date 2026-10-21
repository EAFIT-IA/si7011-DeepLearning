# Session 06 notebooks

**What if we keep the attention and drop the recurrence?** Each notebook changes **one thing** at a time and compares the result with the previous one: before running a cell, guess what will happen; after running it, check whether you were right. Everything is already written and runs as is; each notebook ends with a short *Tu turno* section.

Notebook 1 builds self-attention from scratch and needs no data. Notebook 2 trains a Transformer on the dates task of Session 05 (notebook 5.5): the sentences are generated in the notebook, so there is nothing to download either. Notebook 1 runs in under a minute on a laptop CPU; notebook 2 takes a few minutes with a GPU (Kaggle or Colab) and about 20 minutes on CPU.

Notebook 3 turns a pretrained language model into an assistant: it fine-tunes Qwen2.5-0.5B (base) on 2 000 Spanish question-answer pairs of Dolly-15k, with the masked loss and LoRA of the slides written by hand. **Use a GPU** (Kaggle T4, about 15 minutes) and turn on **Internet** in *Settings*: the model and the data are downloaded from Hugging Face.

Notebooks 4 (CLIP and Chronos) and 5 (DistilBERT on IMDB) are **optional, at home**.

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Autoatención](sesion_06_1_autoatencion.ipynb) | ¿Cómo lee cada token a todos los demás, y cuánto cuesta? |
| 2 | [El bloque Transformer](sesion_06_2_bloque_transformer.ipynb) | ¿Qué gana la tarea de las fechas cuando quitamos la recurrencia? |
| 3 | [Ajuste supervisado (SFT)](sesion_06_3_ajuste_supervisado.ipynb) | ¿Qué le falta a un modelo que solo predice el token siguiente para responder una pregunta? |

## Open online
<!-- open-in-badges -->

| Notebook | |
|---|---|
| `sesion_06_1_autoatencion.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_1_autoatencion.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/juanmartinezv4399/sesion-06-1-autoatencion) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F06_foundation_models%2Fnotebooks%2Fsesion_06_1_autoatencion.ipynb) |
| `sesion_06_2_bloque_transformer.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_2_bloque_transformer.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_2_bloque_transformer.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F06_foundation_models%2Fnotebooks%2Fsesion_06_2_bloque_transformer.ipynb) |
| `sesion_06_3_ajuste_supervisado.ipynb` | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_3_ajuste_supervisado.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_3_ajuste_supervisado.ipynb) [![Abrir en Lightning Studio](https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg)](https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FEAFIT-IA%2Fsi7011-DeepLearning%2Fblob%2Fmain%2Fsessions%2F06_foundation_models%2Fnotebooks%2Fsesion_06_3_ajuste_supervisado.ipynb) |
