# Session 06 notebooks

**What if we keep the attention and drop the recurrence?** Each notebook changes **one thing** at a time and compares the result with the previous one: before running a cell, guess what will happen; after running it, check whether you were right. Everything is already written and runs as is; each notebook ends with a short *Tu turno* section.

Notebook 1 builds self-attention from scratch and needs no data. Notebook 2 trains a Transformer on the dates task of Session 05 (notebook 5.5): the sentences are generated in the notebook, so there is nothing to download either. Notebook 1 runs in under a minute on a laptop CPU; notebook 2 takes a few minutes with a GPU (Kaggle or Colab) and about 20 minutes on CPU.

| # | Notebook | The question it answers |
|---|---|---|
| 1 | [Autoatención](sesion_06_1_autoatencion.ipynb) | ¿Cómo lee cada token a todos los demás, y cuánto cuesta? |
| 2 | [El bloque Transformer](sesion_06_2_bloque_transformer.ipynb) | ¿Qué gana la tarea de las fechas cuando quitamos la recurrencia? |

## Open online
<!-- open-in-badges -->
