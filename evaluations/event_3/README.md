# SI7011 · Evento evaluativo 3 · Secuencias y modelos fundacionales (S05–S06) · 25 %

Igual que los eventos 1 y 2: **responder unas preguntas** en clase y **completar un notebook** en casa.

| Parte | Qué hacen | Modalidad | Peso del evento |
|---|---|---|---:|
| **A · Preguntas** | ocho preguntas de respuesta corta sobre resultados reales de las sesiones 5 y 6 | individual, en clase, sin computador, 45 min | 40 % |
| **B · Notebook** | completar el ejercicio integrador de la sesión 6 | en parejas, en casa, una semana | 60 % |

## Parte A · Preguntas

**Examen:** [`parte_a/examen_parte_a.pdf`](parte_a/examen_parte_a.pdf). Se responde en una o dos frases por literal.

**Para estudiar:** las diapositivas de S05 y S06 y los notebooks de las dos sesiones.

| Pregunta | Tema | Dónde lo vieron |
|---|---|---|
| 1 | partir una serie en el tiempo, fugas, línea base | [notebook 5.1](../../sessions/05_sequences/notebooks/sesion_05_1_series_ventanas.ipynb) |
| 2 | el gradiente a lo largo de la ventana: RNN, LSTM y GRU | [notebook 5.2](../../sessions/05_sequences/notebooks/sesion_05_2_rnn_bptt.ipynb) · [notebook 5.3](../../sessions/05_sequences/notebooks/sesion_05_3_lstm_gru.ipynb) |
| 3 | texto: embeddings, palabras barajadas, GloVe | [notebook 5.4](../../sessions/05_sequences/notebooks/sesion_05_4_texto_embeddings.ipynb) |
| 4 | el cuello de botella del seq2seq y la atención | diapositivas de S05, Parte 5 · [notebook 5.5](../../sessions/05_sequences/notebooks/sesion_05_5_seq2seq_atencion.ipynb) |
| 5 | máscaras de atención y posiciones | [notebook 6.1](../../sessions/06_foundation_models/notebooks/sesion_06_1_autoatencion.ipynb) |
| 6 | el Transformer de las fechas, parámetros y costo de la atención | [notebook 6.2](../../sessions/06_foundation_models/notebooks/sesion_06_2_bloque_transformer.ipynb) |
| 7 | la pérdida del ajuste supervisado y la perplejidad | [notebook 6.3](../../sessions/06_foundation_models/notebooks/sesion_06_3_ajuste_supervisado.ipynb) |
| 8 | congelar, ajustar o LoRA: elegir con evidencia | [notebook 6.5](../../sessions/06_foundation_models/notebooks/sesion_06_5_adaptacion.ipynb) |

## Parte B · Notebook

**Notebook:** el [ejercicio integrador de la sesión 6](../../sessions/06_foundation_models/notebooks/sesion_06_6_integrador_estilo_llm.ipynb) (un LLM que responde con el estilo de *Don Quijote*: TRL, peft y MLflow) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_6_integrador_estilo_llm.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/06_foundation_models/notebooks/sesion_06_6_integrador_estilo_llm.ipynb)

Necesita GPU (Kaggle T4, unos 10 minutos por entrenamiento) con **Internet** activado. En Kaggle, agreguen el dataset [`juanmartinezv4399/si7011-s06-dolly-quijote`](https://www.kaggle.com/datasets/juanmartinezv4399/si7011-s06-dolly-quijote) con **Add Input**.

1. Resuelvan los 4 TODO y ejecuten el notebook de principio a fin (dos entrenamientos: el LoRA del TODO 2 y el de la sección 7, que cambia una sola cosa).
2. Respondan las 5 preguntas de la sección 9 en el mismo notebook, con números de sus corridas.
3. Súbanlo, ejecutado y con las salidas visibles, a `evento3/sesion_06_6_integrador_estilo_llm.ipynb` en el **repositorio de la pareja**.

Es el mismo repositorio de los eventos 1 y 2:

```text
si7011-<apellido1>-<apellido2>/
├── README.md      # nombres de los dos integrantes
├── evento1/
├── evento2/
└── evento3/
    └── sesion_06_6_integrador_estilo_llm.ipynb
```

- No suban los datos, el modelo ni el adaptador, solo el notebook.
- Se califica el último commit antes de la fecha límite.

| Criterio | Peso |
|---|---:|
| Los 4 TODO funcionan y el notebook corre completo (incluidas las verificaciones de `responder_raw`) | 60 % |
| Respuestas de la sección 9, apoyadas en números de sus corridas | 40 % |

No se exige una $J$ mínima ni que el estilo salga perfecto: se evalúa que la comparación de las tres condiciones esté bien hecha y bien leída. Pueden consultar los notebooks del curso y la documentación de Transformers, TRL, peft y MLflow.
