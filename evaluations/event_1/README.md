# SI7011 · Evento evaluativo 1 · Aprendizaje y entrenamiento (S01–S02) · 20 %

Son dos cosas: **responder unas preguntas** en clase y **completar un notebook** en casa.

| Parte | Qué hacen | Modalidad | Peso del evento |
|---|---|---|---:|
| **A · Preguntas** | seis preguntas de respuesta corta sobre figuras de corridas reales | individual, en clase, sin computador, 30 min | 40 % |
| **B · Notebook** | completar el ejercicio integrador de la sesión 2 | en parejas, en casa, una semana | 60 % |

## Parte A · Preguntas

**Examen:** [`parte_a/examen_parte_a.pdf`](parte_a/examen_parte_a.pdf). Se responde en una o dos frases por literal.

**Para estudiar:** las diapositivas de S01 y S02 y los notebooks 1–4 de la sesión 2.

| Pregunta | Tema | Dónde lo vieron |
|---|---|---|
| 1 | diagnosticar curvas de entrenamiento y validación | [notebook 3](../../sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb), secciones 1–4 · [notebook 2](../../sessions/02_deep_training/notebooks/sesion_02_2_optimizacion.ipynb), sección 1 |
| 2, 3, 4 | varianza por capa, inicializaciones, pérdida inicial frente a $\ln C$ | [notebook 1](../../sessions/02_deep_training/notebooks/sesion_02_1_senal_inicializacion.ipynb) |
| 5 | tamaño del lote y $\eta$ | notebook 2, sección 4 |
| 6 | profundidad, Adam frente a AdamW | [notebook 4](../../sessions/02_deep_training/notebooks/sesion_02_4_residual_diagnosticos.ipynb), sección 1 · notebook 3, sección 2 |

## Parte B · Notebook

**Notebook:** el [ejercicio integrador de la sesión 2](../../sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) (Covertype + MLflow) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb)

1. Resuelvan los 4 TODO y ejecuten el notebook de principio a fin (tres corridas: la línea base, su receta y una que cambia una sola cosa).
2. Respondan las 5 preguntas de la sección 11 en el mismo notebook, con números de sus corridas.
3. Súbanlo, ejecutado y con las salidas visibles, a `evento1/sesion_02_6_integrador_covertype.ipynb` en el **repositorio de la pareja**.

**Entrega: un solo repositorio de GitHub por pareja para todo el curso.** Cada evento va en su carpeta:

```text
si7011-<apellido1>-<apellido2>/
├── README.md      # nombres de los dos integrantes
├── evento1/
│   └── sesion_02_6_integrador_covertype.ipynb
├── evento2/
│   └── sesion_03_6_integrador_eurosat.ipynb
└── …
```

- **Privado**, con el profesor como colaborador (`jdmartinev`). Lo crean una vez, en el evento 1, y lo reutilizan.
- No suban los datos ni los checkpoints, solo el notebook.
- Se califica el último commit antes de la fecha límite.

| Criterio | Peso |
|---|---:|
| Los 4 TODO funcionan y el notebook corre completo (incluidas las verificaciones de `predict_raw`) | 60 % |
| Respuestas de la sección 11, apoyadas en números de sus corridas | 40 % |

No se exige un F1 mínimo. Pueden consultar los notebooks del curso y la documentación de PyTorch y MLflow.
