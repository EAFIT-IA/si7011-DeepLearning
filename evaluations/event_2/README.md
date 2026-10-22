# SI7011 · Evento evaluativo 2 · Arquitecturas para visión (S03) · 20 %

Igual que el evento 1: **responder unas preguntas** en clase y **completar un notebook** en casa.

| Parte | Qué hacen | Modalidad | Peso del evento |
|---|---|---|---:|
| **A · Preguntas** | seis preguntas de respuesta corta sobre resultados reales de la sesión 3 | individual, en clase, sin computador, 30 min | 40 % |
| **B · Notebook** | completar el ejercicio integrador de la sesión 3 | en parejas, en casa, una semana | 60 % |

## Parte A · Preguntas

**Examen:** [`parte_a/examen_parte_a.pdf`](parte_a/examen_parte_a.pdf). Se responde en una o dos frases por literal.

**Para estudiar:** las diapositivas de S03 y los notebooks 1–5 de la sesión 3.

| Pregunta | Tema | Dónde lo vieron |
|---|---|---|
| 1 | permutar píxeles: MLP frente a CNN | [notebook 1](../../sessions/03_vision/notebooks/sesion_03_1_imagenes_estructura.ipynb), secciones 2–4 |
| 2 | formas, parámetros y campo receptivo | [notebook 2](../../sessions/03_vision/notebooks/sesion_03_2_convolucion.ipynb), secciones 3, 4 y 6 |
| 3, 4 | BatchNorm, augmentation, profundidad y residuales | [notebook 3](../../sessions/03_vision/notebooks/sesion_03_3_cnn_resnet.ipynb), secciones 1–4 |
| 5 | estrategias de transfer learning | [notebook 4](../../sessions/03_vision/notebooks/sesion_03_4_transfer_learning.ipynb), secciones 2–4 |
| 6 | CNN frente a ViT | [notebook 5](../../sessions/03_vision/notebooks/sesion_03_5_patches_vit.ipynb), secciones 1 y 3–5 |

## Parte B · Notebook

**Notebook:** el [ejercicio integrador de la sesión 3](../../sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb) (EuroSAT + MLflow) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb)

Necesita GPU (Kaggle o Colab). En Kaggle, agreguen el dataset [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) con **Add Input**.

1. Resuelvan los 4 TODO y ejecuten el notebook de principio a fin (tres corridas: la CNN desde cero, su receta y una que cambia una sola cosa).
2. Respondan las 5 preguntas de la sección 11 en el mismo notebook, con números de sus corridas.
3. Súbanlo, ejecutado y con las salidas visibles, a `evento2/sesion_03_6_integrador_eurosat.ipynb` en el **repositorio de la pareja**.

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

No se exige una accuracy mínima. Pueden consultar los notebooks del curso y la documentación de PyTorch, torchvision y MLflow.
