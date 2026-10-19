# SI7011 · Evento evaluativo 2 · Arquitecturas para visión (S03) · 20 %

Evalúa si pueden **justificar con evidencia las decisiones de arquitectura y de transfer learning**, no si recuerdan código. Tiene dos partes cortas, como el evento 1:

| Parte | Modalidad | Tiempo | Peso del evento |
|---|---|---|---:|
| **A · Leer la evidencia** | individual, en clase, sin computador | 30 min | 40 % |
| **B · Ejercicio integrador y documento de análisis** | en parejas, en casa, una semana | ≈ 3 h de trabajo | 60 % |

## Parte A · Leer la evidencia

Son seis preguntas de respuesta corta (una o dos frases por literal), construidas con resultados reales de los notebooks de la sesión 03:

1. permutar los píxeles: qué modelo es el MLP y cuál la CNN, y qué suposición se rompe;
2. un cálculo corto: formas, parámetros y campo receptivo de dos convoluciones;
3. BatchNorm y data augmentation: identificar tres corridas por sus accuracies;
4. profundidad y atajos residuales: ¿sobreajuste o problema de optimización?;
5. tres estrategias de transfer learning según la cantidad de datos, y por qué el backbone usa un $\eta$ menor;
6. CNN frente a ViT: sesgo inductivo, embeddings de posición y número de tokens.

**Examen:** [`parte_a/examen_parte_a.pdf`](parte_a/examen_parte_a.pdf) · [fuente y figuras](parte_a/fuente/). Las figuras salen de corridas reales.

**Para estudiar:** las diapositivas de S03 y los notebooks 1–5 de la sesión 3. Cada pregunta corresponde a algo que hicieron allí:

| Pregunta | Dónde lo vieron |
|---|---|
| 1 | [notebook 1](../../sessions/03_vision/notebooks/sesion_03_1_imagenes_estructura.ipynb), secciones 2–4 |
| 2 | [notebook 2](../../sessions/03_vision/notebooks/sesion_03_2_convolucion.ipynb), secciones 3, 4 y 6 |
| 3, 4 | [notebook 3](../../sessions/03_vision/notebooks/sesion_03_3_cnn_resnet.ipynb), secciones 1–4 |
| 5 | [notebook 4](../../sessions/03_vision/notebooks/sesion_03_4_transfer_learning.ipynb), secciones 2–4 |
| 6 | [notebook 5](../../sessions/03_vision/notebooks/sesion_03_5_patches_vit.ipynb), secciones 1 y 3–5 |

## Parte B · Ejercicio integrador y documento de análisis

**Notebook:** el [ejercicio integrador de la sesión 3](../../sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb) (EuroSAT + MLflow) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/03_vision/notebooks/sesion_03_6_integrador_eurosat.ipynb)

En parejas, una semana. Necesita GPU (Kaggle o Colab); en Kaggle, agreguen el dataset [`apollo2506/eurosat-dataset`](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) con **Add Input**. Tiene dos piezas:

1. **El notebook ejecutado**, con los 4 TODO resueltos (partición y augmentation, modelo, optimizador, `predict_raw`) y las tres corridas registradas en MLflow: la CNN desde cero, su receta y una corrida que cambia una sola cosa.
2. **Un documento de análisis** en Markdown (`docs/analisis.md`, unas 1500 palabras como máximo), con la estructura de [`plantilla_analisis.md`](plantilla_analisis.md):
   1. **Línea base:** la predicción escrita antes de correr la CNN desde cero, el resultado y qué la limita.
   2. **Receta:** cada decisión (estrategia, augmentation, $\eta$ por grupo, schedule, `freeze_bn`) con la predicción escrita **antes** de correr, la corrida de MLflow que la respalda y el resultado.
   3. **Corrida 3:** el cambio único, la predicción y si se cumplió.
   4. **Comparación:** una tabla de las tres corridas, con accuracy **y** parámetros entrenados, y la evaluación en prueba, hecha una sola vez.
   5. **Despliegue y límites:** las verificaciones de `predict_raw`, los pares de clases que más se confunden, qué pasaría con imágenes de otro satélite y qué harían con más tiempo.

Un ejemplo de cómo se registra una decisión con evidencia está en [`ejemplo_evidencia.md`](ejemplo_evidencia.md).

**Entrega: un repositorio de GitHub por pareja.** Pueden partir de [`plantilla_repo/`](plantilla_repo/): copien su contenido a un repositorio nuevo.

```text
evento2-<apellido1>-<apellido2>/
├── README.md                     # integrantes, cómo reproducir, enlace al análisis
├── .gitignore                    # deja fuera las imágenes de EuroSAT
├── notebook/
│   └── sesion_03_6_integrador_eurosat.ipynb   # ejecutado, con las salidas visibles
├── docs/
│   ├── analisis.md               # el documento de análisis
│   └── figuras/                  # las curvas, la matriz de confusión y las tablas que cita el análisis
└── mlflow/
    ├── mlflow.db
    └── mlruns/                   # incluye el best.pt de cada corrida
```

- **Privado**, con el profesor como colaborador (`jdmartinev`).
- Los **dos integrantes hacen commits**: el historial es parte de la evidencia de que ambos trabajaron.
- Las figuras del análisis son archivos en `docs/figuras/` exportados desde el notebook, enlazados desde `analisis.md`. Toda afirmación cita la corrida de MLflow de donde sale.
- **No suban los datos** (las imágenes de EuroSAT ni la carpeta `data/`): el `.gitignore` de la plantilla ya los excluye.
- Si un `best.pt` supera el límite de 100 MB de GitHub (un *fine-tuning* de ResNet-18 pesa unos 45 MB, así que no debería pasar), avísenle al profesor antes de la entrega.
- **Entregan el enlace al repositorio** antes de la fecha límite. Se califica el último commit anterior a esa hora; lo que se suba después no cuenta.

| Criterio | Peso |
|---|---:|
| Notebook funcional: los 4 TODO y el pipeline completo (datos → entrenamiento → despliegue) | 25 % |
| Evidencia: corridas comparables en MLflow que respaldan cada afirmación del documento | 25 % |
| Estrategia y augmentation justificadas para imágenes satelitales | 15 % |
| Predicciones escritas antes de cada experimento | 15 % |
| Análisis de resultados, prueba y límites | 20 % |

**Cómo se califica.** Toda afirmación del documento debe citar un número y la corrida de MLflow de donde sale. Una afirmación sin evidencia vale como máximo la mitad. Una predicción que resultó equivocada no resta, siempre que esté escrita antes del experimento y se pueda comprobar. No se exige una accuracy mínima: se evalúa que la receta esté bien armada y que el análisis se apoye en sus corridas.

**Reglas.** Cada pareja entrega su propio repositorio, con su documento y sus corridas; dos entregas con las mismas mediciones o la misma redacción se califican como una sola. Pueden consultar los notebooks del curso y la documentación de PyTorch, torchvision y MLflow. Si usan asistentes de IA, el documento debe mostrar **sus** mediciones y **sus** corridas: la nota depende de la evidencia registrada en `mlflow.db`.
