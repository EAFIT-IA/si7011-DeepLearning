# SI7011 · Evento evaluativo 1 · Aprendizaje y entrenamiento (S01–S02) · 20 %

Evalúa si pueden **diagnosticar un entrenamiento con evidencia**, no si recuerdan código. Tiene dos partes cortas:

| Parte | Modalidad | Tiempo | Peso del evento |
|---|---|---|---:|
| **A · Leer la evidencia** | individual, en clase, sin computador | 30 min | 40 % |
| **B · Ejercicio integrador y documento de análisis** | en parejas, en casa, una semana | ≈ 3 h de trabajo | 60 % |

## Parte A · Leer la evidencia

Son seis preguntas de respuesta corta (una o dos frases por literal), cada una construida con figuras de corridas reales:

1. diagnosticar dos corridas a partir de sus curvas de entrenamiento y validación, y proponer una acción;
2. un cálculo corto: el factor por el que cambia la varianza en cada capa;
3. identificar tres inicializaciones a partir de std$(a_l)$ y $\lVert\nabla_{W_l}J\rVert$ por capa;
4. interpretar valores de la pérdida inicial frente a $\ln C$;
5. tamaño del lote, ruido del gradiente y cómo ajustar $\eta$;
6. degradación con la profundidad, y Adam frente a AdamW.

**Examen:** [`parte_a/examen_parte_a.pdf`](parte_a/examen_parte_a.pdf) · [fuente y figuras](parte_a/fuente/). Las figuras salen de corridas reales.

**Para estudiar:** las diapositivas de S01 y S02 y los notebooks 1–4 de la sesión 2. Cada pregunta corresponde a algo que hicieron allí:

| Pregunta | Dónde lo vieron |
|---|---|
| 1 | [notebook 3](../../sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb), secciones 1–4 · [notebook 2](../../sessions/02_deep_training/notebooks/sesion_02_2_optimizacion.ipynb), sección 1 |
| 2, 3, 4 | [notebook 1](../../sessions/02_deep_training/notebooks/sesion_02_1_senal_inicializacion.ipynb) |
| 5 | notebook 2, sección 4 |
| 6 | [notebook 4](../../sessions/02_deep_training/notebooks/sesion_02_4_residual_diagnosticos.ipynb), sección 1 · notebook 3, sección 2 |

## Parte B · Ejercicio integrador y documento de análisis

**Notebook:** el [ejercicio integrador de la sesión 2](../../sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) (Covertype + MLflow) · [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb) [![Abrir en Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/EAFIT-IA/si7011-DeepLearning/blob/main/sessions/02_deep_training/notebooks/sesion_02_6_integrador_covertype.ipynb)

En parejas, una semana. Tiene dos piezas:

1. **El notebook ejecutado**, con los 4 TODO resueltos (partición, modelo, optimizador, `predict_raw`) y las tres corridas registradas en MLflow: línea base, su receta y una corrida que cambia una sola cosa.
2. **Un documento de análisis** en Markdown (`docs/analisis.md`, unas 1500 palabras como máximo), con la estructura de [`plantilla_analisis.md`](plantilla_analisis.md):
   1. **Línea base:** qué falló y con qué medición lo saben (pérdida inicial frente a $\ln C$, gradiente, curvas).
   2. **Receta:** cada decisión con la predicción escrita **antes** de correr, la corrida de MLflow que la respalda y el resultado.
   3. **Corrida 3:** el cambio único, la predicción y si se cumplió.
   4. **Comparación:** una tabla de las tres corridas y la evaluación en prueba, hecha una sola vez.
   5. **Despliegue y límites:** las verificaciones de `predict_raw`, por qué el F1 macro y la accuracy cuentan historias distintas, y qué harían con más tiempo.

Un ejemplo de cómo se registra una decisión con evidencia está en [`ejemplo_evidencia.md`](ejemplo_evidencia.md).

**Entrega: un repositorio de GitHub por pareja.** Pueden partir de [`plantilla_repo/`](plantilla_repo/): copien su contenido a un repositorio nuevo.

```text
evento1-<apellido1>-<apellido2>/
├── README.md                     # integrantes, cómo reproducir, enlace al análisis
├── .gitignore                    # deja fuera los datos de Covertype
├── notebook/
│   └── sesion_02_6_integrador_covertype.ipynb   # ejecutado, con las salidas visibles
├── docs/
│   ├── analisis.md               # el documento de análisis
│   └── figuras/                  # las curvas y tablas que cita el análisis
└── mlflow/
    ├── mlflow.db
    └── mlruns/                   # incluye el best.pt de cada corrida
```

- **Privado**, con el profesor como colaborador (`jdmartinev`).
- Los **dos integrantes hacen commits**: el historial es parte de la evidencia de que ambos trabajaron.
- Las figuras del análisis son archivos en `docs/figuras/` exportados desde el notebook, enlazados desde `analisis.md`. Toda afirmación cita la corrida de MLflow de donde sale.
- **No suban los datos** (`covtype.csv` ni la carpeta que crea `fetch_covtype`): el `.gitignore` de la plantilla ya los excluye.
- **Entregan el enlace al repositorio** antes de la fecha límite. Se califica el último commit anterior a esa hora; lo que se suba después no cuenta.

| Criterio | Peso |
|---|---:|
| Notebook funcional: los 4 TODO y el pipeline completo (datos → entrenamiento → despliegue) | 25 % |
| Evidencia: corridas comparables en MLflow que respaldan cada afirmación del documento | 25 % |
| Diagnóstico de la línea base | 15 % |
| Predicciones escritas antes de cada experimento | 15 % |
| Análisis de resultados, prueba y límites | 20 % |

**Cómo se califica.** Toda afirmación del documento debe citar un número y la corrida de MLflow de donde sale. Una afirmación sin evidencia vale como máximo la mitad. Una predicción que resultó equivocada no resta, siempre que esté escrita antes del experimento y se pueda comprobar. No se exige un F1 mínimo: se evalúa que la receta esté bien armada y que el análisis se apoye en sus corridas.

**Reglas.** Cada pareja entrega su propio repositorio, con su documento y sus corridas; dos entregas con las mismas mediciones o la misma redacción se califican como una sola. Pueden consultar los notebooks del curso y la documentación de PyTorch y MLflow. Si usan asistentes de IA, el documento debe mostrar **sus** mediciones y **sus** corridas: la nota depende de la evidencia registrada en `mlflow.db`.
