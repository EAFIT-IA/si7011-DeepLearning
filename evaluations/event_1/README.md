# SI7011 · Evento evaluativo 1 · Aprendizaje y entrenamiento (S01–S02) · 20 %

Evalúa si pueden **diagnosticar un entrenamiento con evidencia**, no si recuerdan código. Tiene dos partes cortas:

| Parte | Modalidad | Tiempo | Peso del evento |
|---|---|---|---:|
| **A · Leer la evidencia** | individual, en clase, sin computador | 30 min | 40 % |
| **B · Diagnóstico de una receta** | en parejas, en casa, una semana | ≈ 3 h de trabajo | 60 % |

## Parte A · Leer la evidencia

Son seis preguntas de respuesta corta (una o dos frases por literal), cada una construida con figuras de corridas reales:

1. diagnosticar dos corridas a partir de sus curvas de entrenamiento y validación, y proponer una acción;
2. un cálculo corto: el factor por el que cambia la varianza en cada capa;
3. identificar tres inicializaciones a partir de std$(a_l)$ y $\lVert\nabla_{W_l}J\rVert$ por capa;
4. interpretar valores de la pérdida inicial frente a $\ln C$;
5. tamaño del lote, ruido del gradiente y cómo ajustar $\eta$;
6. degradación con la profundidad, y Adam frente a AdamW.

**Para estudiar:** las diapositivas de S01 y S02 y los notebooks 1–4 de la sesión 2. Cada pregunta corresponde a algo que hicieron allí:

| Pregunta | Dónde lo vieron |
|---|---|
| 1 | [notebook 3](../../sessions/02_deep_training/notebooks/sesion_02_3_normalizacion_regularizacion.ipynb), secciones 1–4 · [notebook 2](../../sessions/02_deep_training/notebooks/sesion_02_2_optimizacion.ipynb), sección 1 |
| 2, 3, 4 | [notebook 1](../../sessions/02_deep_training/notebooks/sesion_02_1_senal_inicializacion.ipynb) |
| 5 | notebook 2, sección 4 |
| 6 | [notebook 4](../../sessions/02_deep_training/notebooks/sesion_02_4_residual_diagnosticos.ipynb), sección 1 · notebook 3, sección 2 |

## Parte B · Diagnóstico de una receta de entrenamiento

Cada pareja recibe **su propio** notebook (no está en este repositorio). Entrena un MLP sobre `digits`: imágenes de 8×8 que vienen con scikit-learn, sin descargas, y que corre en CPU en segundos.

La receta tiene **exactamente dos fallas**: una de **código**, en las funciones del modelo o del loop, y una de **configuración**, en el diccionario `RECETA`. Cada pareja recibe una combinación distinta.

Para cada falla registran en la bitácora del notebook:

> síntoma → medición → hipótesis → **predicción** (escrita antes de correr) → experimento que cambia **una sola cosa** (registrado en MLflow) → resultado → conclusión

El notebook trae listos los instrumentos de medición de la sesión 2: std$(a_l)$ por capa, $\lVert\nabla_{W_l}J\rVert$ por capa, sobreajustar un lote, $\lVert W\rVert$ y evaluar dos veces. También trae la función `run`, que registra cada corrida en MLflow. Un ejemplo de bitácora bien hecha, con una falla que no le toca a ninguna pareja, está en [`ejemplo_bitacora.md`](ejemplo_bitacora.md).

Después:

1. corrigen las dos fallas y entrenan la receta final (en validación: accuracy ≥ 95 % y pérdida ≤ 0.25);
2. evalúan en prueba **una sola vez** (la celda ya está escrita);
3. completan el despliegue mínimo: `predict_raw` recarga la receta desde MLflow y predice sobre filas crudas, y dos verificaciones comprueban que coincide con la prueba y que el contrato rechaza una fila inválida;
4. responden dos preguntas finales citando corridas de MLflow por su nombre.

**Entrega:** un `.zip` con el notebook ejecutado y `mlflow.db` (con `mlruns/`).

| Criterio | Peso |
|---|---:|
| Diagnóstico de la falla de código | 20 % |
| Diagnóstico de la falla de configuración | 20 % |
| Evidencia: mediciones y corridas comparables en MLflow | 25 % |
| Predicciones escritas antes de cada experimento | 10 % |
| Receta final y uso correcto de prueba | 10 % |
| Despliegue: recarga desde MLflow y predicción sobre datos crudos | 5 % |
| Dos preguntas finales, respondidas con sus corridas | 10 % |

**Cómo se califica el diagnóstico.** Señalar la línea o el valor equivocado sin una medición que lo muestre vale como máximo la mitad. Una predicción que resultó equivocada no resta, siempre que esté escrita antes del experimento y se pueda comprobar.

**Reglas.** Cada pareja trabaja solo con su variante. Pueden consultar los notebooks del curso y la documentación de PyTorch y MLflow. Si usan asistentes de IA, la bitácora debe mostrar **sus** mediciones y **sus** corridas: la nota depende de la evidencia registrada en `mlflow.db`, no del diagnóstico final.
