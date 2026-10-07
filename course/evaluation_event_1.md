# SI7011 · Evento evaluativo 1 · Aprendizaje y entrenamiento (S01–S02) · 20 %

Evalúa si pueden **diagnosticar** un entrenamiento con evidencia, no si recuerdan código.
Dos partes cortas:

| Parte | Modalidad | Tiempo | Peso del evento |
|---|---|---|---:|
| **A · Leer la evidencia** | individual, en clase, sin computador | 30 min | 40 % |
| **B · Diagnóstico de una receta** | en parejas, en casa, una semana | ≈ 3 h de trabajo | 60 % |

## Parte A · Leer la evidencia

Seis preguntas de respuesta corta construidas con corridas reales:

- diagnosticar curvas de entrenamiento y validación y proponer una acción;
- leer mediciones por capa (std$(a_l)$, $\lVert\nabla_{W_l}J\rVert$) e identificar la inicialización;
- dos cálculos cortos (factor de varianza por capa, pérdida inicial esperada);
- tamaño del lote y ruido del gradiente; Adam frente a AdamW; degradación con la profundidad.

Repasen: diapositivas de S01 y S02, Práctica A de S02.

## Parte B · Diagnóstico de una receta de entrenamiento

Cada pareja recibe un notebook que entrena un MLP sobre `digits` (scikit-learn, sin descargas, corre en CPU en segundos).
La receta tiene **exactamente dos fallas**: una de **código** y una de **configuración**. Cada pareja recibe una combinación distinta.

Para cada falla registran: síntoma → medición → hipótesis → **predicción** → experimento que cambia una sola cosa (en MLflow) → resultado → conclusión.
Después corrigen ambas fallas, entrenan la receta final y evalúan en prueba **una sola vez**.

**Entrega:** un `.zip` con el notebook ejecutado y `mlflow.db` (con `mlruns/`).

| Criterio | Peso |
|---|---:|
| Diagnóstico de la falla de código | 20 % |
| Diagnóstico de la falla de configuración | 20 % |
| Evidencia: mediciones y corridas comparables en MLflow | 30 % |
| Predicciones escritas antes de cada experimento | 10 % |
| Receta final (validación ≥ 95 %) y uso correcto de prueba | 10 % |
| Dos preguntas finales, respondidas con sus corridas | 10 % |

Encontrar la falla leyendo el código no basta: la nota depende de la evidencia.
