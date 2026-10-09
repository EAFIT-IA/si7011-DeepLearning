# Documento de análisis · Evento evaluativo 1 · Parte B

**Pareja:** _nombre 1_ · _nombre 2_ · **Máximo 3 páginas.** Cada afirmación lleva un número y el nombre de la corrida de MLflow de donde sale.

## 1. Línea base
- **Síntoma** (con números de la corrida `1-linea-base`):
- **Medición** que lo explica (pérdida inicial frente a ln 7, norma del gradiente, curvas):
- **Diagnóstico** y en qué notebook de la sesión 2 se vio ese caso:

## 2. Su receta
Una fila por decisión: inicialización, normalización, regularización, optimizador y schedule, clipping.

| Decisión | Por qué (qué problema ataca) | Predicción (escrita antes de correr) | Evidencia en `2-mi-receta` | ¿Se cumplió? |
|---|---|---|---|---|
| | | | | |

## 3. Corrida 3: un solo cambio
- **Cambio:**
- **Predicción** (la misma que registraron en `params`):
- **Resultado y conclusión:**

## 4. Comparación
| Corrida | Pérdida inicial | Mejor época | `best_val_loss` | `val_acc` | F1 macro |
|---|---|---|---|---|---|
| 1-linea-base | | | | | |
| 2-mi-receta | | | | | |
| 3-… | | | | | |

**Prueba (una sola vez, corrida elegida por validación):** accuracy ___ · F1 macro ___. ¿Por qué no usaron la prueba para elegir?

## 5. Despliegue y límites
- Resultado de las verificaciones de `predict_raw` (consistencia y contrato):
- ¿Qué pasaría en producción si `predict_raw` no llamara a `model.eval()`?
- Accuracy frente a F1 macro en estos datos: ¿qué clase falla más y por qué?
- Con más tiempo, ¿qué experimento harían primero y qué esperan que muestre?
