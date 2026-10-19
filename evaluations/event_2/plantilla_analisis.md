# Documento de análisis · Evento evaluativo 2 · Parte B

> Copien este archivo a `docs/analisis.md` en su repositorio.

**Pareja:** _nombre 1_ · _nombre 2_ · **Unas 1500 palabras como máximo.** Cada afirmación lleva un número y el nombre de la corrida de MLflow de donde sale. Las figuras van en `docs/figuras/` y se enlazan así: `![Curvas de las tres corridas](figuras/curvas.png)`.

## 1. Línea base: una CNN desde cero
- **Predicción** (escrita antes de correr `1-cnn-desde-cero`): accuracy de validación esperada y por qué.
- **Resultado** (pérdida inicial frente a ln 10, mejor época, `best_val_acc`, curvas):
- **Lectura:** ¿qué limita a esta red? ¿Le falta capacidad, datos o una mejor representación de partida?

## 2. Su receta
Una fila por decisión: estrategia (*linear probe*, *fine-tuning* u otra), augmentation, $\eta$ del backbone y de la cabeza, schedule, `freeze_bn`.

| Decisión | Por qué (qué problema ataca) | Predicción (escrita antes de correr) | Evidencia en `2-mi-receta` | ¿Se cumplió? |
|---|---|---|---|---|
| | | | | |

Para la augmentation: qué transformaciones eligieron y por qué **no cambian la clase** de una imagen satelital.

## 3. Corrida 3: un solo cambio
- **Cambio:**
- **Predicción** (la misma que registraron en `params`):
- **Resultado y conclusión:**

## 4. Comparación
| Corrida | `n_trainable` | Pérdida inicial | Mejor época | `best_val_loss` | `val_acc` | F1 macro |
|---|---|---|---|---|---|---|
| 1-cnn-desde-cero | | | | | | |
| 2-mi-receta | | | | | | |
| 3-… | | | | | | |

**Prueba (una sola vez, corrida elegida por validación):** accuracy ___ · F1 macro ___. ¿Por qué no usaron la prueba para elegir?

**¿Ayudó ImageNet con imágenes satelitales?** Respondan con la tabla: accuracy **y** número de parámetros entrenados.

## 5. Despliegue y límites
- Resultado de las verificaciones de `predict_raw` (consistencia y contrato):
- La matriz de confusión: ¿qué par de clases se confunde más, y por qué es razonable mirando las imágenes?
- ¿Qué pasaría en producción si llegaran imágenes de 128×128 de otro satélite? ¿Qué hace hoy `predict_raw` y qué debería hacer?
- Con más tiempo, ¿qué experimento harían primero y qué esperan que muestre?
