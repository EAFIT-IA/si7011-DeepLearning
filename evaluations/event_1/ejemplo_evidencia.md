# Ejemplo: registrar una decisión con evidencia

Así se ve una entrada bien hecha del documento de análisis. El ejemplo usa otro problema (un MLP sobre `digits` de scikit-learn, semilla 0) para no adelantar nada de Covertype, pero el formato es el que se pide en las secciones 1–3. Las cifras son reales.

**Situación:** las entradas llegan sin escalar (enteros entre 0 y 16), porque falta el `÷ 16` al preparar los datos.

| Paso | Registro |
|---|---|
| **Síntoma** | En la corrida `0-linea-base` la pérdida inicial es **22.8**, cuando con 10 clases debería estar cerca de $\ln 10 = 2.30$. La primera época termina en 12.2 y la norma del gradiente en la época 1 es 41.2. Aun así, al final llega a 96.4 % en validación. |
| **Medición** | `activation_stds(make_model(RECETA), X_train[:256])` da **6.7, 6.8 y 6.6** en las tres capas. Con He y entradas estandarizadas esperaba algo cercano a 1. La escala es estable entre capas, así que el problema no se amplifica con la profundidad: viene **desde la entrada**. `X_train.std()` = 6.0 y `X_train.max()` = 16. |
| **Hipótesis** | Las entradas no están escaladas. Es un error de **datos/código**, no de inicialización: He supone entradas de varianza ≈ 1. |
| **Predicción** (antes de correr) | Si divido `X_all` por 16, la pérdida inicial bajará a ≈ 2.5, la std por capa a ≈ 0.4 y la primera época empezará cerca de 2.2. La accuracy final cambiará poco, porque Adam compensa en parte la escala, pero la pérdida mínima de validación debería bajar. |
| **Experimento** | Una sola cosa: `X_all = validate_rows(df) / 16.0`. Corrida `1-escalar-entradas`. |
| **Resultado** | Pérdida inicial **2.52** · std por capa 0.42, 0.43, 0.41 · primera época 2.24 · norma del gradiente en la época 1: 1.77 · mejor `val_loss` 0.12 (antes 0.16) · mejor `val_acc` 0.972 (antes 0.964). |
| **Conclusión** | La predicción se cumplió en los cuatro números. La falla estaba en la sección 1 (faltaba `/ 16.0`), así que es de código. La pista clave fue la pérdida inicial frente a $\ln C$, y la medición que la confirmó fue la std de la **primera** capa: si el problema hubiera sido la inicialización, la escala habría cambiado capa a capa. |

Fíjense en tres cosas de este registro:

- cada afirmación tiene un **número** y el nombre de la **corrida** de donde sale;
- la predicción es **comprobable**: dice qué números deberían cambiar y hacia dónde;
- el experimento cambia **una sola cosa**, así que el cambio en los resultados se le puede atribuir a esa cosa.
