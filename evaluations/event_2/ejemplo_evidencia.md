# Ejemplo: registrar una decisión con evidencia

Así se ve una entrada bien hecha del documento de análisis. El ejemplo usa otro problema, el de Oxford-IIIT Pets del [notebook 4 de la sesión 03](../../sessions/03_vision/notebooks/sesion_03_4_transfer_learning.ipynb), para no adelantar nada de EuroSAT. Las cifras son reales: son las salidas guardadas de ese notebook. Allí no hay MLflow, así que se cita la sección; en su documento, cada número lleva el nombre de la corrida de MLflow de donde sale.

**Situación:** 37 razas de perros y gatos, **5 imágenes por raza** (185 en total) y una ResNet-18 preentrenada en ImageNet. Hay que elegir entre *linear probe* y *fine-tuning*.

| Paso | Registro |
|---|---|
| **Pregunta** | Con 185 imágenes, ¿conviene entrenar solo una cabeza nueva sobre el backbone congelado, o ajustar toda la red? |
| **Medición** | Parámetros entrenables: *linear probe* **18 981** (la cabeza: 512 × 37 + 37); *fine-tuning* **11 195 493** (el backbone, 11 176 512, más la misma cabeza). Son unos **100 parámetros por imagen** frente a unos **60 000**. |
| **Hipótesis** | Con tan pocas imágenes, el *fine-tuning* puede ajustar el backbone a esas 185 fotos y perder parte de lo que traía de ImageNet. El *linear probe* no puede: solo mueve la cabeza. |
| **Predicción** (antes de correr) | Con 5 por raza, el *probe* superará al *fine-tuning* en validación, y la pérdida de entrenamiento del *fine-tuning* será mucho menor que la que corresponde a su accuracy de validación. Con todas las imágenes (≈ 80 por raza), el *fine-tuning* lo alcanzará o lo superará. |
| **Experimento** | Una sola cosa cambia: la estrategia. Mismas particiones, mismo backbone de ImageNet, misma cabeza. *Fine-tuning* con $\eta$ = 10⁻⁴ en el backbone y 10⁻³ en la cabeza, *warmup* + coseno, `freeze_bn`, 8 épocas (secciones 2 y 4). |
| **Resultado** | 5 por raza: *probe* **0.762**, *fine-tuning* **0.588**; el *fine-tuning* termina con pérdida de entrenamiento 0.159. 20 por raza: 0.834 frente a 0.789. Todas: 0.849 frente a **0.872**. Tiempo con 5 por raza: 0.6 s frente a 5 s. Desde cero, como referencia: 0.086, 0.174 y 0.305. |
| **Conclusión** | La predicción se cumplió en las tres cantidades de datos. Con 185 imágenes, el *fine-tuning* llega a una pérdida de entrenamiento baja pero se queda 17 puntos por debajo del *probe* en validación: memoriza las fotos de entrenamiento. Con ≈ 80 por raza ya hay datos para ajustar el backbone sin perder lo de ImageNet, y gana por 2 puntos. La decisión depende de cuántas imágenes etiquetadas hay, y se puede medir. |

Fíjense en tres cosas de este registro:

- cada afirmación tiene un **número** y dice de dónde sale;
- la predicción es **comprobable**: dice qué números deberían cambiar y hacia dónde;
- el experimento cambia **una sola cosa**, así que el cambio en los resultados se le puede atribuir a esa cosa.
