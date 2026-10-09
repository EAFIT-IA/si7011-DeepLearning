# Evento evaluativo 1 · Parte A · Clave del instructor

Las figuras salen de corridas reales sobre Fashion-MNIST (`parte_a_figs.py`, semilla 0). Las cifras de abajo son las de esas corridas. Cada pregunta vale 1 punto (literales de 0.5); nota = puntos × 5 / 6.

## 1. Dos corridas (0.5 + 0.5)
- **A — sobreajuste.** La pérdida de entrenamiento baja hasta 0.02, y la de validación toca su mínimo (0.45) en la época 6 y sube hasta 0.81. La brecha crece. **B — subajuste por optimización lenta.** Entrenamiento y validación van juntas (0.72 y 0.73 en la época 30) y siguen bajando: el modelo no ha terminado de aprender. Fue SGD con η = 2·10⁻⁴.
- **Acciones.** A: early stopping (quedarse con la época ≈ 6), Dropout, weight decay o más datos; se confirma si la validación deja de subir o la brecha se reduce. B: subir η, agregar momentum o usar Adam; se confirma si las dos curvas bajan más rápido y siguen juntas.
- Media nota si diagnostica sin citar la evidencia. No suma si llama "sobreajuste" a B.

## 2. Cálculo de varianza (0.3 + 0.4 + 0.3)
- a) $n_\mathrm{in}\sigma^2/2 = 400 \cdot 0.0025 / 2 = 0.5$.
- b) Varianza × $0.5^{10} = 1/1024$, así que la std queda × $1/32 \approx 0.031$.
- c) $\sigma = \sqrt{2/400} \approx 0.071$: inicialización **He** (Kaiming).

## 3. Tres inicializaciones (0.6 + 0.4)
- **I = He:** std estable (0.83 → 0.95) y gradientes de 4 a 7 en todas las capas. **II = PyTorch por defecto:** la std cae a 0.02 en cuatro capas y el gradiente de la primera capa es ≈ 6·10⁻⁶. **III = Xavier:** la std decae de forma sostenida (0.72 → 0.006) y los gradientes son pequeños y parejos (≈ 10⁻²).
- **b)** I. Con II, las primeras capas reciben gradientes del orden de 10⁻⁶ y prácticamente no aprenden. Con III aprenden muy despacio y la señal se pierde hacia la salida.
- Acepta III = Xavier si el argumento es "la std cae un factor ≈ 2 por cada dos capas" o "Xavier no compensa la mitad que anula ReLU".

## 4. Pérdida inicial (0.3 + 0.35 + 0.35)
- a) 1.95 ≈ ln 7 = 1.946: está bien, predice casi uniforme.
- b) 38.2 ≫ ln 7: logits enormes. Pesos demasiado grandes (p. ej., N(0, 1)), entradas sin escalar o capa de salida mal inicializada.
- c) 0.03 antes de entrenar es imposible para un modelo nuevo: hay fuga de la etiqueta en las entradas, se cargaron pesos ya entrenados o hay un error en la evaluación (se evalúa sobre otra cosa o la pérdida está mal calculada).

## 5. Tamaño del lote (0.5 + 0.5)
- a) **Q** usa lotes de 16: su curva es mucho más ruidosa (cada paso ve 16 ejemplos), da 1250 pasos por época frente a 20 y tarda más por época (0.82 s frente a 0.14 s). Además termina con menor pérdida (0.34 frente a 0.60), porque dio 64 veces más pasos con el mismo η.
- b) Subir η aproximadamente en proporción al lote (regla lineal; en la práctica algo menos y con cuidado). El warmup empieza con η pequeño y lo sube durante los primeros pasos, para evitar que los primeros pasos grandes, con el modelo y las estadísticas del optimizador todavía sin ajustar, desestabilicen el entrenamiento.

## 6. Profundidad y weight decay (0.5 + 0.5)
- a) **No.** Con 40 capas la pérdida de **entrenamiento** es peor (0.53) que con 6 (0.24), y validación y entrenamiento van juntas (0.58 y 0.53): no hay brecha. Es un problema de optimización (degradación con la profundidad), no de sobreajuste. Dropout lo empeoraría. Lo que corresponde es usar conexiones residuales (y/o normalización) o menos capas.
- b) `Adam(weight_decay=λ)` suma $\lambda\theta$ al **gradiente**, y Adam lo divide por $\sqrt{\hat v}$: los parámetros con gradientes grandes reciben menos regularización efectiva, así que el decaimiento queda acoplado a la escala adaptativa. `AdamW` aplica el decaimiento **directamente a los pesos** ($\theta \leftarrow \theta - \eta\lambda\theta$), separado del paso adaptativo, y así es igual para todos los parámetros. Por eso es más predecible y suele generalizar mejor.
