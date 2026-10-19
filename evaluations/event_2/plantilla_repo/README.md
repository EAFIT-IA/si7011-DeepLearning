# SI7011 · Evento evaluativo 2 · Parte B

**Integrantes:** _nombre 1_ (`usuario-github-1`) · _nombre 2_ (`usuario-github-2`)

## Contenido

| Ruta | Qué es |
|---|---|
| [`docs/analisis.md`](docs/analisis.md) | El documento de análisis |
| [`notebook/sesion_03_6_integrador_eurosat.ipynb`](notebook/sesion_03_6_integrador_eurosat.ipynb) | El ejercicio integrador ejecutado, con los 4 TODO |
| `mlflow/mlflow.db`, `mlflow/mlruns/` | Las corridas registradas en MLflow |

## Cómo reproducir

1. Abrir el notebook en Kaggle o Colab, con GPU, y agregar el dataset EuroSAT (Kaggle: `apollo2506/eurosat-dataset`).
2. Ejecutarlo de principio a fin. Al terminar, copiar `mlflow.db` y `mlruns/` a la carpeta `mlflow/`.

Para ver las corridas, desde la raíz del repositorio:

```bash
cd mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Corrida elegida

_Nombre de la corrida de MLflow elegida por validación y su resultado en prueba (una sola vez)._
