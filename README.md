# TIC_CNN_Modelo_Melanoma

Proyecto para clasificación binaria de imágenes de melanoma usando una CNN (ResNet50, Keras).

## Estructura principal

- `data/`: Datos originales y procesados
- `src/`: Código fuente modular (configuración, carga, entrenamiento, evaluación)
- `notebooks/`: Jupyter Notebooks para cada etapa del pipeline
- `outputs/`: Resultados y modelos entrenados

## Uso básico

1. Configura el entorno en `src/config/config.py` (`ENV = "local"` o `"colab"`)
2. Si no estás usando Colab, instala los paquetes del archivo requirements.txt.
3. Ejecuta los notebooks en orden:

- 01_data_preparation.ipynb
- 02_preprocessing.ipynb
- 03_training_evaluation.ipynb

4. Alternativamente, puedes ejecutar `main.py` desde la carpeta `src`.
5. Los modelos y resultados se guardan en `outputs/`

## Estructura de Outputs

Cada ejecución genera una carpeta timestampeada en `outputs/`:

```
outputs/
  resnet50_YYYYMMDD_HHMMSS/
    ├── best_model.h5           # Modelo final entrenado
    ├── results.json             # Métricas y configuración
    ├── training_curves.png      # Gráficos de entrenamiento
    ├── csv_splits/              # Splits de datos
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── logs/                    # TensorBoard logs
        ├── head_training/
        └── fine_tuning/
```

**Archivos principales:**

- `best_model.h5`: Modelo con mejor AUC en validación
- `results.json`: Accuracy, AUC, loss, configuración y mapeo de etiquetas
- `logs/`: Usar con `tensorboard --logdir=outputs/resnet50_*/logs`

## Requisitos

- Python 3.11.14
- TensorFlow, pandas, scikit-learn

## Contacto

Autor: miltonpastor
