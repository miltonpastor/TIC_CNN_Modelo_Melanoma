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
    ├── logs/                    # TensorBoard logs
    │   ├── head_training/
    │   └── fine_tuning/
    └── openvino/                # Modelo convertido a OpenVINO
        ├── best_model.xml
        └── best_model.bin
```

**Archivos principales:**

- `best_model.h5`: Modelo con mejor AUC en validación
- `results.json`: Accuracy, AUC, loss, configuración y mapeo de etiquetas
- `logs/`: Usar con `tensorboard --logdir=outputs/resnet50_*/logs`

## Continuar Entrenamiento

Si necesitas entrenar más épocas desde un modelo ya entrenado (sin repetir todo el proceso):

1. Abre `scripts/continue_training.py` y configura:
   - `PRETRAINED_MODEL_PATH`: Ruta al modelo .h5 que quieres continuar
   - `ADDITIONAL_EPOCHS`: Cuántas épocas más quieres entrenar
   - `CONTINUE_LR`: Learning rate (por defecto 1e-5, más bajo que el entrenamiento inicial)

2. Ejecuta:

```bash
python scripts/continue_training.py
```

3. Resultados en `outputs/resnet50_continued_YYYYMMDD_HHMMSS/`:
   - `best_model.h5`: Modelo con épocas adicionales
   - `results.json`: **Métricas finales actualizadas** (usar estas)
   - `continue_info.json`: Información del entrenamiento continuado

**Nota:** El script NO repite el head training, solo continúa el fine-tuning desde donde quedó.

## Conversión a OpenVINO

Para convertir el modelo .h5 a formato OpenVINO (optimizado para inferencia):

```bash
python scripts/convert_to_openvino.py outputs/resnet50_YYYYMMDD_HHMMSS/best_model.h5
```

Esto generará los archivos `best_model.xml` y `best_model.bin` en la carpeta `openvino/` dentro del directorio del modelo.

## Requisitos

- Python 3.11.14
- TensorFlow, pandas, scikit-learn

## Contacto

Autor: miltonpastor
