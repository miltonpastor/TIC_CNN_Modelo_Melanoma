# src/config.py
import os
from datetime import datetime

ENV = "local"

# Data loading mode: 'csv' or 'predivided'
# 'csv': Load from CSV and split automatically
# 'predivided': Load from pre-divided train.txt, validation.txt, test.txt
DATA_MODE = 'predivided'
MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'resnet50v2', 'efficientnet-b0', 'densenet121'

# ----- SI DATA_MODE es 'csv'
if ENV == "Colab":
    CSV_PATH = '/content/TIC_CNN_Modelo_Melanoma/data/bcn20000_metadata_2025-07-22.csv'
    IMAGES_FOLDER = '/content/drive/MyDrive/DatasetTIC/ISIC-images'
else:
    CSV_PATH = 'data/bcn20000_metadata_2025-07-22.csv'
    IMAGES_FOLDER = 'data/ISIC-images/'

# Columnas CSV
ID_COLUMN = 'isic_id'
DIAGNOSIS_COLUMN = 'diagnosis_1'
# Tamaño de muestra
SAMPLE_SIZE = 30  # Cambiar a None para usar TODO el dataset

# ----- SI DATA_MODE es 'predivided'
# Folder for pre-divided dataset lists
LISTS_FOLDER = 'data/lists'
# Number of training samples to use (None = use all)
# Validation and test will be proportional
TRAIN_SAMPLE_SIZE = None  # e.g., 10000 for 10k training samples


if ENV == "Colab":
    BASE_OUTPUT_FOLDER = '/content/TIC_CNN_Modelo_Melanoma/outputs'
elif ENV == "local":
    BASE_OUTPUT_FOLDER = 'outputs'

# Carpeta de ejecución con timestamp
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"{MODEL_NAME}_{RUN_TIMESTAMP}")
CSV_SPLIT_FOLDER = os.path.join(OUTPUT_FOLDER, "csv_splits")

# Semilla y proporciones
RANDOM_SEED = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Etiquetas binarias
LABEL_MAPPING = {
    'Benign': 0,
    'Malignant': 1
}

# Configuración de preprocesamiento de imágenes
# ResNet50/ResNet50V2/EfficientNet-B0/DenseNet121 usan 224x224
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# Batch size 
BATCH_SIZE = 32 

# Normalización (ImageNet mean y std para ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation
USE_AUGMENTATION = True
ROTATION_RANGE = 40
ZOOM_RANGE = 0.25
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = True
BRIGHTNESS_RANGE = [0.8, 1.2]
SHEAR_RANGE = 0.15

# Configuración del modelo
MODEL_CONFIG = {
    'input_shape': INPUT_SHAPE,
    'dropout_rate': 0.5,
    'dense_units': 128,
    'num_classes': 1,
}

# Configuración de balanceo de clases
CLASS_BALANCE_CONFIG = {
    'use_oversampling': False,  # Activar oversampling de clase minoritaria
    'minority_class': 1,  # Clase a oversamplear (1 = Malignant)
    'oversample_ratio': 3.0,  # Multiplicador: 3x más ejemplos de malignos
    'use_focal_loss': False,  # Usar Focal Loss en lugar de BCE
    'focal_gamma': 2.0,  # Parámetro gamma de focal loss
    'focal_alpha': 0.25,  # Parámetro alpha de focal loss (peso clase positiva)
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    'head_epochs': 5,
    'finetune_epochs': 20,
    'unfreeze_layers': 30,
    'initial_lr_head': 1e-3,
    'initial_lr_finetune': 5e-5
}

# Configuración de GPU (optimizado para Tesla T4)
GPU_CONFIG = {
    'mixed_precision': True,  # Usar mixed precision (FP16) para ahorrar memoria y acelerar 3x
    'memory_growth': True,  # Permitir crecimiento dinámico de memoria GPU
    'xla': False,  # XLA desactivado (puede causar problemas de compilación)
    # NOTA: prefetch y num_parallel_calls usan tf.data.AUTOTUNE por defecto para mejor rendimiento
    # Los valores abajo son solo de referencia (AUTOTUNE ajusta dinámicamente según CPU/GPU/memoria)
    'prefetch_buffer_size': 'AUTOTUNE',  # Prefetch automático (TensorFlow optimiza)
    'num_parallel_calls': 'AUTOTUNE',  # Paralelismo automático (TensorFlow optimiza)
}
