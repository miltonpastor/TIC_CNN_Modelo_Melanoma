# src/config.py
import os

ENV = os.environ.get("TIC_CNN_ENV", "local")

if ENV == "Colab":
    CSV_PATH = '/content/TIC_CNN_MODELO_MELANOMA/data/bcn20000_metadata_2025-07-22.csv'
    IMAGES_FOLDER = '/content/drive/MyDrive/DatasetTIC/ISIC-images'
    OUTPUT_FOLDER = '/content/TIC_CNN_MODELO_MELANOMA/outputs'
else:
    CSV_PATH = '../data/bcn20000_metadata_2025-07-22.csv'
    IMAGES_FOLDER = '../data/ISIC-images'
    OUTPUT_FOLDER = '../outputs'


CSV_SPLIT_FOLDER = os.path.join(OUTPUT_FOLDER, "csv_splits")

# Columnas CSV
ID_COLUMN = 'isic_id'
DIAGNOSIS_COLUMN = 'diagnosis_1'

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
IMAGE_SIZE = (224, 224)  # Tamaño para ResNet50
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

# Normalización (ImageNet mean y std para ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation
USE_AUGMENTATION = True
ROTATION_RANGE = 20
ZOOM_RANGE = 0.15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False

# Configuración del modelo
MODEL_CONFIG = {
    'input_shape': INPUT_SHAPE,
    'dropout_rate': 0.3,
    'dense_units': 128,
    'num_classes': 1  # Binario
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    'head_epochs': 10,
    'finetune_epochs': 20,
    'unfreeze_layers': 20
}

# Tamaño de muestra
SAMPLE_SIZE = 1000  # None para usar todo el dataset, o un entero para muestrear