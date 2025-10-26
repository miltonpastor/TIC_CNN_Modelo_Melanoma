# src/config.py
import os

# Rutas
CSV_PATH = '../data/bcn20000_metadata_2025-07-22.csv'
#IMAGES_FOLDER = '/content/drive/MyDrive/DatasetTIC/ISIC-images'
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
