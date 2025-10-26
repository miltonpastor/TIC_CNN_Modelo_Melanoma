# src/config.py
import os

# Rutas
CSV_PATH = '/content/drive/MyDrive/DatasetTIC/bcn20000_metadata_2025-07-22.csv'
IMAGES_FOLDER = '/content/drive/MyDrive/DatasetTIC/ISIC-images'
OUTPUT_FOLDER = '/content/drive/MyDrive/Modelo_CNN'
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
# LABEL_MAPPING = {
#     'Benign': 0,
#     'Malignant': 1
# }
