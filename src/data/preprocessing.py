import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from config.config import (
    IMAGE_SIZE, 
    BATCH_SIZE,
    USE_AUGMENTATION,
    ROTATION_RANGE,
    ZOOM_RANGE,
    WIDTH_SHIFT_RANGE,
    HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP,
    VERTICAL_FLIP,
    BRIGHTNESS_RANGE,
    CLASS_BALANCE_CONFIG
)
import pandas as pd

def create_data_generators():
    """
    Crea generadores de datos con augmentation para train y sin augmentation para val/test.
    
    Returns:
        tuple: (train_datagen, val_test_datagen)
    """
    if USE_AUGMENTATION:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,  # Normalización ImageNet
            rotation_range=ROTATION_RANGE,
            zoom_range=ZOOM_RANGE,
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
            vertical_flip=VERTICAL_FLIP,
            brightness_range=BRIGHTNESS_RANGE,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    
    # Para validación y test solo normalización, sin augmentation
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    
    return train_datagen, val_test_datagen


def create_data_flow_from_dataframe(datagen, dataframe, batch_size=BATCH_SIZE, shuffle=True):
    """
    Crea un flujo de datos desde un DataFrame que alimenta un modelo Keras.
    
    Args:
        datagen: ImageDataGenerator
        dataframe: DataFrame con columnas 'filepath' y 'label'
        batch_size: Tamaño del batch
        shuffle: Si se debe mezclar los datos
        
    Returns:
        DirectoryIterator
    """
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode='binary',  # Clasificación binaria
        shuffle=shuffle
    )


def oversample_minority_class(dataframe):
    """
    Sobremuestrea la clase minoritaria replicando sus filas.
    Cada réplica tendrá augmentation diferente durante el entrenamiento.
    
    Args:
        dataframe: DataFrame con columnas 'filepath' y 'label'
        
    Returns:
        DataFrame balanceado con oversampling de clase minoritaria
    """
    if not CLASS_BALANCE_CONFIG['use_oversampling']:
        print("⚠️  Oversampling desactivado")
        return dataframe
    
    minority_class = CLASS_BALANCE_CONFIG['minority_class']
    ratio = CLASS_BALANCE_CONFIG['oversample_ratio']
    
    # Separar clases
    df_minority = dataframe[dataframe['label'] == str(minority_class)].copy()
    df_majority = dataframe[dataframe['label'] != str(minority_class)].copy()
    
    original_minority = len(df_minority)
    original_majority = len(df_majority)
    
    # Calcular cuántas réplicas necesitamos
    target_minority = int(original_minority * ratio)
    
    # Oversamplear (con reemplazo para permitir duplicados)
    df_minority_oversampled = df_minority.sample(
        n=target_minority, 
        replace=True, 
        random_state=42
    )
    
    # Combinar
    df_balanced = pd.concat([df_majority, df_minority_oversampled], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"📊 Oversampling aplicado:")
    print(f"   Clase mayoritaria: {original_majority} muestras")
    print(f"   Clase minoritaria: {original_minority} → {target_minority} muestras ({ratio}x)")
    print(f"   Total: {len(dataframe)} → {len(df_balanced)} muestras")
    print(f"   Nuevo ratio: {original_majority/target_minority:.2f}:1")
    
    return df_balanced


def load_and_preprocess_image(image_path):
    """
    Carga y preprocesa una imagen individual para inferencia.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        np.array: Imagen preprocesada lista para el modelo
    """
    # Cargar imagen
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=IMAGE_SIZE
    )
    
    # Convertir a array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Expandir dimensiones para batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocesar con ResNet50
    img_array = preprocess_input(img_array)
    
    return img_array


"""
def remove_hair_artifacts(image):
    # TODO: por implementar - limpieza de artefactos de pelos
    pass
"""


"""
def preprocess_with_cleaning(image_path):
    # TODO: por implementar - preprocesamiento con limpieza de artefactos
    pass
"""
