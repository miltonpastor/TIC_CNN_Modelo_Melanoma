import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from src.config.config import (
    IMAGE_SIZE, 
    BATCH_SIZE,
    USE_AUGMENTATION,
    ROTATION_RANGE,
    ZOOM_RANGE,
    WIDTH_SHIFT_RANGE,
    HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP,
    VERTICAL_FLIP
)

def create_data_generators():
    """
    Crea generadores de datos con augmentation para train y sin augmentation para val/test.
    Usa preprocess_input de ResNet50 para normalización ImageNet.
    
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
    Crea un flow de datos desde un DataFrame.
    
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


def remove_hair_artifacts(image):
    """
    Elimina artefactos de pelos de imágenes dermatológicas usando morfología.
    Técnica opcional para limpieza de imágenes.
    
    Args:
        image: np.array de la imagen
        
    Returns:
        np.array: Imagen sin artefactos de pelos
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detectar pelos usando blackhat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold para crear máscara
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpainting para rellenar áreas de pelos
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return result


def preprocess_with_cleaning(image_path):
    """
    Carga, limpia y preprocesa una imagen con eliminación de artefactos.
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        np.array: Imagen preprocesada y limpia
    """
    # Cargar imagen
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Limpiar artefactos
    img_clean = remove_hair_artifacts(img)
    
    # Redimensionar
    img_resized = cv2.resize(img_clean, IMAGE_SIZE)
    
    # Expandir dimensiones para batch
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Preprocesar con ResNet50
    img_array = preprocess_input(img_array)
    
    return img_array
