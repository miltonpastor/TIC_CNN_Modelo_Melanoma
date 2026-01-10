import tensorflow as tf
from tensorflow.keras import layers
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
    CLASS_BALANCE_CONFIG,
    GPU_CONFIG
)
import pandas as pd


def get_augmentation_layers():
    """
    Crea capas de augmentation que se ejecutan en GPU.
    Usa tf.keras.layers.RandomFlip, RandomRotation, etc.
    
    Returns:
        tf.keras.Sequential: Modelo de augmentation ejecutable en GPU
    """
    if not USE_AUGMENTATION:
        return None
    
    augmentation_layers = [
        layers.RandomFlip("horizontal" if HORIZONTAL_FLIP else ""),
        layers.RandomFlip("vertical" if VERTICAL_FLIP else ""),
        layers.RandomRotation(factor=ROTATION_RANGE / 360.0),  # Convertir grados a fracción
        layers.RandomZoom(
            height_factor=(-ZOOM_RANGE, ZOOM_RANGE),
            width_factor=(-ZOOM_RANGE, ZOOM_RANGE)
        ),
        layers.RandomTranslation(
            height_factor=HEIGHT_SHIFT_RANGE,
            width_factor=WIDTH_SHIFT_RANGE
        ),
        layers.RandomBrightness(
            factor=(BRIGHTNESS_RANGE[0] - 1.0, BRIGHTNESS_RANGE[1] - 1.0)
        ),
    ]
    
    # Filtrar capas vacías (si alguna configuración está desactivada)
    augmentation_layers = [l for l in augmentation_layers if l is not None]
    
    return tf.keras.Sequential(augmentation_layers, name="data_augmentation")


@tf.function
def load_and_preprocess_image_tf(filepath, label, preprocess_fn, target_size=IMAGE_SIZE):
    """
    Carga y preprocesa una imagen usando tf.io (ejecutable en GPU).
    
    Args:
        filepath: Tensor con ruta de archivo
        label: Tensor con etiqueta
        preprocess_fn: Función de preprocesamiento del modelo (ResNet, EfficientNet, etc.)
        target_size: Tamaño objetivo de la imagen
        
    Returns:
        tuple: (imagen_preprocesada, label)
    """
    # Leer archivo
    img = tf.io.read_file(filepath)
    
    # Decodificar imagen (automáticamente detecta JPEG/PNG)
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Redimensionar
    img = tf.image.resize(img, target_size)
    
    # Preprocesamiento específico del modelo (ImageNet normalization)
    img = preprocess_fn(img)
    
    return img, label


def create_tf_dataset(dataframe, preprocess_fn, batch_size=BATCH_SIZE, 
                      shuffle=True, augment=False, cache=True, 
                      prefetch_buffer=None, num_parallel_calls=None):
    """
    Crea un tf.data.Dataset optimizado para GPU.
    
    Args:
        dataframe: DataFrame con columnas 'filepath' y 'label'
        preprocess_fn: Función de preprocesamiento del modelo base
        batch_size: Tamaño del batch (default: de config.BATCH_SIZE)
        shuffle: Si se debe mezclar (True para train, False para val/test)
        augment: Si se debe aplicar data augmentation (solo train)
        cache: Si se debe cachear en memoria (recomendado para datasets pequeños)
        prefetch_buffer: Tamaño del buffer de prefetch (None = usar config.GPU_CONFIG)
        num_parallel_calls: Número de llamadas paralelas (None = usar config.GPU_CONFIG)
        
    Returns:
        tf.data.Dataset: Pipeline optimizado
    """
    # Usar configuraciones de GPU_CONFIG si no se especifican
    if prefetch_buffer is None:
        # AUTOTUNE para prefetch es mejor que un valor fijo
        prefetch_buffer = tf.data.AUTOTUNE
    
    if num_parallel_calls is None:
        # Usar AUTOTUNE para mejor rendimiento (config es solo referencia)
        num_parallel_calls = tf.data.AUTOTUNE
    # Convertir DataFrame a listas
    filepaths = dataframe['filepath'].values
    labels = dataframe['label'].astype(np.float32).values
    
    # Crear dataset desde tensors
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Shuffle si es necesario (antes de cargar imágenes para ahorrar memoria)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 10000), 
                                  reshuffle_each_iteration=True)
    
    # Cargar y preprocesar imágenes en paralelo
    # AUTOTUNE permite a TensorFlow optimizar automáticamente el paralelismo
    dataset = dataset.map(
        lambda fp, lbl: load_and_preprocess_image_tf(fp, lbl, preprocess_fn),
        num_parallel_calls=num_parallel_calls
    )
    
    # Cache en memoria (acelera lecturas repetidas)
    # Para datasets grandes (>10GB), considerar cache en disco: cache('/tmp/cache')
    if cache:
        dataset = dataset.cache()
    
    # Batch antes de augmentation para que la GPU procese batches completos
    dataset = dataset.batch(batch_size)
    
    # Aplicar augmentation en GPU (después de batch para procesamiento paralelo)
    if augment:
        aug_model = get_augmentation_layers()
        if aug_model is not None:
            dataset = dataset.map(
                lambda x, y: (aug_model(x, training=True), y),
                num_parallel_calls=num_parallel_calls
            )
    
    # Prefetch: permite que la GPU entrene mientras la CPU prepara el siguiente batch
    # CRÍTICO para mantener GPU ocupada al 100%
    dataset = dataset.prefetch(buffer_size=prefetch_buffer)
    
    return dataset


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