import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model

def build_cnn_classifier(arch,
                          input_shape, 
                          dropout_rate,
                          dense_units,
                          num_classes):
    """
    Construye modelo CNN para clasificación de melanoma.
    
    Args:
        arch: Arquitectura base ('resnet50', 'resnet50v2', 'efficientnet-b0', 'densenet121')
        input_shape: Dimensiones de entrada
        dropout_rate: Tasa de dropout (0.3-0.5)
        dense_units: Unidades en capa densa
        num_classes: 1 para binario, >1 para multiclase
    """

    # Selección de arquitectura base
    if arch == "resnet50":
        base = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_fn = tf.keras.applications.resnet.preprocess_input

    elif arch == "resnet50v2":
        base = tf.keras.applications.ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input

    elif arch == "efficientnet-b0":
        base = tf.keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    elif arch == "densenet121":
        base = tf.keras.applications.DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        preprocess_fn = tf.keras.applications.densenet.preprocess_input

    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

    # Head de clasificación
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    output = Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs=base.input, outputs=output)
    
    return model, base, preprocess_fn

def freeze_base(base_model):
    """Congela la base para head training."""
    base_model.trainable = False

def unfreeze_last_n_layers(base_model, n_layers=20):
    """Descongela las últimas N capas para fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False