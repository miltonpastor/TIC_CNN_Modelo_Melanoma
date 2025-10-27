import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model

def build_resnet50_classifier(input_shape=(224, 224, 3), 
                               dropout_rate=0.3,
                               dense_units=128,
                               num_classes=1):
    """
    Construye modelo ResNet50 para clasificación de melanoma.
    
    Args:
        input_shape: Dimensiones de entrada
        dropout_rate: Tasa de dropout (0.3-0.5)
        dense_units: Unidades en capa densa
        num_classes: 1 para binario, >1 para multiclase
    """
    base = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = BatchNormalization()(x)
    
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    output = Dense(num_classes, activation=activation)(x)
    
    model = Model(inputs=base.input, outputs=output)
    
    return model, base

def freeze_base(base_model):
    """Congela la base para head training."""
    base_model.trainable = False

def unfreeze_last_n_layers(base_model, n_layers=20):
    """Descongela las últimas N capas para fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False