import tensorflow as tf
from tensorflow.keras.optimizers import AdamW

def get_optimizer(config):
    """
    Retorna el optimizador configurado.
    Soporta: Adam, AdamW, SAM (si disponible)
    """
    opt_name = config['optimizer'].lower()
    lr = config['learning_rate']
    
    if opt_name == 'adamw':
        return AdamW(
            learning_rate=lr,
            weight_decay=config.get('weight_decay', 0.01)
        )
    elif opt_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    # elif opt_name == 'sam':
    #     # Implementación de SAM si tu infraestructura lo permite
    #     pass
    else:
        raise ValueError(f"Optimizador {opt_name} no soportado")