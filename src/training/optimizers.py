import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss para clasificación binaria.
    Reduce el peso de ejemplos fáciles y se enfoca en ejemplos difíciles.
    
    Args:
        gamma: Factor de enfoque (default: 2.0). Mayor = más enfoque en difíciles.
        alpha: Factor de balanceo de clases (default: 0.25 para clase positiva).
    
    Returns:
        Función de pérdida para usar en model.compile()
    
    Referencia: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip para estabilidad numérica
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calcular cross entropy
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Calcular focal weight
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = K.pow(1 - pt, gamma)
        
        # Aplicar alpha balancing
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Focal loss final
        loss = alpha_weight * focal_weight * cross_entropy
        
        return K.mean(loss)
    
    return focal_loss_fixed


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