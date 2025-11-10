from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def calculate_class_weights(labels):
    """
    Calcula los pesos de clase para balancear un dataset desbalanceado.
    
    Args:
        labels: Array o Series con las etiquetas (0, 1, etc.)
        
    Returns:
        dict: Diccionario {clase: peso} para usar en model.fit()
        
    Ejemplo:
        >>> weights = calculate_class_weights(train_df['label'])
        >>> {0: 0.56, 1: 5.0}  # Más peso a la clase minoritaria
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print(f"📊 Class weights calculados: {class_weight_dict}")
    
    return class_weight_dict
