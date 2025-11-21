import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from evaluation.plots import plot_confusion_matrix

def evaluate_model(model, test_generator):
    """Evalúa el modelo y retorna métricas detalladas."""
    test_generator.reset()
    
    # Predicciones
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Métricas
    test_loss, test_acc, test_auc = model.evaluate(test_generator, verbose=0)
    
    # Reporte de clasificación
    report = classification_report(
        y_true, 
        y_pred_classes,
        target_names=['Benigno', 'Maligno'],
        output_dict=True
    )
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Graficar matriz de confusión
    plot_confusion_matrix(cm)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'auc': test_auc,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred_classes
    }