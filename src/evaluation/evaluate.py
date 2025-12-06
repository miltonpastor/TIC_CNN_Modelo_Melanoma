import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import pandas as pd
from evaluation.plots import plot_confusion_matrix, plot_calibration_curve, plot_roc_curve, plot_precision_recall_curve

def evaluate_model(model, test_generator):
    """Evalúa el modelo y retorna métricas detalladas."""
    test_generator.reset()
    
    # Predicciones
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Métricas (el modelo retorna: loss, accuracy, auc)
    test_loss, test_acc, test_auc_keras = model.evaluate(test_generator, verbose=0)
    
    # Calcular AUC-ROC y PR-AUC con sklearn (usando probabilidades)
    y_pred_proba = y_pred.flatten()
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
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
    
    # Graficar curva de calibración
    plot_calibration_curve(y_true, y_pred_proba)
    
    # Graficar curva ROC
    plot_roc_curve(y_true, y_pred_proba)
    
    # Graficar curva Precision-Recall
    plot_precision_recall_curve(y_true, y_pred_proba)
    
    return {
        'accuracy': test_acc,
        'loss': test_loss,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier_score,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred_classes,
        'y_pred_proba': y_pred_proba
    }