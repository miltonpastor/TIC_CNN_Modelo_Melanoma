import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import pandas as pd
import os
from evaluation.plots import plot_calibration_curve, plot_roc_curve, plot_precision_recall_curve
from config.config import OUTPUT_FOLDER

def evaluate_model_without_threshold(model, test_generator):
    """Evalúa métricas que NO requieren umbral (automáticas en cada run).
    
    Args:
        model: Modelo de TensorFlow a evaluar
        test_generator: Generador de datos de prueba
    
    Returns:
        dict: Métricas calculadas (AUROC, AUPRC, Brier Score)
    """
    test_generator.reset()
    
    # Predicciones (probabilidades)
    y_pred_proba = model.predict(test_generator).flatten()
    y_true = test_generator.classes
    
    # Métricas sin umbral
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    # Gráficos
    plot_calibration_curve(y_true, y_pred_proba)
    plot_roc_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(y_true, y_pred_proba)
    
    # Guardar scores para análisis posterior
    save_scores(y_true, y_pred_proba, dataset_name='test')
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier_score
    }

def save_scores(y_true, y_pred_proba, dataset_name='test', output_folder=OUTPUT_FOLDER):
    """Guarda los scores de predicción para análisis posterior.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred_proba: Probabilidades predichas
        dataset_name: Nombre del dataset ('validation' o 'test')
    """
    scores_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_score': y_pred_proba
    })
    
    scores_path = os.path.join(output_folder, f'prediction_scores_{dataset_name}.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores ({dataset_name}) guardados en: {scores_path}")


def evaluate_model_with_threshold(model, test_generator, threshold=0.5):
    """
    Evalúa el modelo con un umbral específico.
    
    Args:
        model: Modelo de TensorFlow a evaluar
        test_generator: Generador de datos de prueba
        threshold: Umbral para clasificación (default: 0.5)
    
    Returns:
        tuple: (report, cm, accuracy, y_true, y_pred_proba)
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    test_generator.reset()
    
    # Predicciones (probabilidades)
    y_pred_proba = model.predict(test_generator).flatten()
    y_true = test_generator.classes
    
    # Aplicar umbral
    y_pred_classes = (y_pred_proba > threshold).astype(int)
    
    # Calcular métricas
    report = classification_report(
        y_true, 
        y_pred_classes,
        target_names=['Benign', 'Malignant'],
        output_dict=True
    )
    
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Calcular accuracy
    accuracy = (y_pred_classes == y_true).mean()
    
    return report, cm, accuracy, y_true, y_pred_proba