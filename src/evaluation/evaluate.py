import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import pandas as pd
import os
from evaluation.plots import plot_calibration_curve, plot_roc_curve, plot_precision_recall_curve
from config.config import OUTPUT_FOLDER

def evaluate_model_without_threshold(model, test_data):
    """Evalúa métricas que NO requieren umbral (automáticas en cada run).
    
    Args:
        model: Modelo de TensorFlow a evaluar
        test_data: tf.data.Dataset o generador de datos de prueba
    
    Returns:
        dict: Métricas calculadas (AUROC, AUPRC, Brier Score)
    """
    # Guardar scores de forma incremental (evita OOM)
    y_true, y_pred_proba = save_scores_incremental(model, test_data, dataset_name='test')
    
    # Métricas sin umbral
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    # Gráficos
    plot_calibration_curve(y_true, y_pred_proba)
    plot_roc_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(y_true, y_pred_proba)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier_score
    }

def save_scores(y_true, y_pred_proba, dataset_name='test', output_folder=OUTPUT_FOLDER):
    """Guarda los scores de predicción para análisis posterior.
    
    Args:
        y_true: Etiquetas verdaderas (numpy array)
        y_pred_proba: Probabilidades predichas (numpy array)
        dataset_name: Nombre del dataset ('validation' o 'test')
    """
    scores_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_score': y_pred_proba
    })
    
    scores_path = os.path.join(output_folder, f'prediction_scores_{dataset_name}.csv')
    scores_df.to_csv(scores_path, index=False)
    print(f"Scores ({dataset_name}) guardados en: {scores_path}")


def save_scores_incremental(model, dataset, dataset_name='test', output_folder=OUTPUT_FOLDER):
    """Guarda scores procesando y escribiendo en batches incrementales.
    Evita cargar todo el dataset en memoria (previene OOM en EC2).
    
    Args:
        model: Modelo de TensorFlow para predicciones
        dataset: tf.data.Dataset o generador
        dataset_name: Nombre del dataset ('validation' o 'test')
        output_folder: Carpeta de salida
    
    Returns:
        tuple: (y_true_array, y_pred_array) para calcular métricas
    """
    scores_path = os.path.join(output_folder, f'prediction_scores_{dataset_name}.csv')
    
    # Listas para acumular (más eficiente que numpy para append)
    all_true = []
    all_pred = []
    
    # Procesar batch por batch
    batch_count = 0
    
    # Crear/truncar archivo con header
    with open(scores_path, 'w') as f:
        f.write('true_label,predicted_score\n')
    
    print(f"\n💾 Guardando scores de {dataset_name} incrementalmente...")
    
    for batch_x, batch_y in dataset:
        batch_count += 1
        
        # Predecir batch actual
        batch_pred = model.predict(batch_x, verbose=0).flatten()
        batch_true = batch_y.numpy()
        
        # Guardar batch actual al CSV (append mode)
        batch_df = pd.DataFrame({
            'true_label': batch_true,
            'predicted_score': batch_pred
        })
        batch_df.to_csv(scores_path, mode='a', header=False, index=False)
        
        # Acumular para retornar (para métricas)
        all_true.extend(batch_true)
        all_pred.extend(batch_pred)
        
        # Progress cada 50 batches
        if batch_count % 50 == 0:
            print(f"  Procesados {batch_count} batches ({len(all_true)} muestras)...")
        
        # Limpiar memoria cada 100 batches
        if batch_count % 100 == 0:
            tf.keras.backend.clear_session()
    
    print(f"✅ Scores ({dataset_name}) guardados en: {scores_path}")
    print(f"   Total: {len(all_true)} muestras en {batch_count} batches")
    
    # Retornar arrays para cálculo de métricas
    return np.array(all_true), np.array(all_pred)


def evaluate_model_with_threshold(model, test_data, threshold=0.5):
    """
    Evalúa el modelo con un umbral específico.
    Optimizado para evitar OOM procesando en batches.
    
    Args:
        model: Modelo de TensorFlow a evaluar
        test_data: tf.data.Dataset o generador de datos de prueba
        threshold: Umbral para clasificación (default: 0.5)
    
    Returns:
        tuple: (report, cm, accuracy, y_true, y_pred_proba)
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Procesar en batches para evitar OOM
    all_true = []
    all_pred = []
    
    print("\\n🔍 Procesando predicciones en batches...")
    for batch_count, (batch_x, batch_y) in enumerate(test_data):
        batch_pred = model.predict(batch_x, verbose=0).flatten()
        batch_true = batch_y.numpy()
        
        all_true.extend(batch_true)
        all_pred.extend(batch_pred)
        
        if (batch_count + 1) % 50 == 0:
            print(f"  Procesados {batch_count + 1} batches...")
    
    # Convertir a numpy arrays
    y_true = np.array(all_true)
    y_pred_proba = np.array(all_pred)
    
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