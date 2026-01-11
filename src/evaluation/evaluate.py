import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import numpy as np
import pandas as pd
import os
from evaluation.plots import plot_calibration_curve, plot_roc_curve, plot_precision_recall_curve
from config.config import OUTPUT_FOLDER

def evaluate_model_without_threshold(model, test_data):
    """Evalúa métricas que NO requieren umbral (automáticas en cada run).
    OPTIMIZADO: Lee scores desde CSV en chunks para evitar OOM.
    
    Args:
        model: Modelo de TensorFlow a evaluar
        test_data: tf.data.Dataset o generador de datos de prueba
    
    Returns:
        dict: Métricas calculadas (AUROC, AUPRC, Brier Score)
    """
    # Guardar scores de forma incremental (evita OOM) - retorna path del CSV
    scores_path = save_scores_incremental(model, test_data, dataset_name='test')
    
    # Leer scores desde CSV en chunks (evita cargar todo en memoria)
    print("\n📊 Calculando métricas desde CSV...")
    
    # Para métricas que requieren todos los datos, leer en chunks y calcular
    # Usamos iteradores para no cargar todo
    y_true = []
    y_pred_proba = []
    
    # Leer CSV en chunks de 10k muestras
    chunk_size = 10000
    for chunk in pd.read_csv(scores_path, chunksize=chunk_size):
        y_true.extend(chunk['true_label'].values)
        y_pred_proba.extend(chunk['predicted_score'].values)
    
    # Convertir a numpy arrays (ahora que ya tenemos todo)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Métricas sin umbral
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    # Gráficos
    plot_calibration_curve(y_true, y_pred_proba)
    plot_roc_curve(y_true, y_pred_proba)
    plot_precision_recall_curve(y_true, y_pred_proba)
    
    # Liberar memoria
    del y_true, y_pred_proba
    import gc
    gc.collect()
    
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


def save_scores_incremental(model, dataset, dataset_name='test', output_folder=OUTPUT_FOLDER, chunk_size=1000):
    """Guarda scores procesando y escribiendo en batches incrementales.
    OPTIMIZADO para evitar OOM: no acumula todo en memoria.
    
    Args:
        model: Modelo de TensorFlow para predicciones
        dataset: tf.data.Dataset o generador
        dataset_name: Nombre del dataset ('validation' o 'test')
        output_folder: Carpeta de salida
        chunk_size: Tamaño de chunk para liberar memoria (default: 1000 muestras)
    
    Returns:
        str: Ruta del archivo CSV guardado (para cargar después si es necesario)
    """
    scores_path = os.path.join(output_folder, f'prediction_scores_{dataset_name}.csv')
    
    # Chunk temporal para escribir periódicamente (evita crear DataFrames grandes)
    chunk_true = []
    chunk_pred = []
    
    # Procesar batch por batch
    batch_count = 0
    total_samples = 0
    
    # Crear/truncar archivo con header
    with open(scores_path, 'w') as f:
        f.write('true_label,predicted_score\n')
    
    print(f"\n💾 Guardando scores de {dataset_name} incrementalmente...")
    
    for batch_x, batch_y in dataset:
        batch_count += 1
        
        # Predecir batch actual
        batch_pred = model.predict(batch_x, verbose=0).flatten()
        batch_true = batch_y.numpy()
        
        # Acumular en chunk temporal
        chunk_true.extend(batch_true)
        chunk_pred.extend(batch_pred)
        total_samples += len(batch_true)
        
        # Escribir chunk al CSV cuando alcance el tamaño límite
        if len(chunk_true) >= chunk_size:
            batch_df = pd.DataFrame({
                'true_label': chunk_true,
                'predicted_score': chunk_pred
            })
            batch_df.to_csv(scores_path, mode='a', header=False, index=False)
            
            # Limpiar chunk (liberar memoria)
            chunk_true = []
            chunk_pred = []
            
            # Limpiar sesión de Keras
            import gc
            gc.collect()
            tf.keras.backend.clear_session()
        
        # Progress cada 50 batches
        if batch_count % 50 == 0:
            print(f"  Procesados {batch_count} batches ({total_samples} muestras)...")
    
    # Escribir chunk final (si quedó algo)
    if chunk_true:
        batch_df = pd.DataFrame({
            'true_label': chunk_true,
            'predicted_score': chunk_pred
        })
        batch_df.to_csv(scores_path, mode='a', header=False, index=False)
    
    print(f"✅ Scores ({dataset_name}) guardados en: {scores_path}")
    print(f"   Total: {total_samples} muestras en {batch_count} batches")
    
    # Limpiar memoria final
    import gc
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Retornar path (no arrays grandes en memoria)
    return scores_path


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