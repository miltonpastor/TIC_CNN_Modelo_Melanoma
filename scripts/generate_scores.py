#!/usr/bin/env python3
"""
Script para generar scores de predicción desde un modelo ya entrenado.
Útil para análisis posterior con notebooks.
"""
import os
import sys
import tensorflow as tf
import numpy as np
import json

# ============================================================================
# CONFIGURACIÓN - Modifica estos valores según tus necesidades
# ============================================================================
RUN_DIR = 'densenet121_20260111_090433'  # Nombre del directorio del run
DATASET = 'test'                 # 'validation' o 'test'
# ============================================================================

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.evaluate import save_scores
from data.data_loader import load_predivided_data
from data.preprocessing import create_tf_dataset

def generate_scores(run_dir, dataset='validation'):
    """
    Genera y guarda scores de predicción desde un modelo entrenado.
    
    Args:
        run_dir: Directorio del run (ej: outputs/resnet50_20251212_082430)
        dataset: Dataset a procesar ('validation' o 'test')
    """
    # Cargar el modelo
    model_path = os.path.join(run_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return
    
    print(f"📦 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Cargar configuración del modelo para obtener la arquitectura
    results_path = os.path.join(run_dir, 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
            arch = results.get('architecture', 'resnet50')
    else:
        print("⚠️ No se encontró results.json, usando arquitectura por defecto: resnet50")
        arch = 'resnet50'
    
    # Obtener función de preprocesamiento según la arquitectura
    if arch == "resnet50":
        preprocess_fn = tf.keras.applications.resnet.preprocess_input
    elif arch == "resnet50v2":
        preprocess_fn = tf.keras.applications.resnet_v2.preprocess_input
    elif arch == "efficientnet-b0":
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
    elif arch == "densenet121":
        preprocess_fn = tf.keras.applications.densenet.preprocess_input
    else:
        print(f"⚠️ Arquitectura desconocida '{arch}', usando ResNet50 por defecto")
        preprocess_fn = tf.keras.applications.resnet.preprocess_input
    
    # Cargar datos (DataFrames)
    print(f"📊 Cargando datos de {dataset}...")
    train_df, val_df, test_df = load_predivided_data()
    
    # Seleccionar el DataFrame correcto
    if dataset == 'validation':
        df = val_df
    elif dataset == 'test':
        df = test_df
    else:
        print(f"❌ Error: Dataset '{dataset}' no válido. Usa 'validation' o 'test'")
        return
    
    # Crear tf.data.Dataset (sin augmentation, sin shuffle para evaluación)
    print(f"🔧 Creando pipeline de datos para {len(df)} muestras...")
    data = create_tf_dataset(
        df,
        preprocess_fn=preprocess_fn,
        shuffle=False,
        augment=False,
        cache=False  # No cachear para datasets de evaluación grandes
    )
    
    # Generar predicciones
    print(f"🔍 Generando predicciones...")
    predictions = model.predict(data, verbose=1)
    
    # Obtener las etiquetas verdaderas del DataFrame
    true_labels = df['label'].astype(int).values
    
    # Guardar scores
    save_scores(true_labels, predictions.flatten(), dataset_name=dataset, output_folder=run_dir)
    
    print(f"\n✅ Scores guardados exitosamente en: {run_dir}/prediction_scores_{dataset}.csv")
    print(f"📊 Total de muestras procesadas: {len(true_labels)}")
    benign_count = int(np.count_nonzero(true_labels == 0))
    malignant_count = int(np.count_nonzero(true_labels == 1))
    print(f"   Clase 0 (Benign): {benign_count}")
    print(f"   Clase 1 (Malignant): {malignant_count}")


if __name__ == "__main__":
    # Construir path del run
    run_dir = os.path.join('outputs', RUN_DIR)
    
    if not os.path.exists(run_dir):
        print(f"❌ Error: No se encontró el directorio {run_dir}")
        sys.exit(1)
    
    # Validar dataset
    if DATASET not in ['validation', 'test']:
        print(f"❌ Error: DATASET debe ser 'validation' o 'test' (recibido: {DATASET})")
        sys.exit(1)
    
    # Ejecutar generación de scores
    generate_scores(run_dir, DATASET)
