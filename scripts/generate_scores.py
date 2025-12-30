#!/usr/bin/env python3
"""
Script para generar scores de predicción desde un modelo ya entrenado.
Útil para análisis posterior con notebooks.
"""
import os
import sys
import tensorflow as tf
import numpy as np

# ============================================================================
# CONFIGURACIÓN - Modifica estos valores según tus necesidades
# ============================================================================
RUN_DIR = 'resnet50_20251224_170207'  # Nombre del directorio del run
DATASET = 'validation'                 # 'validation' o 'test'
# ============================================================================

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.evaluate import save_scores
from data.data_loader import load_predivided_data
from data.preprocessing import create_data_generators, create_data_flow_from_dataframe

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
    
    # Cargar datos (DataFrames) y crear generadores Keras
    print(f"📊 Cargando datos de {dataset}...")
    train_df, val_df, test_df = load_predivided_data()
    
    # Crear datagens (usamos el de validación/test sin augmentation)
    _, val_test_datagen = create_data_generators()

    # Construir el generator adecuado desde el DataFrame
    if dataset == 'validation':
        generator = create_data_flow_from_dataframe(val_test_datagen, val_df, shuffle=False)
    elif dataset == 'test':
        generator = create_data_flow_from_dataframe(val_test_datagen, test_df, shuffle=False)
    else:
        print(f"❌ Error: Dataset '{dataset}' no válido. Usa 'validation' o 'test'")
        return
    
    # Generar predicciones
    print(f"🔍 Generando predicciones para {len(generator.classes)} muestras...")
    generator.reset()
    predictions = model.predict(generator)
    true_labels = generator.classes
    

    save_scores(true_labels, predictions.flatten(), dataset_name=dataset, output_folder=run_dir)
    
    
    print(f"\n✅ Scores guardados exitosamente en: {run_dir}/prediction_scores_{dataset}.csv")
    print(f"📊 Total de muestras procesadas: {len(true_labels)}")
    benign_count = int(np.count_nonzero(np.asarray(true_labels) == 0))
    malignant_count = int(np.count_nonzero(np.asarray(true_labels) == 1))
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
