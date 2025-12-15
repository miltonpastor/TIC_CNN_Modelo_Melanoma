#!/usr/bin/env python3
"""
Script para generar scores de predicción desde un modelo ya entrenado.
Útil para análisis posterior con notebooks.
"""
import os
import sys
import tensorflow as tf

# ============================================================================
# CONFIGURACIÓN - Modifica estos valores según tus necesidades
# ============================================================================
RUN_DIR = 'resnet50_20251212_082430'  # Nombre del directorio del run
DATASET = 'validation'                 # 'validation' o 'test'
# ============================================================================

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.evaluate import save_scores
from data.data_loader import load_data

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
    
    # Cargar datos
    print(f"📊 Cargando datos de {dataset}...")
    train_gen, val_gen, test_gen = load_data()
    
    if dataset == 'validation':
        generator = val_gen
    elif dataset == 'test':
        generator = test_gen
    else:
        print(f"❌ Error: Dataset '{dataset}' no válido. Usa 'validation' o 'test'")
        return
    
    # Generar predicciones
    print(f"🔍 Generando predicciones para {len(generator.classes)} muestras...")
    generator.reset()
    predictions = model.predict(generator)
    true_labels = generator.classes
    
    # Guardar scores en el directorio del run
    # Cambiar OUTPUT_FOLDER temporalmente
    from config.config import OUTPUT_FOLDER
    import config.config as config_module
    original_output = config_module.OUTPUT_FOLDER
    config_module.OUTPUT_FOLDER = run_dir
    
    save_scores(true_labels, predictions.flatten(), dataset_name=dataset)
    
    # Restaurar OUTPUT_FOLDER
    config_module.OUTPUT_FOLDER = original_output
    
    print(f"\n✅ Scores guardados exitosamente en: {run_dir}/prediction_scores_{dataset}.csv")
    print(f"📊 Total de muestras procesadas: {len(true_labels)}")
    print(f"   Clase 0 (Benign): {(true_labels == 0).sum()}")
    print(f"   Clase 1 (Malignant): {(true_labels == 1).sum()}")


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
