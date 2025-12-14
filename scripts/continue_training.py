"""
Script para continuar el entrenamiento desde un modelo pre-entrenado.

Uso:
    python scripts/continue_training.py

Configura las rutas y épocas directamente en este archivo.
"""

import sys
import os
sys.path.append('src')

import tensorflow as tf
from data.data_loader import load_predivided_data
from data.preprocessing import create_data_generators, create_data_flow_from_dataframe
from utils.class_weights import calculate_class_weights
from training.train import TwoStageTrainer
from evaluation.evaluate import evaluate_model
from evaluation.metrics import save_results
from datetime import datetime
import json

# ============ CONFIGURACIÓN ============
# Ruta al modelo pre-entrenado
PRETRAINED_MODEL_PATH = 'outputs/resnet50_20251212_082430/best_model.h5'

# Épocas ya completadas (head_training: 4 + fine_tuning: 15 = 19)
COMPLETED_EPOCHS = 19

# Épocas objetivo para fine-tuning (querías 50 en total)
TARGET_FINE_TUNING_EPOCHS = 30

# Épocas adicionales a entrenar
ADDITIONAL_EPOCHS = TARGET_FINE_TUNING_EPOCHS - 15  # 30 - 15 = 15 épocas más

# Learning rate para continuar (puede ser más bajo)
CONTINUE_LR = 1e-5 

# Carpeta de salida
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = f"outputs/resnet50_continued_{RUN_TIMESTAMP}"

# ============ CÓDIGO ============

def main():
    print("="*60)
    print("🔄 CONTINUANDO ENTRENAMIENTO DESDE MODELO PRE-ENTRENADO")
    print("="*60)
    print(f"📂 Modelo base: {PRETRAINED_MODEL_PATH}")
    print(f"📊 Épocas completadas: {COMPLETED_EPOCHS}")
    print(f"📊 Épocas adicionales: {ADDITIONAL_EPOCHS}")
    print(f"🎯 Total épocas fine-tuning objetivo: {TARGET_FINE_TUNING_EPOCHS}")
    print(f"📁 Resultados en: {OUTPUT_FOLDER}")
    print("="*60)
    
    # Actualizar OUTPUT_FOLDER globalmente para que TwoStageTrainer lo use
    import config.config as config
    config.OUTPUT_FOLDER = OUTPUT_FOLDER
    
    # Cargar datos
    print("\n📥 Cargando datos...")
    train_df, val_df, test_df = load_predivided_data()
    
    # Calcular class weights
    class_weight_dict = calculate_class_weights(train_df['label'])
    
    # Crear generadores
    print("\n🔧 Creando generadores de datos...")
    train_datagen, val_test_datagen = create_data_generators()
    train_generator = create_data_flow_from_dataframe(train_datagen, train_df)
    val_generator = create_data_flow_from_dataframe(val_test_datagen, val_df, shuffle=False)
    test_generator = create_data_flow_from_dataframe(val_test_datagen, test_df, shuffle=False)
    
    # Cargar modelo pre-entrenado
    print(f"\n🔄 Cargando modelo desde: {PRETRAINED_MODEL_PATH}")
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
    
    # Verificar capas entrenables
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"   Capas entrenables: {trainable_count}/{len(model.layers)}")
    
    # Crear trainer (base no se usa en continue_fine_tuning, pero se requiere para inicializar)
    trainer = TwoStageTrainer(model, None, config.TRAINING_CONFIG)
    
    # Continuar entrenamiento usando el método del trainer
    print(f"\n🚀 Iniciando entrenamiento adicional...")
    history = trainer.continue_fine_tuning(
        train_generator,
        val_generator,
        additional_epochs=ADDITIONAL_EPOCHS,
        learning_rate=CONTINUE_LR,
        class_weight=class_weight_dict
    )
    
    print("\n✅ Entrenamiento completado!")
    
    # Evaluar modelo
    print("\n📊 Evaluando modelo en test set...")
    eval_results = evaluate_model(trainer.model, test_generator)
    
    # Guardar resultados
    print("\n💾 Guardando resultados...")
    save_results(
        eval_results,
        None,  # No hay history_a (ya se entrenó antes)
        history,  # Solo el history de las épocas adicionales
        len(train_df), len(val_df), len(test_df)
    )
    
    # Guardar información de continuación
    continue_info = {
        "original_model": PRETRAINED_MODEL_PATH,
        "completed_epochs_before": COMPLETED_EPOCHS,
        "additional_epochs_trained": ADDITIONAL_EPOCHS,
        "total_epochs_now": COMPLETED_EPOCHS + ADDITIONAL_EPOCHS,
        "continue_learning_rate": CONTINUE_LR,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(OUTPUT_FOLDER, "continue_info.json"), 'w') as f:
        json.dump(continue_info, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 PROCESO COMPLETADO")
    print("="*60)
    print(f"📈 Test Accuracy: {eval_results['accuracy']:.4f}")
    print(f"📈 AUROC: {eval_results['roc_auc']:.4f}")
    print(f"📈 AUPRC: {eval_results['pr_auc']:.4f}")
    print(f"📂 Resultados guardados en: {OUTPUT_FOLDER}")
    print("="*60)

if __name__ == "__main__":
    main()
