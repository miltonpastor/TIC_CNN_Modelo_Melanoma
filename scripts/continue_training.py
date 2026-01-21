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
from data.preprocessing import create_tf_dataset, oversample_minority_class
from utils.class_weights import calculate_class_weights
from training.train import TwoStageTrainer
from evaluation.evaluate import evaluate_model_without_threshold
from evaluation.metrics import save_results
from config.config import BATCH_SIZE, CLASS_BALANCE_CONFIG
from datetime import datetime
import json

# ============ CONFIGURACIÓN ============
# Ruta al modelo pre-entrenado
PRETRAINED_MODEL_PATH = 'outputs/comparision_resnet/resnet50_20260110_001917-normal/best_model.h5'

# Épocas ya completadas (head_training: 4 + fine_tuning: 15 = 19)
COMPLETED_EPOCHS = 19

# Épocas objetivo para fine-tuning
TARGET_FINE_TUNING_EPOCHS = 29  # 19 + 10 épocas adicionales

# Épocas adicionales a entrenar
ADDITIONAL_EPOCHS = 10  # 10 épocas más

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
    
    # Aplicar oversampling si está configurado
    train_df_balanced = oversample_minority_class(train_df)
    
    # Calcular class weights
    if CLASS_BALANCE_CONFIG['use_class_weights']:
        class_weight_dict = calculate_class_weights(train_df_balanced['label'])
    else:
        class_weight_dict = None
        print("⚠️  Class weights desactivado")
    
    # Cargar modelo pre-entrenado PRIMERO para obtener preprocess_fn
    print(f"\n🔄 Cargando modelo desde: {PRETRAINED_MODEL_PATH}")
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH)
    
    # Verificar capas entrenables
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"   Capas entrenables: {trainable_count}/{len(model.layers)}")
    
    # Determinar función de preprocesamiento según el modelo
    model_name = PRETRAINED_MODEL_PATH.split('/')[-2].split('_')[0]
    if 'resnet50' in model_name:
        from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_fn
    elif 'efficientnet' in model_name:
        from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_fn
    elif 'densenet' in model_name:
        from tensorflow.keras.applications.densenet import preprocess_input as preprocess_fn
    else:
        from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_fn
        print("⚠️  Modelo no reconocido, usando preprocess de ResNet50")
    
    # Crear datasets tf.data optimizados
    print("\n🔧 Creando datasets TensorFlow...")
    train_dataset = create_tf_dataset(
        train_df_balanced,
        preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True
    )
    
    val_dataset = create_tf_dataset(
        val_df,
        preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    test_dataset = create_tf_dataset(
        test_df,
        preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False
    )
    
    # Crear trainer (base no se usa en continue_fine_tuning, pero se requiere para inicializar)
    trainer = TwoStageTrainer(model, None, config.TRAINING_CONFIG)
    
    # Continuar entrenamiento usando el método del trainer
    print(f"\n🚀 Iniciando entrenamiento adicional...")
    history = trainer.continue_fine_tuning(
        train_dataset,
        val_dataset,
        additional_epochs=ADDITIONAL_EPOCHS,
        learning_rate=CONTINUE_LR,
        class_weight=class_weight_dict
    )
    
    print("\n✅ Entrenamiento completado!")
    
    # Evaluar modelo (métricas sin umbral, consistentes con save_results)
    print("\n📊 Evaluando modelo en test set...")
    eval_results = evaluate_model_without_threshold(trainer.model, test_dataset)
    
    # Guardar resultados
    print("\n💾 Guardando resultados...")
    save_results(
        eval_results,
        None,  # No hay history_a (ya se entrenó antes)
        history,  # Solo el history de las épocas adicionales
        len(train_df), 
        len(val_df), 
        len(test_df)
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
    print(f"📈 AUROC: {eval_results['roc_auc']:.4f}")
    print(f"📈 AUPRC: {eval_results['pr_auc']:.4f}")
    print(f"📈 Brier Score: {eval_results['brier_score']:.4f}")
    print(f"📂 Resultados guardados en: {OUTPUT_FOLDER}")
    print("="*60)

if __name__ == "__main__":
    main()
