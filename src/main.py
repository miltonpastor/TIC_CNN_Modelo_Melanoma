# src/main.py
import os
import tensorflow as tf
from data.data_loader import load_and_clean_data, load_predivided_data
from data.split_data import create_splits
from config.config import (
    OUTPUT_FOLDER, SAMPLE_SIZE, MODEL_CONFIG as config_model, 
    TRAINING_CONFIG as config_train, DATA_MODE, MODEL_NAME, 
    TRAIN_SAMPLE_SIZE, BATCH_SIZE, GPU_CONFIG
)
from models.transfer_learning import build_cnn_classifier
from training.train import TwoStageTrainer
from evaluation.evaluate import evaluate_model_without_threshold
from evaluation.plots import plot_two_stage_training
from evaluation.metrics import save_results
from utils.class_weights import calculate_class_weights


def configure_gpu():
    """
    Configura GPU para optimizar rendimiento en Tesla T4.
    - Habilita mixed precision (FP16) para acelerar cómputo
    - Configura memory growth para evitar OOM
    """
    # Detectar GPUs disponibles
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Habilitar memory growth (evita reservar toda la memoria)
            if GPU_CONFIG['memory_growth']:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth habilitado en {len(gpus)} GPU(s)")
            
            # Habilitar mixed precision (FP16) para Tesla T4
            # Acelera operaciones matriciales ~3x y reduce uso de memoria
            if GPU_CONFIG['mixed_precision']:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"✅ Mixed precision habilitado: {policy.name}")
                print(f"   Compute dtype: {policy.compute_dtype}")
                print(f"   Variable dtype: {policy.variable_dtype}")
            
        except RuntimeError as e:
            print(f"⚠️  Error configurando GPU: {e}")
    else:
        print("⚠️  No se detectaron GPUs. Ejecutando en CPU.")


def main():
    # Configurar GPU ANTES de crear cualquier modelo o dataset
    configure_gpu()
    
    # Cargar datos según el modo configurado
    if DATA_MODE == 'predivided':
        print("Loading pre-divided dataset...")
        train_df, val_df, test_df = load_predivided_data(train_sample_size=TRAIN_SAMPLE_SIZE)
    else:  # 'csv' mode
        print("Loading from CSV and creating splits...")
        df = load_and_clean_data(sample_size=SAMPLE_SIZE)
        train_df, val_df, test_df = create_splits(df)

    # Aplicar oversampling a train_df si está activado
    from data.preprocessing import oversample_minority_class
    train_df_balanced = oversample_minority_class(train_df)
    
    # Calcular class weights para balancear clases
    class_weight_dict = calculate_class_weights(train_df_balanced['label'])

    # Construir modelo
    model, base, preprocess_fn = build_cnn_classifier(
        arch=MODEL_NAME,
        input_shape=config_model['input_shape'],
        dropout_rate=config_model['dropout_rate'],
        dense_units=config_model['dense_units'],
        num_classes=config_model['num_classes']
    )
    
    # NUEVO: Crear pipelines tf.data optimizados
    from data.preprocessing import create_tf_dataset
    
    print("\n📊 Creando pipelines tf.data optimizados...")
    
    # Train: con augmentation, shuffle y cache
    # Usa configuraciones de GPU_CONFIG automáticamente
    train_dataset = create_tf_dataset(
        train_df_balanced,
        preprocess_fn=preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment=True,  # Augmentation en GPU
        cache=True     # Cache en memoria para reutilizar
    )
    
    # Validation: sin augmentation, sin shuffle, con cache
    val_dataset = create_tf_dataset(
        val_df,
        preprocess_fn=preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,  # Sin augmentation para validación
        cache=True
    )
    
    # Test: sin augmentation, sin shuffle, con cache
    test_dataset = create_tf_dataset(
        test_df,
        preprocess_fn=preprocess_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
        cache=True
    )
    
    print(f"✅ Train dataset: {len(train_df_balanced)} muestras")
    print(f"✅ Val dataset: {len(val_df)} muestras")
    print(f"✅ Test dataset: {len(test_df)} muestras")

    # Entrenar modelo
    trainer = TwoStageTrainer(model, base, config_train)
    history_a = trainer.stage_a_head_training(train_dataset, val_dataset, class_weight_dict)
    history_b = trainer.stage_b_fine_tuning(train_dataset, val_dataset, class_weight_dict)

    # Graficar resultados
    plot_two_stage_training(history_a, history_b)

    # Guardar scores del validation set
    from evaluation.evaluate import save_scores
    val_predictions = trainer.model.predict(val_dataset)
    val_true = tf.concat([y for x, y in val_dataset], axis=0).numpy()
    save_scores(val_true, val_predictions.flatten(), dataset_name='validation')

    # Evaluar modelo en test set (métricas sin umbral)
    eval_results = evaluate_model_without_threshold(trainer.model, test_dataset)

    # Guardar resultados
    save_results(
        eval_results,
        history_a, history_b,
        len(train_df), len(val_df), len(test_df)
    )

    print(f"AUROC: {eval_results['roc_auc']:.4f}, AUPRC: {eval_results['pr_auc']:.4f}, Brier Score: {eval_results['brier_score']:.4f}")
    print(f"Pipeline completo finalizado. Resultados en {OUTPUT_FOLDER}")
    print(f"\nPara evaluar métricas con umbral, ejecuta: python scripts/evaluate_with_threshold.py --run {os.path.basename(OUTPUT_FOLDER)} --threshold 0.5")

if __name__ == "__main__":
    main()
