# src/main.py
from data.data_loader import load_and_clean_data, load_predivided_data
from data.split_data import create_splits
from config.config import OUTPUT_FOLDER, SAMPLE_SIZE, MODEL_CONFIG as config_model , TRAINING_CONFIG as config_train, DATA_MODE
from models.transfer_learning import build_resnet50_classifier
from training.train import TwoStageTrainer
from evaluation.evaluate import evaluate_model
from evaluation.plots import plot_two_stage_training
from evaluation.metrics import save_results
from utils.class_weights import calculate_class_weights

def main():
    # Cargar datos según el modo configurado
    if DATA_MODE == 'predivided':
        print("Loading pre-divided dataset...")
        train_df, val_df, test_df = load_predivided_data()
    else:  # 'csv' mode
        print("Loading from CSV and creating splits...")
        df = load_and_clean_data(sample_size=SAMPLE_SIZE)
        train_df, val_df, test_df = create_splits(df)

    # Calcular class weights para balancear clases
    class_weight_dict = calculate_class_weights(train_df['label'])

    # Crear generadores de datos
    from data.preprocessing import create_data_generators, create_data_flow_from_dataframe
    train_datagen, val_test_datagen = create_data_generators()
    train_generator = create_data_flow_from_dataframe(train_datagen, train_df)
    val_generator = create_data_flow_from_dataframe(val_test_datagen, val_df, shuffle=False)
    test_generator = create_data_flow_from_dataframe(val_test_datagen, test_df, shuffle=False)


    # Construir modelo
    model, base = build_resnet50_classifier(
        input_shape=config_model['input_shape'],
        dropout_rate=config_model['dropout_rate'],
        dense_units=config_model['dense_units'],
        num_classes=config_model['num_classes']
    )

    # Entrenar modelo
    trainer = TwoStageTrainer(model, base, config_train)
    history_a = trainer.stage_a_head_training(train_generator, val_generator, class_weight_dict)
    history_b = trainer.stage_b_fine_tuning(train_generator, val_generator, class_weight_dict)

    # Graficar resultados
    plot_two_stage_training(history_a, history_b)

    # Guardar scores del validation set
    from evaluation.evaluate import save_scores
    val_generator.reset()
    val_predictions = trainer.model.predict(val_generator)
    val_true = val_generator.classes
    save_scores(val_true, val_predictions.flatten(), dataset_name='validation')

    # Evaluar modelo en test set (métricas sin umbral)
    from evaluation.evaluate import evaluate_model_without_threshold
    eval_results = evaluate_model_without_threshold(trainer.model, test_generator)

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
