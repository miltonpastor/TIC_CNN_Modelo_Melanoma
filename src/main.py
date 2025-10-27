# src/main.py
from data.data_loader import load_and_clean_data
from data.split_data import create_splits
from config.config import OUTPUT_FOLDER, SAMPLE_SIZE, MODEL_CONFIG as config_model , TRAINING_CONFIG as config_train
from models.transfer_learning import build_resnet50_classifier
from training.train import TwoStageTrainer
from evaluation.plots import plot_two_stage_training

def main():
    # Cargar y limpiar datos
    df = load_and_clean_data(sample_size=SAMPLE_SIZE)

    # Crear splits
    train_df, val_df, test_df = create_splits(df)

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
    history_a = trainer.stage_a_head_training(train_generator, val_generator)
    history_b = trainer.stage_b_fine_tuning(train_generator, val_generator)

    # Graficar resultados
    plot_two_stage_training(history_a, history_b)

    # Evaluar modelo
    test_loss, test_acc, test_auc = trainer.model.evaluate(test_generator)

    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Pipeline completo finalizado. Resultados en {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
