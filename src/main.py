# src/main.py
from data.data_loader import load_and_clean_data
from data.split_data import create_splits
from config.config import OUTPUT_FOLDER

def main():
    # 1. Cargar y limpiar datos
    df = load_and_clean_data()

    # 2. Crear splits
    train_df, val_df, test_df = create_splits(df)

    # 3. Crear generadores de datos
    from data.preprocessing import create_data_generators, create_data_flow_from_dataframe
    train_datagen, val_test_datagen = create_data_generators()
    train_generator = create_data_flow_from_dataframe(train_datagen, train_df)
    val_generator = create_data_flow_from_dataframe(val_test_datagen, val_df, shuffle=False)
    test_generator = create_data_flow_from_dataframe(val_test_datagen, test_df, shuffle=False)


    print(f"Pipeline completo finalizado. Resultados en {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
