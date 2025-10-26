# src/main.py
from data.data_loader import load_and_clean_data
from data.split_data import create_splits
from config.config import OUTPUT_FOLDER

def main():
    # 1. Cargar y limpiar datos
    df = load_and_clean_data()

    # 2. Crear splits
    train_df, val_df, test_df = create_splits(df)


    print(f"Pipeline completo finalizado. Resultados en {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
