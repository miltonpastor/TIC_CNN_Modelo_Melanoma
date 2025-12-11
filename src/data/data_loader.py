import os
import pandas as pd
from config.config import CSV_PATH, IMAGES_FOLDER, ID_COLUMN, DIAGNOSIS_COLUMN, LABEL_MAPPING, DATA_MODE, LISTS_FOLDER

def load_and_clean_data(sample_size=None, random_state=42):
    df = pd.read_csv(CSV_PATH)
    df_clean = df[df[DIAGNOSIS_COLUMN].isin(LABEL_MAPPING.keys())].copy()
    df_clean['label'] = df_clean[DIAGNOSIS_COLUMN].map(LABEL_MAPPING)
    df_simple = pd.DataFrame({
        'filepath': df_clean[ID_COLUMN].apply(lambda x: f"{IMAGES_FOLDER}/{x}.jpg"),
        'label': df_clean['label']
    })
    df_simple['label'] = df_simple['label'].astype(str) # Convertir a string para compatibilidad con Keras

    if sample_size is not None:
        available = len(df_simple)
        if sample_size > available:
            # Reducir tamaño solicitado para evitar ValueError
            print(f"[data_loader] sample_size={sample_size} mayor que dataset ({available}). Usando {available}.")
            sample_size = available
        df_simple = df_simple.sample(n=sample_size, random_state=random_state)

    return df_simple

def load_from_txt(txt_path):
    """Load dataset from txt file with format: filepath label"""
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filepath, label = parts
                data.append({'filepath': filepath, 'label': label})
    return pd.DataFrame(data)

def load_predivided_data():
    """Load pre-divided train, validation and test sets from txt files"""
    train_df = load_from_txt(os.path.join(LISTS_FOLDER, 'train.txt'))
    val_df = load_from_txt(os.path.join(LISTS_FOLDER, 'validation.txt'))
    test_df = load_from_txt(os.path.join(LISTS_FOLDER, 'test.txt'))
    return train_df, val_df, test_df