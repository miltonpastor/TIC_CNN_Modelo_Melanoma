import pandas as pd
from config.config import CSV_PATH, IMAGES_FOLDER, ID_COLUMN, DIAGNOSIS_COLUMN, LABEL_MAPPING

def load_and_clean_data(sample_size=None, random_state=42):
    df = pd.read_csv(CSV_PATH)
    # Filtrar solo Benign y Malignant
    df_clean = df[df[DIAGNOSIS_COLUMN].isin(LABEL_MAPPING.keys())].copy()
    df_clean['label'] = df_clean[DIAGNOSIS_COLUMN].map(LABEL_MAPPING)
    df_simple = pd.DataFrame({
        'filepath': df_clean[ID_COLUMN].apply(lambda x: f"{IMAGES_FOLDER}/{x}.jpg"),
        'label': df_clean['label']
    })
    df_simple['label'] = df_simple['label'].astype(str) # Convertir a string para compatibilidad con Keras

    if sample_size is not None:
        df_simple = df_simple.sample(n=sample_size, random_state=random_state)

    return df_simple