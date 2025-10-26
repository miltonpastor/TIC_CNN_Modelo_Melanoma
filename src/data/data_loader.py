import pandas as pd
from config.config import CSV_PATH, IMAGES_FOLDER, ID_COLUMN, DIAGNOSIS_COLUMN, LABEL_MAPPING

def load_and_clean_data():
    df = pd.read_csv(CSV_PATH)
    # Filtrar solo Benign y Malignant
    df_clean = df[df[DIAGNOSIS_COLUMN].isin(LABEL_MAPPING.keys())].copy()
    df_clean['label'] = df_clean[DIAGNOSIS_COLUMN].map(LABEL_MAPPING)
    df_simple = pd.DataFrame({
        'filepath': df_clean[ID_COLUMN].apply(lambda x: f"{IMAGES_FOLDER}/{x}.jpg"),
        'label': df_clean['label']
    })
    return df_simple