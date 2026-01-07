import os
import pandas as pd
from config.config import CSV_PATH, IMAGES_FOLDER, ID_COLUMN, DIAGNOSIS_COLUMN, LABEL_MAPPING, DATA_MODE, LISTS_FOLDER, TRAIN_SAMPLE_SIZE

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

def load_predivided_data(train_sample_size=None, random_state=42):
    """Load pre-divided train, validation and test sets from txt files
    
    Args:
        train_sample_size: Number of training samples to use (None = use all)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df with proportional sampling
    """
    train_df = load_from_txt(os.path.join(LISTS_FOLDER, 'train.txt'))
    val_df = load_from_txt(os.path.join(LISTS_FOLDER, 'validation.txt'))
    test_df = load_from_txt(os.path.join(LISTS_FOLDER, 'test.txt'))
    
    # If train_sample_size is specified, sample proportionally
    if train_sample_size is not None:
        total_train = len(train_df)
        if train_sample_size > total_train:
            print(f"[data_loader] train_sample_size={train_sample_size} mayor que dataset ({total_train}). Usando {total_train}.")
            train_sample_size = total_train
        
        # Calculate proportion and apply to validation and test
        proportion = train_sample_size / total_train
        val_sample_size = max(1, int(len(val_df) * proportion))
        test_sample_size = max(1, int(len(test_df) * proportion))
        
        print(f"[data_loader] Muestreando proporcionalmente:")
        print(f"  Train: {train_sample_size}/{total_train} ({proportion*100:.1f}%)")
        print(f"  Val: {val_sample_size}/{len(val_df)} ({proportion*100:.1f}%)")
        print(f"  Test: {test_sample_size}/{len(test_df)} ({proportion*100:.1f}%)")
        
        train_df = train_df.sample(n=train_sample_size, random_state=random_state)
        val_df = val_df.sample(n=val_sample_size, random_state=random_state)
        test_df = test_df.sample(n=test_sample_size, random_state=random_state)
    
    return train_df, val_df, test_df