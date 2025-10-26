import os
from sklearn.model_selection import train_test_split
from src.config.config import TRAIN_SIZE, RANDOM_SEED, CSV_SPLIT_FOLDER

def create_splits(df):
    # Split train/temp
    train_df, temp_df = train_test_split(
        df,
        test_size=1-TRAIN_SIZE,
        stratify=df['label'],
        random_state=RANDOM_SEED,
        shuffle=True
    )
    # Split val/test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=RANDOM_SEED,
        shuffle=True
    )
    # Guardar CSVs
    os.makedirs(CSV_SPLIT_FOLDER, exist_ok=True)
    train_df.to_csv(os.path.join(CSV_SPLIT_FOLDER, "train.csv"), index=False)
    val_df.to_csv(os.path.join(CSV_SPLIT_FOLDER, "val.csv"), index=False)
    test_df.to_csv(os.path.join(CSV_SPLIT_FOLDER, "test.csv"), index=False)
    
    return train_df, val_df, test_df