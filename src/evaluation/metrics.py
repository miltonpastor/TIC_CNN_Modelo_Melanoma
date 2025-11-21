import json
import os
from datetime import datetime
from config.config import OUTPUT_FOLDER, TRAINING_CONFIG, MODEL_CONFIG, LABEL_MAPPING


def save_results(eval_results, history_a, history_b, train_size, val_size, test_size):
    """Guarda métricas y configuración en results.json"""
    results = {
        "model": "ResNet50",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label_mapping": LABEL_MAPPING,
        "dataset": {
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "total_samples": train_size + val_size + test_size
        },
        "training_config": {
            "head_training_epochs": TRAINING_CONFIG['head_epochs'],
            "fine_tuning_epochs": TRAINING_CONFIG['finetune_epochs'],
            "total_epochs": TRAINING_CONFIG['head_epochs'] + TRAINING_CONFIG['finetune_epochs'],
            "unfreeze_layers": TRAINING_CONFIG['unfreeze_layers'],
            "head_learning_rate": 1e-3,
            "fine_tuning_learning_rate": 1e-5
        },
        "model_config": MODEL_CONFIG,
        "final_metrics": {
            "test_accuracy": float(eval_results['accuracy']),
            "test_loss": float(eval_results['loss']),
            "test_auc": float(eval_results['auc'])
        },
        "classification_report": eval_results['classification_report'],
        "confusion_matrix": eval_results['confusion_matrix'].tolist(),
        "training_history": {
            "head_training": {
                "final_train_acc": float(history_a.history['accuracy'][-1]),
                "final_val_acc": float(history_a.history['val_accuracy'][-1]),
                "final_train_loss": float(history_a.history['loss'][-1]),
                "final_val_loss": float(history_a.history['val_loss'][-1])
            },
            "fine_tuning": {
                "final_train_acc": float(history_b.history['accuracy'][-1]),
                "final_val_acc": float(history_b.history['val_accuracy'][-1]),
                "final_train_loss": float(history_b.history['loss'][-1]),
                "final_val_loss": float(history_b.history['val_loss'][-1])
            }
        }
    }
    
    results_path = os.path.join(OUTPUT_FOLDER, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Resultados guardados en: {results_path}")
    return results_path
