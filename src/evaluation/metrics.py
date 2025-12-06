import json
import os
from datetime import datetime
from config.config import OUTPUT_FOLDER, TRAINING_CONFIG, MODEL_CONFIG, LABEL_MAPPING


def save_results(eval_results, history_a, history_b, train_size, val_size, test_size):
    """Guarda métricas y configuración en results.json"""
    
    # Extraer métricas del classification report
    report = eval_results['classification_report']
    
    results = {
        "model_info": {
            "architecture": "ResNet50",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "dataset_info": {
            "train_samples": train_size,
            "val_samples": val_size,
            "test_samples": test_size,
            "total_samples": train_size + val_size + test_size,
            "label_mapping": LABEL_MAPPING
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
        "training_history": {
            "head_training": {
                "final_train_accuracy": float(history_a.history['accuracy'][-1]),
                "final_val_accuracy": float(history_a.history['val_accuracy'][-1]),
                "final_train_loss": float(history_a.history['loss'][-1]),
                "final_val_loss": float(history_a.history['val_loss'][-1])
            },
            "fine_tuning": {
                "final_train_accuracy": float(history_b.history['accuracy'][-1]),
                "final_val_accuracy": float(history_b.history['val_accuracy'][-1]),
                "final_train_loss": float(history_b.history['loss'][-1]),
                "final_val_loss": float(history_b.history['val_loss'][-1])
            }
        },
        "evaluation_metrics": {
            "overall_performance": {
                "accuracy": float(eval_results['accuracy']),
                "loss": float(eval_results['loss'])
            },
            "discrimination_metrics": {
                "auroc": float(eval_results.get('roc_auc', 0.0)),
                "auprc": float(eval_results.get('pr_auc', 0.0))
            },
            "calibration_metrics": {
                "brier_score": float(eval_results.get('brier_score', 0.0))
            },
            "per_class_metrics": {
                "benign": {
                    "precision": float(report['Benigno']['precision']),
                    "recall": float(report['Benigno']['recall']),
                    "f1_score": float(report['Benigno']['f1-score']),
                    "support": int(report['Benigno']['support'])
                },
                "malignant": {
                    "precision": float(report['Maligno']['precision']),
                    "recall": float(report['Maligno']['recall']),
                    "f1_score": float(report['Maligno']['f1-score']),
                    "support": int(report['Maligno']['support'])
                }
            },
            "averaged_metrics": {
                "macro_avg": {
                    "precision": float(report['macro avg']['precision']),
                    "recall": float(report['macro avg']['recall']),
                    "f1_score": float(report['macro avg']['f1-score'])
                },
                "weighted_avg": {
                    "precision": float(report['weighted avg']['precision']),
                    "recall": float(report['weighted avg']['recall']),
                    "f1_score": float(report['weighted avg']['f1-score'])
                }
            },
            "confusion_matrix": {
                "matrix": eval_results['confusion_matrix'].tolist(),
                "labels": ["Benign", "Malignant"],
                "description": "[[TN, FP], [FN, TP]]"
            }
        }
    }
    
    results_path = os.path.join(OUTPUT_FOLDER, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Resultados guardados en: {results_path}")
    return results_path
