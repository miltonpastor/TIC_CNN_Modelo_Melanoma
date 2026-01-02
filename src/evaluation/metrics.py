import json
import os
from datetime import datetime
from config.config import OUTPUT_FOLDER, TRAINING_CONFIG, MODEL_CONFIG, LABEL_MAPPING, MODEL_NAME


def save_results(eval_results, history_a, history_b, train_size, val_size, test_size):
    """Guarda métricas automáticas (sin umbral) y configuración en results.json.

    Esta función ahora tolera `history_a` o `history_b` en `None`,
    registrando únicamente las fases disponibles del entrenamiento.
    """

    def _extract_phase_metrics(history):
        """Extrae las últimas métricas conocidas de un objeto History de Keras."""
        if history is None or getattr(history, 'history', None) is None:
            return None
        h = history.history
        metrics_map = {}
        # Lista de métricas comunes usadas en el proyecto
        for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'auc', 'val_auc']:
            if key in h and h[key]:
                try:
                    metrics_map[f"final_{key}"] = float(h[key][-1])
                except (TypeError, ValueError):
                    # En caso de que el valor no sea convertible a float
                    metrics_map[f"final_{key}"] = h[key][-1]
        return metrics_map if metrics_map else None

    head_metrics = _extract_phase_metrics(history_a)
    finetune_metrics = _extract_phase_metrics(history_b)

    results = {
        "model_info": {
            "architecture": MODEL_NAME,
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
            **({"head_training": head_metrics} if head_metrics is not None else {}),
            **({"fine_tuning": finetune_metrics} if finetune_metrics is not None else {})
        },
        "evaluation_metrics": {
            "auroc": float(eval_results['roc_auc']),
            "auprc": float(eval_results['pr_auc']),
            "brier_score": float(eval_results['brier_score'])
        }
    }
    
    results_path = os.path.join(OUTPUT_FOLDER, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Resultados guardados en: {results_path}")
    return results_path


def save_threshold_evaluation(report, cm, accuracy, threshold, eval_dir):
    """
    Guarda resultados de evaluación con un umbral específico.
    
    Args:
        report: Reporte de clasificación de sklearn (dict)
        cm: Matriz de confusión (numpy array)
        accuracy: Accuracy calculada (float)
        threshold: Umbral usado para clasificación (float)
        eval_dir: Directorio donde guardar los resultados
    
    Returns:
        str: Ruta del archivo guardado
    """
    results = {
        "evaluation_info": {
            "threshold": threshold,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "accuracy": float(accuracy),
            "per_class": {
                "benign": {
                    "precision": float(report['Benign']['precision']),
                    "recall": float(report['Benign']['recall']),
                    "f1_score": float(report['Benign']['f1-score']),
                    "support": int(report['Benign']['support'])
                },
                "malignant": {
                    "precision": float(report['Malignant']['precision']),
                    "recall": float(report['Malignant']['recall']),
                    "f1_score": float(report['Malignant']['f1-score']),
                    "support": int(report['Malignant']['support'])
                }
            },
            "averaged": {
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
            }
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": ["Benign", "Malignant"],
            "description": "[[TN, FP], [FN, TP]]"
        }
    }
    
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path
