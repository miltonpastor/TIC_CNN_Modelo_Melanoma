import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from config.config import OUTPUT_FOLDER

def _save_plot(filename):
    """Guarda el plot actual en la carpeta figures."""
    plots_dir = os.path.join(OUTPUT_FOLDER, "figures")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(plots_dir, f"{filename}_{timestamp}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    return file_path

def plot_two_stage_training(history_a, history_b):
    """Grafica accuracy y loss combinando Etapa A (head) + Etapa B (fine-tuning)."""
    acc = history_a.history['accuracy'] + history_b.history['accuracy']
    val_acc = history_a.history['val_accuracy'] + history_b.history['val_accuracy']
    loss = history_a.history['loss'] + history_b.history['loss']
    val_loss = history_a.history['val_loss'] + history_b.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Entrenamiento')
    plt.plot(epochs, val_acc, 'r--', label='Validación')
    plt.title('Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Entrenamiento')
    plt.plot(epochs, val_loss, 'r--', label='Validación')
    plt.title('Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    return _save_plot("training_curves")

def plot_confusion_matrix(cm):
    """Grafica y guarda la matriz de confusión."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Matriz de Confusión')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predicho')
    plt.tight_layout()
    return _save_plot("confusion_matrix")

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10):
    """
    Grafica la curva de calibración del modelo.
    
    Args:
        y_true: Etiquetas reales (0 o 1)
        y_pred_proba: Probabilidades predichas por el modelo
        n_bins: Número de bins para la calibración
    
    Returns:
        str: Ruta del archivo guardado
    """
    plt.figure(figsize=(10, 8))
    
    # Calcular la curva de calibración
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Gráfica principal
    plt.subplot(2, 1, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectamente calibrado')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', 
             label=f'Modelo (n_bins={n_bins})', linewidth=2, markersize=8)
    plt.ylabel('Fracción de positivos', fontsize=12)
    plt.xlabel('Probabilidad media predicha', fontsize=12)
    plt.title('Curva de Calibración', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Histograma de predicciones
    plt.subplot(2, 1, 2)
    plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Probabilidad predicha', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Probabilidades Predichas', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _save_plot("calibration_curve")

def plot_roc_curve(y_true, y_pred_proba):
    """
    Grafica la curva ROC (Receiver Operating Characteristic).
    
    Args:
        y_true: Etiquetas reales (0 o 1)
        y_pred_proba: Probabilidades predichas por el modelo
    
    Returns:
        str: Ruta del archivo guardado
    """
    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    
    # Curva ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5000)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _save_plot("roc_curve")

def plot_precision_recall_curve(y_true, y_pred_proba):
    """
    Grafica la curva Precision-Recall (PR).
    
    Args:
        y_true: Etiquetas reales (0 o 1)
        y_pred_proba: Probabilidades predichas por el modelo
    
    Returns:
        str: Ruta del archivo guardado
    """
    # Calcular la curva PR
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Baseline (prevalencia de la clase positiva)
    baseline = np.sum(y_true) / len(y_true)
    
    plt.figure(figsize=(10, 8))
    
    # Curva PR
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', 
                label=f'Baseline (Prevalence = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _save_plot("precision_recall_curve")


