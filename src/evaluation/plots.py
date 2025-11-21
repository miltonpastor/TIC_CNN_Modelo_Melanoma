import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
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
