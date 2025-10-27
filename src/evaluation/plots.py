import matplotlib.pyplot as plt
import os
from datetime import datetime
from config.config import OUTPUT_FOLDER

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
    plt.show()

    # Guardar con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(OUTPUT_FOLDER, f"training_curves_{timestamp}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path
