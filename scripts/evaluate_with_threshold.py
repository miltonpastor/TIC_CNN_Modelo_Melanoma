#!/usr/bin/env python3
"""
Script para evaluar métricas especificando un umbral.
Guarda resultados en outputs/<run>/evaluations/threshold_<value>/

Uso:
    python scripts/evaluate_with_threshold.py --run resnet50_20260105_095956 --threshold 0.5
    python scripts/evaluate_with_threshold.py -r resnet50_20260105_095956 -t 0.5
"""
import os
import sys
import argparse

# Reducir verbosidad de TensorFlow (0=todo, 1=sin INFO, 2=sin WARNING, 3=solo ERROR)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import tensorflow as tf

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.plots import plot_confusion_matrix
from evaluation.metrics import save_threshold_evaluation
from evaluation.evaluate import evaluate_model_with_threshold
from data.data_loader import load_predivided_data
from data.preprocessing import create_data_generators, create_data_flow_from_dataframe

# Intentar habilitar memory growth en GPU para evitar reservas completas de VRAM
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"⚠️ No se pudo habilitar memory growth en GPU: {e}")

def evaluate_with_threshold(run_dir, threshold):
    """
    Evalúa el modelo con un umbral específico.
    
    Args:
        run_dir: Directorio del run (ej: outputs/resnet50_20251212_082430)
        threshold: Umbral para clasificación (ej: 0.5)
    """
    # Cargar el modelo
    model_path = os.path.join(run_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return
    
    print(f"📦 Cargando modelo desde: {model_path}")
    # compile=False evita el warning de absl sobre métricas compiladas cuando solo hacemos inferencia
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Cargar datos de test como DataFrame y convertir a generator
    print("📊 Cargando datos de test...")
    _, _, test_df = load_predivided_data()
    _, val_test_datagen = create_data_generators()
    test_generator = create_data_flow_from_dataframe(val_test_datagen, test_df, shuffle=False)
    
    # Evaluar con el umbral especificado
    print(f"🔍 Evaluando con umbral {threshold}...")
    report, cm, accuracy, y_true, y_pred_proba = evaluate_model_with_threshold(
        model, test_generator, threshold
    )
    
    # Crear directorio de salida
    eval_dir = os.path.join(run_dir, 'evaluations', f'threshold_{threshold:.3f}')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Guardar matriz de confusión
    cm_path = os.path.join(eval_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, ['Benign', 'Malignant'], cm_path)
    print(f"✅ Matriz de confusión guardada en: {cm_path}")
    
    # Guardar resultados
    save_threshold_evaluation(report, cm, accuracy, threshold, eval_dir)
    
    print(f"\n✅ Evaluación completada con threshold={threshold}")
    print(f"📁 Resultados guardados en: {eval_dir}")
    print(f"\n📊 Métricas:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Benign    - Precision: {report['Benign']['precision']:.4f}, Recall: {report['Benign']['recall']:.4f}, F1: {report['Benign']['f1-score']:.4f}")
    print(f"   Malignant - Precision: {report['Malignant']['precision']:.4f}, Recall: {report['Malignant']['recall']:.4f}, F1: {report['Malignant']['f1-score']:.4f}")
    print(f"\n📋 Matriz de Confusión:")
    print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")


if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description='Evaluar modelo con un umbral específico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/evaluate_with_threshold.py --run resnet50_20260105_095956 --threshold 0.5
  python scripts/evaluate_with_threshold.py -r resnet50_20260105_095956 -t 0.45
  python scripts/evaluate_with_threshold.py -r resnet50_20260105_095956 -t 0.6
        """
    )
    
    parser.add_argument(
        '--run', '-r',
        type=str,
        required=True,
        help='Nombre del directorio del run (ej: resnet50_20260105_095956)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Umbral para clasificación binaria entre 0.0 y 1.0 (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Construir path del run
    run_dir = os.path.join('outputs', args.run)
    
    if not os.path.exists(run_dir):
        print(f"❌ Error: No se encontró el directorio {run_dir}")
        print(f"\nDirectorios disponibles en outputs/:")
        outputs_path = 'outputs'
        if os.path.exists(outputs_path):
            runs = [d for d in os.listdir(outputs_path) if os.path.isdir(os.path.join(outputs_path, d))]
            for run in sorted(runs):
                print(f"  - {run}")
        sys.exit(1)
    
    # Validar threshold
    if not 0 < args.threshold < 1:
        print(f"❌ Error: El umbral debe estar entre 0 y 1 (recibido: {args.threshold})")
        sys.exit(1)
    
    # Ejecutar evaluación
    evaluate_with_threshold(run_dir, args.threshold)
