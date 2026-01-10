"""
Script para comparar métricas de validación entre diferentes arquitecturas
usando los logs de TensorBoard
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
from collections import defaultdict
import seaborn as sns
import tensorflow as tf

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['font.size'] = 11

def read_tensorboard_logs(log_dir):
    """Lee los eventos de TensorBoard y extrae las métricas"""
    metrics = defaultdict(lambda: {'steps': [], 'values': []})
    
    # Buscar archivos de eventos
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                try:
                    for event in summary_iterator(event_file):
                        for value in event.summary.value:
                            # Solo nos interesan las métricas epoch_ 
                            metric_name = value.tag
                            if not metric_name.startswith('epoch_'):
                                continue
                            
                            # Intentar extraer el valor
                            metric_value = None
                            if value.HasField('simple_value'):
                                metric_value = value.simple_value
                            elif value.HasField('tensor'):
                                tensor_proto = value.tensor
                                tensor = tf.make_ndarray(tensor_proto)
                                if tensor.size == 1:
                                    metric_value = float(tensor)
                            
                            if metric_value is not None:
                                metrics[metric_name]['steps'].append(event.step)
                                metrics[metric_name]['values'].append(metric_value)
                except Exception as e:
                    print(f"Error leyendo {event_file}: {e}")
    
    return metrics

def extract_model_metrics(base_path, model_name):
    """Extrae métricas de validación de un modelo"""
    model_data = {
        'name': model_name,
        'head_training': {'validation': {}},
        'fine_tuning': {'validation': {}}
    }
    
    # Head training - validación
    ht_val_path = os.path.join(base_path, 'logs', 'head_training', 'validation')
    if os.path.exists(ht_val_path):
        model_data['head_training']['validation'] = read_tensorboard_logs(ht_val_path)
        print(f"  Head Training (Val): {len(model_data['head_training']['validation'])} métricas")
    
    # Fine tuning - validación
    ft_val_path = os.path.join(base_path, 'logs', 'fine_tuning', 'validation')
    if os.path.exists(ft_val_path):
        model_data['fine_tuning']['validation'] = read_tensorboard_logs(ft_val_path)
        print(f"  Fine Tuning (Val): {len(model_data['fine_tuning']['validation'])} métricas")
    
    return model_data

def smooth_curve(points, factor=0.85):
    """Suaviza una curva usando promedio exponencial móvil"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_comparison(models_data, output_dir):
    """Crea gráficas comparativas de las métricas de validación"""
    
    # Colores para cada modelo
    colors = {
        'densenet121': '#1f77b4',      # Azul
        'efficientnet-b0': '#ff7f0e',  # Naranja
        'resnet50': '#2ca02c'          # Verde
    }
    
    # Crear figura con 2 subplots (solo validación)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('Comparación de Arquitecturas - Métricas de Validación (Head Training + Fine Tuning)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # 1. Validation Accuracy
    ax = axes[0]
    ax.set_title('Validation Accuracy', fontweight='bold', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Validation Loss
    ax2 = axes[1]
    ax2.set_title('Validation Loss', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    for model_data in models_data:
        model_name = model_data['name']
        color = colors.get(model_name, '#000000')
        
        # Concatenar épocas de head training y fine tuning
        all_acc_steps = []
        all_acc_values = []
        all_loss_steps = []
        all_loss_values = []
        
        # Head training
        ht_val = model_data['head_training']['validation']
        if 'epoch_accuracy' in ht_val:
            ht_epochs = len(ht_val['epoch_accuracy']['values'])
            all_acc_steps.extend(range(ht_epochs))
            all_acc_values.extend(ht_val['epoch_accuracy']['values'])
        
        if 'epoch_loss' in ht_val:
            ht_loss_epochs = len(ht_val['epoch_loss']['values'])
            all_loss_steps.extend(range(ht_loss_epochs))
            all_loss_values.extend(ht_val['epoch_loss']['values'])
        
        # Fine tuning (continuar desde donde terminó head training)
        ft_val = model_data['fine_tuning']['validation']
        if 'epoch_accuracy' in ft_val:
            ft_start = max(all_acc_steps) + 1 if all_acc_steps else 0
            ft_epochs = len(ft_val['epoch_accuracy']['values'])
            all_acc_steps.extend(range(ft_start, ft_start + ft_epochs))
            all_acc_values.extend(ft_val['epoch_accuracy']['values'])
        
        if 'epoch_loss' in ft_val:
            ft_start_loss = max(all_loss_steps) + 1 if all_loss_steps else 0
            ft_loss_epochs = len(ft_val['epoch_loss']['values'])
            all_loss_steps.extend(range(ft_start_loss, ft_start_loss + ft_loss_epochs))
            all_loss_values.extend(ft_val['epoch_loss']['values'])
        
        # Plot accuracy
        if all_acc_values:
            # Versión original (semitransparente)
            ax.plot(all_acc_steps, all_acc_values, color=color, alpha=0.3, linewidth=1)
            # Versión suavizada (línea principal)
            smoothed_acc = smooth_curve(all_acc_values)
            ax.plot(all_acc_steps, smoothed_acc, color=color, linewidth=2.5, 
                   label=f'{model_name.upper()} (max: {max(all_acc_values):.4f})')
            
            print(f"{model_name} - Val Accuracy: {len(all_acc_values)} epochs, "
                  f"range: {min(all_acc_values):.4f} - {max(all_acc_values):.4f}")
        
        # Plot loss
        if all_loss_values:
            # Versión original (semitransparente)
            ax2.plot(all_loss_steps, all_loss_values, color=color, alpha=0.3, linewidth=1)
            # Versión suavizada (línea principal)
            smoothed_loss = smooth_curve(all_loss_values)
            ax2.plot(all_loss_steps, smoothed_loss, color=color, linewidth=2.5,
                    label=f'{model_name.upper()} (min: {min(all_loss_values):.4f})')
            
            print(f"{model_name} - Val Loss: {len(all_loss_values)} epochs, "
                  f"range: {min(all_loss_values):.4f} - {max(all_loss_values):.4f}")
    
    # Configurar leyendas
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Ajustar límites del eje Y
    ax.set_ylim([0.7, 1.0])
    ax2.set_ylim([0.0, 0.8])
    
    # Guardar
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'architecture_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nGráfica guardada en: {output_file}")
    plt.close()

def main():
    # Directorio base
    base_dir = '/home/milton/Documents/proyects/TIC_CNN_Modelo_Melanoma/outputs/comparison'
    
    # Modelos a comparar
    models = [
        {'path': 'densenet121_20260109_004724', 'name': 'densenet121'},
        {'path': 'efficientnet-b0_20260109_084143', 'name': 'efficientnet-b0'},
        {'path': 'resnet50_20260110_001917', 'name': 'resnet50'}
    ]
    
    # Extraer métricas de cada modelo
    models_data = []
    for model in models:
        model_path = os.path.join(base_dir, model['path'])
        print(f"\nProcesando {model['name']}...")
        model_data = extract_model_metrics(model_path, model['name'])
        models_data.append(model_data)
    
    # Generar gráficas comparativas
    print("\nGenerando gráficas comparativas...")
    plot_comparison(models_data, base_dir)

if __name__ == '__main__':
    main()
