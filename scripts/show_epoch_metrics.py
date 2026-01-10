"""
Script para mostrar accuracy, loss y AUC de cada época desde los logs de TensorBoard
"""

import os
import sys
from glob import glob
import tensorflow as tf
from collections import defaultdict
import argparse


def read_epoch_metrics(log_dir):
    """
    Lee los archivos de eventos de TensorBoard y extrae métricas por época
    
    Args:
        log_dir: Directorio con los logs de TensorBoard
        
    Returns:
        dict: Métricas organizadas por fase (train/validation) y época
    """
    metrics = {
        'head_training': {'train': defaultdict(dict), 'validation': defaultdict(dict)},
        'fine_tuning': {'train': defaultdict(dict), 'validation': defaultdict(dict)}
    }
    
    # Buscar archivos de eventos en todas las subcarpetas
    event_files = glob(f'{log_dir}/**/events.out.tfevents.*', recursive=True)
    
    if not event_files:
        print(f"❌ No se encontraron archivos de eventos en {log_dir}")
        return None
    
    print(f"📂 Encontrados {len(event_files)} archivos de eventos\n")
    
    for event_file in event_files:
        # Determinar fase y tipo (train/validation)
        phase = None
        split_type = None
        
        if 'head_training' in event_file:
            phase = 'head_training'
        elif 'fine_tuning' in event_file:
            phase = 'fine_tuning'
        else:
            continue
            
        if '/train/' in event_file:
            split_type = 'train'
        elif '/validation/' in event_file:
            split_type = 'validation'
        else:
            continue
        
        # Leer eventos
        try:
            for record in tf.data.TFRecordDataset(event_file):
                event = tf.compat.v1.Event.FromString(record.numpy())
                
                for value in event.summary.value:
                    tag = value.tag
                    
                    # Buscar métricas epoch_
                    if tag.startswith('epoch_'):
                        metric_name = tag.replace('epoch_', '')
                        step = event.step
                        
                        # Extraer valor
                        if value.HasField('simple_value'):
                            metric_value = value.simple_value
                        elif value.HasField('tensor'):
                            tensor_proto = value.tensor
                            tensor = tf.make_ndarray(tensor_proto)
                            if tensor.size == 1:
                                metric_value = float(tensor)
                            else:
                                continue
                        else:
                            continue
                        
                        # Guardar métrica por época
                        metrics[phase][split_type][step][metric_name] = metric_value
                        
        except Exception as e:
            print(f"⚠️  Error leyendo {event_file}: {e}")
            continue
    
    return metrics


def display_metrics(metrics, phase_filter=None):
    """
    Muestra las métricas en formato tabla
    
    Args:
        metrics: Diccionario con las métricas
        phase_filter: Filtrar por fase específica ('head_training' o 'fine_tuning')
    """
    if not metrics:
        return
    
    phases = [phase_filter] if phase_filter else ['head_training', 'fine_tuning']
    
    for phase in phases:
        if phase not in metrics:
            continue
            
        phase_data = metrics[phase]
        
        if not phase_data['train'] and not phase_data['validation']:
            continue
        
        print(f"\n{'='*80}")
        print(f"📊 {phase.upper().replace('_', ' ')}")
        print(f"{'='*80}\n")
        
        # Combinar épocas de train y validation
        all_epochs = sorted(set(list(phase_data['train'].keys()) + list(phase_data['validation'].keys())))
        
        if not all_epochs:
            print("No hay datos disponibles\n")
            continue
        
        # Encabezado
        print(f"{'Época':<8} {'Tipo':<12} {'Loss':<12} {'Accuracy':<12} {'AUC':<12}")
        print("-" * 80)
        
        for epoch in all_epochs:
            # Mostrar train
            if epoch in phase_data['train']:
                train_data = phase_data['train'][epoch]
                loss = train_data.get('loss', train_data.get('Loss', 'N/A'))
                acc = train_data.get('accuracy', train_data.get('Accuracy', 'N/A'))
                auc = train_data.get('auc', train_data.get('AUC', 'N/A'))
                
                # Formatear valores
                loss_str = f"{loss:.6f}" if isinstance(loss, float) else str(loss)
                acc_str = f"{acc:.6f}" if isinstance(acc, float) else str(acc)
                auc_str = f"{auc:.6f}" if isinstance(auc, float) else str(auc)
                
                print(f"{epoch:<8} {'Train':<12} {loss_str:<12} {acc_str:<12} {auc_str:<12}")
            
            # Mostrar validation (usa los mismos nombres que train)
            if epoch in phase_data['validation']:
                val_data = phase_data['validation'][epoch]
                loss = val_data.get('loss', val_data.get('Loss', 'N/A'))
                acc = val_data.get('accuracy', val_data.get('Accuracy', 'N/A'))
                auc = val_data.get('auc', val_data.get('AUC', 'N/A'))
                
                # Formatear valores
                loss_str = f"{loss:.6f}" if isinstance(loss, float) else str(loss)
                acc_str = f"{acc:.6f}" if isinstance(acc, float) else str(acc)
                auc_str = f"{auc:.6f}" if isinstance(auc, float) else str(auc)
                
                print(f"{epoch:<8} {'Validation':<12} {loss_str:<12} {acc_str:<12} {auc_str:<12}")
            
            if epoch < all_epochs[-1]:  # No imprimir línea después de la última época
                print()


def main():
    parser = argparse.ArgumentParser(
        description='Muestra métricas (acc, loss, AUC) de cada época desde logs de TensorBoard'
    )
    parser.add_argument(
        'log_dir',
        help='Directorio con los logs de TensorBoard'
    )
    parser.add_argument(
        '--phase',
        choices=['head_training', 'fine_tuning', 'all'],
        default='all',
        help='Filtrar por fase específica (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio
    if not os.path.exists(args.log_dir):
        print(f"❌ Error: El directorio {args.log_dir} no existe")
        sys.exit(1)
    
    # Leer métricas
    print(f"🔍 Leyendo métricas desde: {args.log_dir}")
    metrics = read_epoch_metrics(args.log_dir)
    
    if not metrics:
        print("❌ No se pudieron leer las métricas")
        sys.exit(1)
    
    # Mostrar métricas
    phase_filter = None if args.phase == 'all' else args.phase
    display_metrics(metrics, phase_filter)
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
