#!/usr/bin/env python3
"""Análisis detallado del validation loss desde TensorBoard logs."""

import tensorflow as tf
from glob import glob
from collections import defaultdict

def read_validation_loss(log_dir):
    """Lee el validation loss de los archivos de eventos."""
    
    event_files = glob(f'{log_dir}/**/validation/events.out.tfevents.*', recursive=True)
    
    results = {}
    
    for event_file in sorted(event_files):
        stage = 'head_training' if 'head_training' in event_file else 'fine_tuning'
        
        epoch_losses = []
        
        for record in tf.data.TFRecordDataset(event_file):
            event = tf.compat.v1.Event.FromString(record.numpy())
            
            for value in event.summary.value:
                if value.tag == 'epoch_loss':
                    # Intentar obtener el valor del tensor si simple_value es 0
                    loss_value = value.simple_value
                    if loss_value == 0.0 and value.HasField('tensor'):
                        # Decodificar el tensor
                        tensor_proto = value.tensor
                        loss_value = tf.make_ndarray(tensor_proto).item()
                    
                    epoch_losses.append({
                        'epoch': event.step,
                        'loss': loss_value
                    })
        
        # Ordenar por época
        epoch_losses.sort(key=lambda x: x['epoch'])
        results[stage] = epoch_losses
    
    return results

def analyze_no_improvement(losses, stage_name):
    """Analiza cuántas épocas consecutivas el loss no mejoró."""
    
    print(f"\n{'='*80}")
    print(f"{stage_name.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    if not losses:
        print("No se encontraron datos de validation loss")
        return
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 0
    total_epochs = len(losses)
    
    improvement_periods = []
    
    print(f"{'Época':<8} {'Val Loss':<12} {'Estado'}")
    print("-" * 60)
    
    for item in losses:
        epoch = item['epoch']
        loss = item['loss']
        
        if loss < best_loss:
            # Mejoró
            if epochs_without_improvement > 0:
                improvement_periods.append({
                    'count': epochs_without_improvement,
                    'ended_at': epoch
                })
                max_epochs_without_improvement = max(max_epochs_without_improvement, epochs_without_improvement)
            
            best_loss = loss
            epochs_without_improvement = 0
            status = "✓ MEJOR"
        else:
            # No mejoró
            epochs_without_improvement += 1
            status = f"  Sin mejorar ({epochs_without_improvement} épocas)"
        
        print(f"{epoch:<8} {loss:<12.6f} {status}")
    
    # Contar épocas sin mejora al final
    if epochs_without_improvement > 0:
        improvement_periods.append({
            'count': epochs_without_improvement,
            'ended_at': 'final'
        })
        max_epochs_without_improvement = max(max_epochs_without_improvement, epochs_without_improvement)
    
    print(f"\n{'='*80}")
    print("📊 RESUMEN")
    print(f"{'='*80}")
    print(f"  • Mejor validation loss: {best_loss:.6f}")
    print(f"  • Total de épocas: {total_epochs}")
    print(f"  • Máximo de épocas consecutivas sin mejora: {max_epochs_without_improvement}")
    print(f"  • Épocas sin mejora al final del entrenamiento: {epochs_without_improvement}")
    
    if improvement_periods:
        print(f"\n  Periodos sin mejora:")
        for i, period in enumerate(improvement_periods, 1):
            ended = f"época {period['ended_at']}" if isinstance(period['ended_at'], int) else period['ended_at']
            print(f"    {i}. {period['count']} épocas (hasta {ended})")
    
    print(f"{'='*80}\n")
    
    return max_epochs_without_improvement

def main():
    log_dir = '/home/milton/Documents/proyects/TIC_CNN_Modelo_Melanoma/outputs/resnet50_20260103_235009/logs'
    
    print("📁 Analizando Validation Loss desde TensorBoard logs")
    print(f"   Directorio: {log_dir}\n")
    
    results = read_validation_loss(log_dir)
    
    max_no_improvement = {}
    
    for stage in ['head_training', 'fine_tuning']:
        if stage in results:
            max_no_improvement[stage] = analyze_no_improvement(results[stage], stage)
    
    # Resumen final
    print("\n" + "="*80)
    print("🎯 RESUMEN GENERAL")
    print("="*80)
    for stage, max_count in max_no_improvement.items():
        print(f"  {stage.replace('_', ' ').title()}: {max_count} épocas máximo sin mejora")
    print("="*80)

if __name__ == "__main__":
    main()
