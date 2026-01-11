#!/usr/bin/env python3
"""
Script para verificar que las optimizaciones de memoria funcionan correctamente.
Muestra el uso de memoria en tiempo real durante el proceso de evaluación.
"""

import psutil
import os
import sys
import gc

def get_memory_usage_gb():
    """Retorna el uso de memoria del proceso actual en GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def print_memory_status(label=""):
    """Imprime el estado actual de memoria del sistema y proceso."""
    process_mem = get_memory_usage_gb()
    
    # Memoria del sistema
    vm = psutil.virtual_memory()
    system_total = vm.total / (1024 ** 3)
    system_used = vm.used / (1024 ** 3)
    system_percent = vm.percent
    
    print(f"\n{'='*60}")
    if label:
        print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"🔹 Proceso Python: {process_mem:.2f} GB")
    print(f"🔹 Sistema total: {system_used:.2f} / {system_total:.2f} GB ({system_percent:.1f}%)")
    print(f"{'='*60}\n")

def test_memory_optimization():
    """Prueba las optimizaciones de memoria."""
    print("\n🧪 PRUEBA DE OPTIMIZACIÓN DE MEMORIA\n")
    
    print_memory_status("Inicio del script")
    
    # Simular carga de TensorFlow
    print("⏳ Importando TensorFlow...")
    import tensorflow as tf
    print_memory_status("Después de importar TensorFlow")
    
    # Simular carga de modelo
    print("⏳ Simulando carga de modelo grande...")
    # Crear un modelo dummy grande
    from tensorflow.keras import layers, models
    
    dummy_model = models.Sequential([
        layers.Dense(2048, activation='relu', input_shape=(224*224*3,)),
        layers.Dense(2048, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    dummy_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    print_memory_status("Después de crear modelo dummy")
    
    # Simular procesamiento incremental
    print("⏳ Simulando procesamiento incremental de datos...")
    import pandas as pd
    import numpy as np
    
    # Simular guardado incremental (como save_scores_incremental)
    chunk_size = 1000
    total_samples = 50000
    batches = total_samples // 32
    
    chunk_true = []
    chunk_pred = []
    
    temp_csv = '/tmp/test_scores.csv'
    with open(temp_csv, 'w') as f:
        f.write('true_label,predicted_score\n')
    
    for batch_idx in range(batches):
        # Simular predicción
        batch_true = np.random.randint(0, 2, size=32)
        batch_pred = np.random.random(32)
        
        # Acumular en chunk
        chunk_true.extend(batch_true)
        chunk_pred.extend(batch_pred)
        
        # Escribir y limpiar cuando alcance chunk_size
        if len(chunk_true) >= chunk_size:
            df_chunk = pd.DataFrame({
                'true_label': chunk_true,
                'predicted_score': chunk_pred
            })
            df_chunk.to_csv(temp_csv, mode='a', header=False, index=False)
            
            # LIMPIAR (esto es lo importante)
            chunk_true = []
            chunk_pred = []
            gc.collect()
            
        # Mostrar progreso cada 500 batches
        if (batch_idx + 1) % 500 == 0:
            print(f"  Procesados {batch_idx + 1}/{batches} batches...")
            current_mem = get_memory_usage_gb()
            print(f"    💾 Memoria actual: {current_mem:.2f} GB")
    
    # Escribir chunk final
    if chunk_true:
        df_chunk = pd.DataFrame({
            'true_label': chunk_true,
            'predicted_score': chunk_pred
        })
        df_chunk.to_csv(temp_csv, mode='a', header=False, index=False)
    
    print_memory_status("Después de procesamiento incremental")
    
    # Simular lectura en chunks
    print("⏳ Leyendo scores desde CSV en chunks...")
    y_true = []
    y_pred = []
    
    for chunk in pd.read_csv(temp_csv, chunksize=10000):
        y_true.extend(chunk['true_label'].values)
        y_pred.extend(chunk['predicted_score'].values)
    
    print(f"✅ Leídas {len(y_true)} muestras desde CSV")
    print_memory_status("Después de leer CSV completo")
    
    # Limpiar
    del dummy_model, y_true, y_pred
    gc.collect()
    tf.keras.backend.clear_session()
    
    print_memory_status("Después de limpieza final")
    
    # Limpiar archivo temporal
    os.remove(temp_csv)
    
    print("\n✅ Prueba completada exitosamente!\n")
    print("📝 Notas:")
    print("  - Si la memoria NO crece descontroladamente durante el procesamiento,")
    print("    las optimizaciones están funcionando correctamente.")
    print("  - La memoria debería mantenerse relativamente estable durante el loop.")
    print("  - La limpieza final debería reducir la memoria significativamente.\n")

if __name__ == "__main__":
    # Verificar que psutil está instalado
    try:
        import psutil
    except ImportError:
        print("❌ Error: psutil no está instalado.")
        print("   Instalar con: pip install psutil")
        sys.exit(1)
    
    test_memory_optimization()
