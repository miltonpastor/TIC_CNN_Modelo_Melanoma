# Optimizaciones para EC2 con Tesla T4

## ✅ Solución aplicada al problema OOM

### Problema

El proceso se mataba (`Killed`) al guardar los scores de test debido a que se cargaba todo el dataset en memoria RAM:

```python
# ❌ ANTES: Cargaba TODOS los datos en memoria
y_true = tf.concat([y for x, y in test_data], axis=0).numpy()
```

### Solución

Se optimizó el código para procesar y guardar datos **en batches incrementales**:

```python
# ✅ AHORA: Procesa batch por batch, guarda incrementalmente
for batch_x, batch_y in dataset:
    batch_pred = model.predict(batch_x, verbose=0).flatten()
    batch_df.to_csv(scores_path, mode='a', header=False, index=False)
```

### Archivos modificados

- [`src/evaluation/evaluate.py`](src/evaluation/evaluate.py): Nueva función `save_scores_incremental()`
- [`src/main.py`](src/main.py): Usa `save_scores_incremental()` para validation y test

### Mejoras adicionales

1. **Liberación de memoria**: Cada 100 batches se limpia la sesión de Keras
2. **Progress tracking**: Muestra progreso cada 50 batches
3. **Escritura incremental**: CSV se escribe en modo append, evitando acumular en memoria

---

## 🔧 Configuraciones recomendadas para EC2

### 1. Verificar memoria disponible

```bash
# Ver memoria RAM disponible
free -h

# Monitorear uso de memoria en tiempo real
watch -n 1 free -h
```

### 2. Reducir BATCH_SIZE si es necesario

Si aún tienes problemas de memoria, edita [`src/config/config.py`](src/config/config.py):

```python
# Reducir de 32 a 16 o 8 si tienes OOM
BATCH_SIZE = 16  # Usar 16 o 8 en lugar de 32
```

**Trade-offs:**

- ✅ **Menor batch size**: Menos uso de memoria RAM/VRAM
- ❌ **Menor batch size**: Entrenamiento más lento (más iteraciones)

### 3. Desactivar cache en datasets grandes

Si el dataset de test es muy grande (>10GB), desactiva el cache en [`src/main.py`](src/main.py):

```python
# Test: sin cache si el dataset es muy grande
test_dataset = create_tf_dataset(
    test_df,
    preprocess_fn=preprocess_fn,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment=False,
    cache=False  # Cambiar de True a False
)
```

### 4. Monitorear GPU durante entrenamiento

```bash
# Terminal 1: Ejecutar entrenamiento
python src/main.py

# Terminal 2: Monitorear GPU en tiempo real
watch -n 1 nvidia-smi
```

### 5. Configurar swap (si no tienes suficiente RAM)

```bash
# Verificar swap actual
free -h

# Crear archivo swap de 8GB (ajustar según necesidad)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Hacer permanente (opcional)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 📊 Rendimiento esperado en Tesla T4

### Configuración actual (optimizada)

- **Mixed Precision**: ✅ Habilitado (FP16)
- **Memory Growth**: ✅ Habilitado
- **XLA**: ✅ Habilitado
- **Batch Size**: 32
- **Cache**: Habilitado en val/test

### Tiempos estimados (dataset completo ~17k imágenes)

- **Head training (5 epochs)**: ~10-15 min
- **Fine-tuning (20 epochs)**: ~40-60 min
- **Evaluation (test)**: ~2-3 min
- **Guardado de scores**: <1 min (ahora optimizado)

---

## 🐛 Troubleshooting

### Error: `Killed` durante predicciones

**Causa**: OOM (Out of Memory) en RAM
**Solución**: ✅ Ya implementada en el código

### Error: `ResourceExhaustedError` en GPU

**Causa**: VRAM insuficiente en GPU
**Solución**:

1. Reducir `BATCH_SIZE` a 16 o 8
2. Verificar que `mixed_precision=True` en config

### Error: Proceso muy lento

**Causa**: CPU bottleneck o I/O lento
**Solución**:

1. Verificar que se esté usando GPU: `nvidia-smi`
2. Usar disco SSD en lugar de HDD
3. Verificar `prefetch_buffer_size='AUTOTUNE'` en config

### Warnings de TensorFlow

Para reducir verbosidad, ejecuta con:

```bash
TF_CPP_MIN_LOG_LEVEL=2 python src/main.py
```

---

## 📝 Notas importantes

1. **No interrumpir durante guardado**: El proceso ahora guarda incrementalmente, pero aún así espera a que termine
2. **Espacio en disco**: Asegúrate de tener ~5-10GB libres en disco para logs, modelos y scores
3. **Instancia recomendada**: `g4dn.xlarge` (1x Tesla T4, 16GB RAM, 4 vCPUs)
4. **EBS optimizado**: Usa volumen gp3 con al menos 100 IOPS para mejor I/O
