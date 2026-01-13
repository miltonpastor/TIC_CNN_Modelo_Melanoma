# Resumen de Optimizaciones GPU - Pipeline de Datos

## 🎯 Problema Resuelto

**GPU-Util: 0-1%** → **GPU-Util esperado: 70-95%**

## 🔧 Cambios Implementados

### 1. Pipeline tf.data (reemplaza ImageDataGenerator)

**Archivo:** [src/data/preprocessing.py](src/data/preprocessing.py)

```python
# ANTES (CPU-bound):
ImageDataGenerator(...).flow_from_dataframe(...)

# DESPUÉS (GPU-optimized):
dataset = tf.data.Dataset.from_tensor_slices(...)
    .shuffle(10000)                                    # ← Shuffle eficiente
    .map(..., num_parallel_calls=AUTOTUNE)            # ← Carga paralela
    .cache()                                           # ← Cache en memoria (2x-10x speedup)
    .batch(32)                                         # ← Batch antes de augmentation
    .map(augment, num_parallel_calls=AUTOTUNE)        # ← Augmentation GPU paralela
    .prefetch(AUTOTUNE)                                # ← GPU nunca espera (CRÍTICO)
```

**Impacto:**

- ✅ `prefetch(AUTOTUNE)`: GPU entrena batch N mientras CPU prepara batch N+1
- ✅ `cache()`: Epoch 2+ leen de RAM (no disco) → 5-10x más rápido
- ✅ `num_parallel_calls=AUTOTUNE`: TensorFlow ajusta paralelismo automáticamente

---

### 2. Data Augmentation en GPU

**Archivo:** [src/data/preprocessing.py](src/data/preprocessing.py)

```python
# ANTES (NumPy/CPU):
ImageDataGenerator(rotation_range=40, zoom_range=0.25, ...)

# DESPUÉS (Keras layers/GPU):
tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.111),
    layers.RandomZoom((-0.25, 0.25)),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomBrightness((-0.2, 0.2)),
])
```

**Impacto:**

- ✅ Ejecuta en CUDA (Tensor Cores)
- ✅ Procesa batches completos en paralelo
- ✅ Sin transferencias CPU↔GPU

---

### 3. Mixed Precision (FP16)

**Archivo:** [src/main.py](src/main.py)

```python
def configure_gpu():
    # FP16 para Tesla T4 (Tensor Cores)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
```

**Impacto:**

- ✅ Tensor Cores de T4: ~3x más rápido en FP16 vs FP32
- ✅ Reduce memoria 50% → permite batches más grandes
- ✅ Mantiene precisión: variables en FP32, cómputo en FP16

---

## 📊 Resultados Esperados

| Métrica | Antes | Después |
|---------|-------|---------|
| GPU-Util | 0-1% | 70-95% |
| Tiempo/epoch | 180-240s | 60-90s |
| Uso memoria GPU | ~6GB | ~8-10GB |
| Throughput | ~50 img/s | ~200-300 img/s |

---

## ✅ Validez Metodológica Preservada

- ✅ **Batch size = 32** (sin cambios)
- ✅ **Train augmentation ON, Val/Test OFF** (separación correcta)
- ✅ **Augmentations equivalentes** (mismos rangos/parámetros)
- ✅ **Preprocesamiento ImageNet idéntico**
- ✅ **Class weights sin cambios**

---

## 🚀 Uso

### 1. Verificar optimizaciones

```bash
python test_gpu_optimizations.py
```

### 2. Entrenar modelo

```bash
python src/main.py
```

### 3. Monitorear GPU en tiempo real

```bash
watch -n 0.5 nvidia-smi
```

**Esperar:**

- Primera línea: `GPU-Util: 70-95%` ✅
- Memory-Usage: `~8-12GB / 15GB`

---

## 📝 Archivos Modificados

| Archivo | Cambio Principal |
|---------|-----------------|
| [src/data/preprocessing.py](src/data/preprocessing.py) | Pipeline tf.data + augmentation GPU |
| [src/main.py](src/main.py) | Mixed precision + uso de tf.data |
| [src/config/config.py](src/config/config.py) | Batch size restaurado a 32 |
| [src/evaluation/evaluate.py](src/evaluation/evaluate.py) | Soporte tf.data.Dataset |

---

## ⚠️ Notas

1. **Primera época más lenta:** Cache se construye
2. **RAM aumentará ~2-4GB:** Cache almacena datos
3. **Requiere TensorFlow ≥2.8:** Para capas Random*
4. **CUDA 11.x recomendado:** Para Tensor Cores

---

**Documentación completa:** [OPTIMIZACIONES_GPU.md](OPTIMIZACIONES_GPU.md)
