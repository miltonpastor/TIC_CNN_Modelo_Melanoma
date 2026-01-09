# Guía de Instalación y Configuración GPU (NVIDIA RTX 3060)

Esta guía te ayudará a configurar tu NVIDIA GeForce RTX 3060 para entrenar modelos de Deep Learning con TensorFlow.

## Requisitos Previos

- NVIDIA GeForce RTX 3060 (12GB VRAM)
- Ubuntu 22.04 o superior
- Python 3.10+

## 1. Instalar NVIDIA Drivers

Primero, verifica si ya tienes drivers instalados:

```bash
nvidia-smi
```

Si no funciona o necesitas actualizar, instala los drivers:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

Después del reinicio, verifica nuevamente:

```bash
nvidia-smi
```

## 2. Instalar CUDA Toolkit y cuDNN

TensorFlow 2.16.2 requiere CUDA 12.x. Instala CUDA Toolkit 12.3:

```bash
# Descargar e instalar el keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Actualizar repositorios e instalar CUDA
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

Agregar CUDA al PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verificar instalación de CUDA:

```bash
nvcc --version
```

## 3. Configurar el Entorno Python

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -r requirements.txt

# Instalar TensorFlow con soporte CUDA
pip install tensorflow[and-cuda]
```

## 4. Verificar Configuración GPU

Ejecuta este comando para verificar que TensorFlow detecta tu GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Salida esperada:**
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

También puedes verificar la versión de TensorFlow y CUDA:

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('CUDA:', tf.test.is_built_with_cuda()); print('GPU disponible:', tf.test.is_gpu_available())"
```

## 5. Entrenar el Modelo

Una vez configurado todo, puedes entrenar el modelo:

### Opción A: Entrenamiento desde cero

```bash
# Desde el directorio raíz del proyecto
python src/main.py
```

### Opción B: Continuar entrenamiento existente

```bash
# Edita primero scripts/continue_training.py con la ruta del modelo
python scripts/continue_training.py
```

## Monitoreo y Optimización

### Monitorear uso de GPU en tiempo real

```bash
watch -n 1 nvidia-smi
```

### Visualizar entrenamiento con TensorBoard

```bash
tensorboard --logdir=outputs/resnet50_*/logs
```

Abre en el navegador: http://localhost:6006

## Solución de Problemas

### Error: Out of Memory (OOM)

Si encuentras errores de memoria GPU, reduce el `batch_size` en `src/config/config.py`:

```python
BATCH_SIZE = 16  # Prueba valores menores: 8, 4
```

### GPU no detectada

Verifica que los drivers estén correctamente instalados:

```bash
nvidia-smi
nvcc --version
```

### Versión de CUDA incompatible

Asegúrate de tener CUDA 12.x instalado. TensorFlow 2.16.2 requiere esta versión.

```bash
nvcc --version
```

## Notas Adicionales

- **VRAM disponible:** RTX 3060 tiene 12GB, suficiente para este proyecto
- **Batch size recomendado:** 32 (ajustar según uso de memoria)
- **Uso típico de VRAM:** ~8-10GB durante entrenamiento
- **Logs:** Los logs de TensorBoard se guardan en `outputs/resnet50_*/logs/`

## Resultados

Después del entrenamiento, encontrarás:

- Modelo entrenado: `outputs/resnet50_*/best_model.h5`
- Métricas: `outputs/resnet50_*/results.json`
- Gráficos: `outputs/resnet50_*/figures/`
- Logs: `outputs/resnet50_*/logs/`

Ver [README.md](README.md) para más detalles sobre la estructura de outputs.