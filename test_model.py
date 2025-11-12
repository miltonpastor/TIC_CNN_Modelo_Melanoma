import sys
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Cargar configuración
with open("outputs/resnet50_20251110_220446/results.json", "r") as f:
    config = json.load(f)

# Cargar modelo
model = keras.models.load_model("outputs/resnet50_20251110_220446/best_model.h5")

# Mapeo de labels
label_mapping = {v: k for k, v in config["label_mapping"].items()}
input_shape = tuple(config["model_config"]["input_shape"][:2])

# Leer y preprocesar imagen (IGUAL QUE EN ENTRENAMIENTO)
image_path = sys.argv[1]
img = image.load_img(image_path, target_size=input_shape)
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
# IMPORTANTE: Usar el mismo preprocesamiento que en entrenamiento (ResNet50)
img = preprocess_input(img)

# Predicción
prediction = model.predict(img, verbose=0)[0][0]
class_idx = int(prediction > 0.5)
class_name = label_mapping[class_idx]

print(f"\nResultado:")
print(f"  Clase: {class_idx}")
print(f"  Nombre: {class_name}")
print(f"  Score: {prediction:.4f}")
