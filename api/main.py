import json
from pathlib import Path
from typing import Dict, Any
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from openvino.runtime import Core

app = FastAPI(title="Melanoma CNN API", version="1.0.0")

# Cargar configuración del modelo
MODELS_PATH = Path(__file__).parent / "models_v1"
with open(MODELS_PATH / "results.json", "r") as f:
    config = json.load(f)

# Mapeo inverso de labels
LABEL_MAPPING = {v: k for k, v in config["label_mapping"].items()}
INPUT_SHAPE = tuple(config["model_config"]["input_shape"][:2])

# Cargar modelo OpenVINO
ie = Core()
model = ie.read_model(model=str(MODELS_PATH / "best_model.xml"))
compiled_model = ie.compile_model(model=model, device_name="CPU")
output_layer = compiled_model.output(0)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


def preprocess_resnet50(img: np.ndarray) -> np.ndarray:
    """
    Aplica el preprocesamiento de ResNet50 (ImageNet) manualmente.
    Replica el comportamiento de keras.applications.resnet50.preprocess_input
    
    Args:
        img: Imagen en formato RGB como numpy array
        
    Returns:
        Imagen preprocesada para ResNet50
    """
    # Convertir a float32
    img = img.astype(np.float32)
    
    # Convertir RGB a BGR (ResNet50 usa BGR)
    img = img[..., ::-1]
    
    # Restar medias de ImageNet en orden BGR
    mean = [103.939, 116.779, 123.68]
    img[..., 0] -= mean[0]  # B
    img[..., 1] -= mean[1]  # G
    img[..., 2] -= mean[2]  # R
    
    return img


@app.post("/api/v1/cnn/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Leer imagen con PIL (convirtiendo a RGB)
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert('RGB')
    
    # Preprocesar (igual que en entrenamiento)
    img = img.resize(INPUT_SHAPE)
    img = np.array(img)  # Convertir PIL a numpy array
    img = preprocess_resnet50(img)  # Preprocesamiento ResNet50/ImageNet
    img = np.expand_dims(img, axis=0)
    
    # Predicción
    result = compiled_model([img])[output_layer][0][0]
    score = float(result)
    class_idx = int(score > 0.5)
    class_name = LABEL_MAPPING[class_idx]
    
    # Calcular confianza: qué tan lejos está de 0.5 (umbral de decisión)
    # Si score > 0.5, confianza = score; si score < 0.5, confianza = 1 - score
    confidence = score if score > 0.5 else (1 - score)
    confidence_percentage = round(confidence * 100, 2)
    
    return {
        "class": class_idx,
        "class_name": class_name,
        "score": score,
        "confidence": confidence,
        "confidence_percentage": confidence_percentage
    }
