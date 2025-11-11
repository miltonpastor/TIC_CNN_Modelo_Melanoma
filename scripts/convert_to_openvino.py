#!/usr/bin/env python3
"""
Script simple para convertir modelos .h5 a formato OpenVINO.
Uso: python convert_to_openvino.py <ruta_al_modelo.h5>
"""

import sys
import os
from pathlib import Path

def convert_h5_to_openvino(h5_path):
    """Convierte un modelo .h5 a formato OpenVINO IR."""
    
    # Verificar que el archivo existe
    if not os.path.exists(h5_path):
        print(f"❌ Error: No se encuentra el archivo {h5_path}")
        return False
    
    try:
        import tensorflow as tf
        import openvino as ov
        
        # Cargar modelo Keras
        model = tf.keras.models.load_model(h5_path)
        
        # Obtener directorio del modelo
        model_dir = os.path.dirname(h5_path)
        model_name = os.path.splitext(os.path.basename(h5_path))[0]
        
        # Crear carpeta openvino dentro del directorio del modelo
        openvino_dir = os.path.join(model_dir, "openvino")
        os.makedirs(openvino_dir, exist_ok=True)
        
        # Ruta de salida para OpenVINO
        output_path = os.path.join(openvino_dir, f"{model_name}.xml")
        
        # Convertir a OpenVINO IR
        ov_model = ov.convert_model(model)
        
        # Guardar modelo en formato IR
        ov.save_model(ov_model, output_path)
        
        print(f"✅ Conversión exitosa: {output_path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error: pip install openvino tensorflow ({e})")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Función principal."""
    
    if len(sys.argv) != 2:
        print("Uso: python convert_to_openvino.py <ruta_al_modelo.h5>")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    success = convert_h5_to_openvino(h5_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
