# Evaluación del Modelo

Este proyecto separa las métricas de evaluación en dos categorías:

## 📊 Métricas Automáticas (Sin Umbral)

Estas métricas se calculan **automáticamente** en cada ejecución del pipeline y **no requieren un umbral**:

- **AUROC** (Area Under ROC Curve)
- **AUPRC** (Area Under Precision-Recall Curve)  
- **Brier Score** (calibración del modelo)

### Gráficos Automáticos

Se generan en `outputs/<run>/figures/`:

- `roc_curve_*.png` - Curva ROC
- `precision_recall_curve_*.png` - Curva Precision-Recall
- `calibration_curve_*.png` - Curva de calibración

### Resultados Automáticos

Se guardan en `outputs/<run>/results.json`:

```json
{
  "evaluation_metrics": {
    "threshold_independent": {
      "auroc": 0.9257,
      "auprc": 0.6500,
      "brier_score": 0.0909
    }
  }
}
```

## 🎯 Métricas con Umbral (Manuales)

Estas métricas **requieren un umbral específico** y se calculan manualmente ejecutando un script:

- **Accuracy**
- **Precision** (por clase)
- **Recall** (por clase)
- **F1-Score** (por clase)
- **Matriz de Confusión**

### Uso del Script

**Configuración:** Edita las constantes al inicio del archivo `scripts/evaluate_with_threshold.py`:

```python
RUN_DIR = 'resnet50_20251212_082430'  # Nombre del directorio del run
THRESHOLD = 0.5                        # Umbral para clasificación binaria (0.0 - 1.0)
```

**Ejecución:**

```bash
# Ejecutar con las constantes configuradas
python scripts/evaluate_with_threshold.py
```

### Resultados con Umbral

Los resultados se guardan en `outputs/<run>/evaluations/threshold_<valor>/`:

```
outputs/resnet50_20251212_082430/
└── evaluations/
    ├── threshold_0.500/
    │   ├── evaluation_results.json
    │   └── confusion_matrix.png
    ├── threshold_0.300/
    │   ├── evaluation_results.json
    │   └── confusion_matrix.png
    └── threshold_0.700/
        ├── evaluation_results.json
        └── confusion_matrix.png
```

**evaluation_results.json** contiene:

```json
{
  "evaluation_info": {
    "threshold": 0.5,
    "timestamp": "2025-12-13 10:30:00"
  },
  "metrics": {
    "accuracy": 0.8827,
    "per_class": {
      "benign": {
        "precision": 0.9728,
        "recall": 0.8917,
        "f1_score": 0.9305
      },
      "malignant": {
        "precision": 0.5075,
        "recall": 0.8173,
        "f1_score": 0.6262
      }
    }
  },
  "confusion_matrix": {
    "matrix": [[13435, 1632], [376, 1682]],
    "description": "[[TN, FP], [FN, TP]]"
  }
}
```

## 🔄 Workflow Completo

1. **Entrenar el modelo:**

   ```bash
   python src/main.py
   ```

   Esto genera automáticamente las métricas sin umbral.

2. **Evaluar con diferentes umbrales:**

   ```bash
   # Editar las constantes en scripts/evaluate_with_threshold.py y ejecutar
   python scripts/evaluate_with_threshold.py
   
   # Probar threshold 0.3, 0.5, 0.7 modificando la constante THRESHOLD
   ```

3. **Comparar resultados** entre diferentes umbrales revisando los archivos JSON generados.
