import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard
)
import os
from config.config import OUTPUT_FOLDER

class TwoStageTrainer:
    """Entrenador con estrategia de Head Training + Fine-tuning."""
    
    def __init__(self, model, base, config):
        self.model = model
        self.base = base
        self.config = config
        
    def stage_a_head_training(self, train_data, val_data):
        """
        Etapa A: Entrenar solo la cabeza del modelo.
        LR: 1e-3 a 1e-4
        Épocas: 5-10
        """
        print("🟢 ETAPA A: Head Training (base congelada)")        
        # Congelar base
        self.base.trainable = False
        
        # Compilar con LR moderado
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = self._get_callbacks(stage='head_training')
        
        history_a = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['head_epochs'],
            callbacks=callbacks
        )
        
        return history_a
    
    def stage_b_fine_tuning(self, train_data, val_data, unfreeze_layers=20):
        """
        Etapa B: Fine-tuning de las últimas capas.
        LR: 1e-5 a 1e-6
        """
        print("🟢 ETAPA B: Fine-tuning (últimas capas descongeladas)")
        
        # Descongelar últimas N capas
        self.base.trainable = True
        for layer in self.base.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Compilar con LR bajo
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = self._get_callbacks(stage='fine_tuning')
        
        history_b = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['finetune_epochs'],
            callbacks=callbacks
        )
        
        return history_b
    
    def _get_callbacks(self, stage):
        """Configura callbacks según la etapa y guarda en outputs."""
        checkpoint_path = os.path.join(OUTPUT_FOLDER, "checkpoints", f"{stage}_best.h5")
        log_dir = os.path.join(OUTPUT_FOLDER, "logs", stage)
        
        # Crear carpetas si no existen
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]