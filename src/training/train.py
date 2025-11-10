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
        
    def stage_a_head_training(self, train_data, val_data, class_weight=None):
        """
        Etapa A: Entrenar solo la cabeza del modelo.
        LR: 1e-2
        Épocas: 15
        """
        print("🟢 ETAPA A: Head Training (base congelada)")        
        # Congelar base
        self.base.trainable = False
        
        # Compilar con LR alto
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['initial_lr_head']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = self._get_callbacks(stage='head_training', save_model=False)
        
        history_a = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['head_epochs'],
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history_a
    
    def stage_b_fine_tuning(self, train_data, val_data, class_weight=None):
        """
        Etapa B: Fine-tuning de las últimas capas.
        LR: 1e-4
        Épocas: 30
        """
        print("🟢 ETAPA B: Fine-tuning (últimas capas descongeladas)")
        
        # Descongelar últimas N capas 
        self.base.trainable = True
        for layer in self.base.layers[:-self.config['unfreeze_layers']]:
            layer.trainable = False
        
        # Compilar con LR ajustado
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['initial_lr_finetune']),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = self._get_callbacks(stage='fine_tuning')
        
        history_b = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config['finetune_epochs'],
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        return history_b
    
    def _get_callbacks(self, stage, save_model=True):
        """Configura callbacks según la etapa y guarda en outputs."""
        log_dir = os.path.join(OUTPUT_FOLDER, "logs", stage)
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                min_lr=1e-8,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        
        if save_model:
            checkpoint_path = os.path.join(OUTPUT_FOLDER, "best_model.h5")
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks