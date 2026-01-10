import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, 
    ModelCheckpoint, TensorBoard, Callback
)
import os
import signal
from config.config import OUTPUT_FOLDER, CLASS_BALANCE_CONFIG
from training.optimizers import focal_loss


class InterruptionHandler(Callback):
    """Guarda el modelo al interceptar Ctrl+C."""
    
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.interrupted = False
        self.current_epoch = 0
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        print("\n⚠️  Interrupción detectada. Guardando modelo...")
        self.interrupted = True
        self.model.stop_training = True
    
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        if self.interrupted:
            self.model.save(self.filepath)
            print(f"✅ Modelo guardado en: {self.filepath}")
            print(f"📍 Época alcanzada: {self.current_epoch}")
            self.interrupted = False

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
        
        # Seleccionar loss function
        if CLASS_BALANCE_CONFIG['use_focal_loss']:
            loss_fn = focal_loss(
                gamma=CLASS_BALANCE_CONFIG['focal_gamma'],
                alpha=CLASS_BALANCE_CONFIG['focal_alpha']
            )
            print(f"   Usando Focal Loss (gamma={CLASS_BALANCE_CONFIG['focal_gamma']}, alpha={CLASS_BALANCE_CONFIG['focal_alpha']})")
        else:
            loss_fn = 'binary_crossentropy'
            print("   Usando Binary Crossentropy")
        
        # Compilar con LR alto
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['initial_lr_head']),
            loss=loss_fn,
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
        
        # Seleccionar loss function (igual que en stage A)
        if CLASS_BALANCE_CONFIG['use_focal_loss']:
            loss_fn = focal_loss(
                gamma=CLASS_BALANCE_CONFIG['focal_gamma'],
                alpha=CLASS_BALANCE_CONFIG['focal_alpha']
            )
        else:
            loss_fn = 'binary_crossentropy'
        
        # Compilar con LR ajustado
        self.model.compile(
            optimizer=Adam(learning_rate=self.config['initial_lr_finetune']),
            loss=loss_fn,
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
        
        # Guardar modelo final (última época)
        final_model_path = os.path.join(OUTPUT_FOLDER, "final_model.h5")
        self.model.save(final_model_path)
        print(f"💾 Modelo final guardado en: {final_model_path}")
        
        return history_b
    
    def continue_fine_tuning(self, train_data, val_data, additional_epochs, learning_rate, class_weight=None):
        """
        Continúa el fine-tuning desde un modelo ya entrenado.
        
        Args:
            train_data: Generador de datos de entrenamiento
            val_data: Generador de datos de validación
            additional_epochs: Número de épocas adicionales a entrenar
            learning_rate: Learning rate para continuar (típicamente más bajo)
            class_weight: Pesos de clase para balanceo
        
        Returns:
            history: Historial de entrenamiento
        """
        print(f"🟢 CONTINUANDO Fine-tuning ({additional_epochs} épocas adicionales)")
        print(f"   Learning rate: {learning_rate}")
        
        # Seleccionar loss function (igual que en otras etapas)
        if CLASS_BALANCE_CONFIG['use_focal_loss']:
            loss_fn = focal_loss(
                gamma=CLASS_BALANCE_CONFIG['focal_gamma'],
                alpha=CLASS_BALANCE_CONFIG['focal_alpha']
            )
        else:
            loss_fn = 'binary_crossentropy'
        
        # Re-compilar con nuevo learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = self._get_callbacks(stage='continued_fine_tuning')
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=additional_epochs,
            callbacks=callbacks,
            class_weight=class_weight
        )
        
        # Guardar modelo final (última época)
        final_model_path = os.path.join(OUTPUT_FOLDER, "final_model.h5")
        self.model.save(final_model_path)
        print(f"💾 Modelo final guardado en: {final_model_path}")
        
        return history
    
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
        
        # Guardar modelo en interrupciones
        interrupt_path = os.path.join(OUTPUT_FOLDER, f"interrupted_{stage}.h5")
        callbacks.append(InterruptionHandler(filepath=interrupt_path))
        
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