"""
Model Training Pipeline
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline for ML models
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 5,
        checkpoint_dir: str = "./checkpoints"
    ):
        """
        Initialize ModelTrainer

        Args:
            model_name: Name of the model
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation split ratio
            early_stopping_patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.best_val_acc = 0
        self.no_improve_count = 0

    def build_model(
        self,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> bool:
        """
        Build the model

        Args:
            num_classes: Number of output classes
            input_shape: Input image shape

        Returns:
            True if successful
        """
        try:
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
            from tensorflow.keras.models import Model
            from tensorflow.keras.optimizers import Adam

            logger.info(f"Building {self.model_name} model for {num_classes} classes")

            # Load pre-trained ResNet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )

            # Freeze base model layers
            base_model.trainable = False

            # Add custom top layers
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(num_classes, activation='softmax')(x)

            # Create model
            self.model = Model(inputs=base_model.input, outputs=predictions)

            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            logger.info("Model built successfully")
            self.model.summary()
            return True

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False

    def train(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        val_images: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None
    ) -> bool:
        """
        Train the model

        Args:
            train_images: Training images
            train_labels: Training labels
            val_images: Validation images
            val_labels: Validation labels

        Returns:
            True if successful
        """
        if self.model is None:
            logger.error("Model not built. Call build_model first.")
            return False

        try:
            logger.info(f"Starting training for {self.epochs} epochs")
            logger.info(f"Training set: {len(train_images)} samples")

            if val_images is not None:
                logger.info(f"Validation set: {len(val_images)} samples")
                validation_data = (val_images, val_labels)
            else:
                validation_data = None

            # Train model
            history = self.model.fit(
                train_images,
                train_labels,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=[],
                verbose=1
            )

            # Store history
            self.history['train_loss'] = history.history['loss']
            self.history['train_acc'] = history.history['accuracy']

            if 'val_loss' in history.history:
                self.history['val_loss'] = history.history['val_loss']
                self.history['val_acc'] = history.history['val_accuracy']

            logger.info("Training completed")
            return True

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False

    def save_model(self, model_path: str) -> bool:
        """
        Save trained model

        Args:
            model_path: Path to save model

        Returns:
            True if successful
        """
        try:
            if self.model is None:
                logger.error("No model to save")
                return False

            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def save_checkpoint(self, checkpoint_name: str) -> bool:
        """
        Save model checkpoint

        Args:
            checkpoint_name: Name of checkpoint

        Returns:
            True if successful
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.h5"
            self.save_model(str(checkpoint_path))

            # Save history
            history_path = self.checkpoint_dir / f"{checkpoint_name}_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

            logger.info(f"Checkpoint saved: {checkpoint_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Load trained model

        Args:
            model_path: Path to model file

        Returns:
            True if successful
        """
        try:
            from tensorflow.keras.models import load_model

            self.model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_training_stats(self) -> Dict:
        """
        Get training statistics

        Returns:
            Dictionary with training stats
        """
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "history": self.history,
            "best_val_accuracy": max(self.history['val_acc']) if self.history['val_acc'] else 0
        }
