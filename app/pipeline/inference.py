"""
Inference Pipeline
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Complete inference pipeline for predictions
    """

    def __init__(
        self,
        model=None,
        category_names: List[str] = None,
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize InferencePipeline

        Args:
            model: Trained model
            category_names: Names of output categories
            input_size: Input image size
        """
        self.model = model
        self.category_names = category_names or []
        self.input_size = input_size

    def set_model(self, model):
        """
        Set the model for inference

        Args:
            model: Trained model
        """
        self.model = model

    def set_categories(self, category_names: List[str]):
        """
        Set category names

        Args:
            category_names: List of category names
        """
        self.category_names = category_names

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess image for inference

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Resize
            image = cv2.resize(image, self.input_size)

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Normalize
            image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None

    def predict_single(self, image_path: str) -> Dict:
        """
        Predict on a single image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with predictions and confidence
        """
        if self.model is None:
            logger.error("Model not set")
            return {"error": "Model not set"}

        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)

            # Predict
            predictions = self.model.predict(image_batch)
            predicted_label = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))

            # Get category name
            category_name = (
                self.category_names[predicted_label]
                if predicted_label < len(self.category_names)
                else f"Class_{predicted_label}"
            )

            return {
                "image_path": image_path,
                "predicted_class": int(predicted_label),
                "predicted_category": category_name,
                "confidence": confidence,
                "all_predictions": {
                    self.category_names[i] if i < len(self.category_names) else f"Class_{i}": float(pred)
                    for i, pred in enumerate(predictions[0])
                }
            }

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return {"error": str(e)}

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Predict on multiple images

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction results
        """
        results = []

        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)

        return results

    def predict_from_directory(self, directory: str) -> List[Dict]:
        """
        Predict on all images in directory

        Args:
            directory: Directory containing images

        Returns:
            List of prediction results
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        directory = Path(directory)

        image_paths = [
            str(f) for f in directory.rglob('*')
            if f.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_paths)} images in {directory}")

        return self.predict_batch(image_paths)

    def predict_top_k(
        self,
        image_path: str,
        k: int = 5
    ) -> Dict:
        """
        Get top-k predictions

        Args:
            image_path: Path to image file
            k: Number of top predictions

        Returns:
            Dictionary with top-k predictions
        """
        if self.model is None:
            logger.error("Model not set")
            return {"error": "Model not set"}

        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}

            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)

            # Predict
            predictions = self.model.predict(image_batch)
            predictions = predictions[0]

            # Get top-k
            top_k_indices = np.argsort(predictions)[-k:][::-1]
            top_k_predictions = [
                {
                    "rank": i + 1,
                    "class": int(idx),
                    "category": (
                        self.category_names[idx]
                        if idx < len(self.category_names)
                        else f"Class_{idx}"
                    ),
                    "confidence": float(predictions[idx])
                }
                for i, idx in enumerate(top_k_indices)
            ]

            return {
                "image_path": image_path,
                "top_k_predictions": top_k_predictions
            }

        except Exception as e:
            logger.error(f"Error in top-k prediction: {str(e)}")
            return {"error": str(e)}
