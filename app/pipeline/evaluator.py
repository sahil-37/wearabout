"""
Model Evaluation and Testing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and test trained models
    """

    def __init__(self, model=None):
        """
        Initialize ModelEvaluator

        Args:
            model: Pre-trained model (optional)
        """
        self.model = model
        self.predictions = None
        self.true_labels = None
        self.metrics = {}

    def set_model(self, model):
        """
        Set the model to evaluate

        Args:
            model: Model to evaluate
        """
        self.model = model

    def evaluate(
        self,
        test_images: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict:
        """
        Evaluate model on test set

        Args:
            test_images: Test images
            test_labels: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not set")
            return {}

        try:
            logger.info(f"Evaluating model on {len(test_images)} samples")

            # Get predictions
            predictions = self.model.predict(test_images)
            predicted_labels = np.argmax(predictions, axis=1)

            # Store for later use
            self.predictions = predicted_labels
            self.true_labels = test_labels

            # Calculate metrics
            accuracy = np.mean(predicted_labels == test_labels)

            self.metrics = {
                "accuracy": float(accuracy),
                "total_samples": len(test_labels),
                "correct_predictions": int(np.sum(predicted_labels == test_labels)),
                "incorrect_predictions": int(len(test_labels) - np.sum(predicted_labels == test_labels))
            }

            logger.info(f"Accuracy: {accuracy:.4f}")
            return self.metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}

    def get_per_class_metrics(
        self,
        num_classes: int,
        category_names: List[str]
    ) -> Dict:
        """
        Calculate per-class metrics

        Args:
            num_classes: Number of classes
            category_names: Names of categories

        Returns:
            Dictionary with per-class metrics
        """
        if self.predictions is None or self.true_labels is None:
            logger.error("No predictions available. Call evaluate first.")
            return {}

        try:
            per_class_metrics = {}

            for class_idx in range(num_classes):
                class_mask = self.true_labels == class_idx

                if not np.any(class_mask):
                    continue

                class_predictions = self.predictions[class_mask]
                class_true = self.true_labels[class_mask]

                accuracy = np.mean(class_predictions == class_true)
                total = np.sum(class_mask)
                correct = np.sum(class_predictions == class_true)

                class_name = category_names[class_idx] if class_idx < len(category_names) else f"Class_{class_idx}"

                per_class_metrics[class_name] = {
                    "accuracy": float(accuracy),
                    "total_samples": int(total),
                    "correct_predictions": int(correct)
                }

            return per_class_metrics

        except Exception as e:
            logger.error(f"Error calculating per-class metrics: {str(e)}")
            return {}

    def get_confusion_matrix(self, num_classes: int) -> np.ndarray:
        """
        Calculate confusion matrix

        Args:
            num_classes: Number of classes

        Returns:
            Confusion matrix
        """
        if self.predictions is None or self.true_labels is None:
            logger.error("No predictions available")
            return np.array([])

        try:
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

            for true_label, pred_label in zip(self.true_labels, self.predictions):
                confusion_matrix[true_label, pred_label] += 1

            return confusion_matrix

        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {str(e)}")
            return np.array([])

    def print_metrics(self, category_names: List[str] = None):
        """
        Print evaluation metrics

        Args:
            category_names: Names of categories
        """
        logger.info("=" * 60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)

        if self.metrics:
            logger.info(f"Overall Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"Correct: {self.metrics['correct_predictions']}/{self.metrics['total_samples']}")

        if category_names and self.predictions is not None:
            per_class = self.get_per_class_metrics(len(category_names), category_names)
            if per_class:
                logger.info("\nPer-Class Metrics:")
                for class_name, metrics_dict in per_class.items():
                    logger.info(f"  {class_name}:")
                    logger.info(f"    Accuracy: {metrics_dict['accuracy']:.4f}")
                    logger.info(f"    Samples: {metrics_dict['total_samples']}")

    def save_results(
        self,
        output_dir: str,
        category_names: List[str] = None
    ) -> bool:
        """
        Save evaluation results to file

        Args:
            output_dir: Directory to save results
            category_names: Names of categories

        Returns:
            True if successful
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "overall_metrics": self.metrics,
                "per_class_metrics": self.get_per_class_metrics(
                    len(category_names) if category_names else 0,
                    category_names or []
                )
            }

            results_path = output_dir / "evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to {results_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
