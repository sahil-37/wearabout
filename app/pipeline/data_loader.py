"""
Data Loading and Preprocessing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and preprocess data for training and evaluation
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize DataLoader

        Args:
            data_dir: Directory containing images organized by category
            image_size: Size to resize images to
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.categories = []
        self.category_to_idx = {}
        self.image_paths = []
        self.labels = []

    def load_image_paths(self) -> Dict[str, List[str]]:
        """
        Load image paths from directory structure

        Returns:
            Dictionary mapping categories to image paths
        """
        logger.info(f"Loading images from {self.data_dir}")

        category_images = {}
        idx = 0

        # Walk through directory structure
        for category_dir in sorted(self.data_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            self.categories.append(category_name)
            self.category_to_idx[category_name] = idx
            idx += 1

            # Find all images in category
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
            images = [
                f for f in category_dir.rglob('*')
                if f.suffix.lower() in image_extensions
            ]

            category_images[category_name] = [str(img) for img in images]

            logger.info(f"  {category_name}: {len(images)} images")

        logger.info(f"Total categories: {len(self.categories)}")
        return category_images

    def prepare_dataset(
        self,
        category_images: Dict[str, List[str]]
    ) -> Tuple[List[str], List[int]]:
        """
        Prepare dataset paths and labels

        Args:
            category_images: Dictionary of category -> image paths

        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []

        for category, images in category_images.items():
            label_idx = self.category_to_idx[category]
            for img_path in images:
                image_paths.append(img_path)
                labels.append(label_idx)

        self.image_paths = image_paths
        self.labels = labels

        logger.info(f"Total samples: {len(image_paths)}")
        return image_paths, labels

    def split_dataset(
        self
    ) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """
        Split dataset into train, validation, test sets

        Returns:
            Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
        """
        if not self.image_paths:
            raise ValueError("Dataset not prepared. Call prepare_dataset first.")

        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.image_paths,
            self.labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.labels
        )

        # Second split: train vs val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_val_labels
        )

        logger.info(f"Train samples: {len(train_paths)}")
        logger.info(f"Val samples: {len(val_paths)}")
        logger.info(f"Test samples: {len(test_paths)}")

        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess a single image

        Args:
            image_path: Path to the image

        Returns:
            Preprocessed image array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                return None

            # Resize
            image = cv2.resize(image, self.image_size)

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def load_batch(
        self,
        image_paths: List[str],
        labels: List[int],
        batch_size: int = 32
    ):
        """
        Load images in batches

        Args:
            image_paths: List of image paths
            labels: List of corresponding labels
            batch_size: Batch size

        Yields:
            Tuples of (image_batch, label_batch)
        """
        num_samples = len(image_paths)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)

            batch_images = []
            batch_labels = []

            for i in range(start_idx, end_idx):
                image = self.load_image(image_paths[i])
                if image is not None:
                    batch_images.append(image)
                    batch_labels.append(labels[i])

            if batch_images:
                yield (
                    np.array(batch_images),
                    np.array(batch_labels)
                )

    def get_category_name(self, label_idx: int) -> str:
        """
        Get category name from label index

        Args:
            label_idx: Label index

        Returns:
            Category name
        """
        if 0 <= label_idx < len(self.categories):
            return self.categories[label_idx]
        return "Unknown"

    def get_category_stats(self) -> Dict[str, int]:
        """
        Get statistics about dataset

        Returns:
            Dictionary with category statistics
        """
        stats = {}
        for category in self.categories:
            label_idx = self.category_to_idx[category]
            count = sum(1 for l in self.labels if l == label_idx)
            stats[category] = count
        return stats

    def save_metadata(self, output_path: str) -> bool:
        """
        Save dataset metadata

        Args:
            output_path: Path to save metadata JSON

        Returns:
            True if successful
        """
        try:
            metadata = {
                "categories": self.categories,
                "category_to_idx": self.category_to_idx,
                "num_categories": len(self.categories),
                "total_samples": len(self.image_paths),
                "stats": self.get_category_stats()
            }

            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Metadata saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False
