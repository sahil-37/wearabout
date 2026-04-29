"""
MLflow Client Wrapper for experiment tracking and model management
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class MLflowClientWrapper:
    """
    Wrapper for MLflow operations with graceful fallback when MLflow is unavailable.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "buy-me-that-look"
    ):
        """
        Initialize MLflow client.

        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enabled = MLFLOW_AVAILABLE
        self._client = None

        if self.enabled and tracking_uri:
            try:
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._client = MlflowClient(tracking_uri)
                logger.info(f"MLflow connected to {tracking_uri}")
            except Exception as e:
                logger.warning(f"Failed to connect to MLflow: {e}")
                self.enabled = False

    def log_inference(
        self,
        model_name: str,
        latency_ms: float,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
        success: bool = True
    ) -> None:
        """
        Log inference metrics.

        Args:
            model_name: Name of the model
            latency_ms: Inference latency in milliseconds
            input_shape: Shape of input data
            output_shape: Shape of output data
            success: Whether inference was successful
        """
        if not self.enabled:
            return

        try:
            with mlflow.start_run(run_name=f"inference_{model_name}"):
                mlflow.log_metric("latency_ms", latency_ms)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("success", success)

                if input_shape:
                    mlflow.log_param("input_shape", str(input_shape))
                if output_shape:
                    mlflow.log_param("output_shape", str(output_shape))

        except Exception as e:
            logger.warning(f"Failed to log inference: {e}")

    def log_prediction(
        self,
        model_name: str,
        features: Dict[str, Any],
        prediction: Any,
        confidence: Optional[float] = None
    ) -> None:
        """
        Log a single prediction.

        Args:
            model_name: Name of the model
            features: Input features
            prediction: Model prediction
            confidence: Prediction confidence score
        """
        if not self.enabled:
            return

        try:
            with mlflow.start_run(run_name=f"prediction_{model_name}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("prediction", str(prediction))

                if confidence is not None:
                    mlflow.log_metric("confidence", confidence)

                # Log feature summary
                mlflow.log_param("num_features", len(features))

        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

    def get_latest_model_version(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> Optional[str]:
        """
        Get the latest model version from the registry.

        Args:
            model_name: Registered model name
            stage: Model stage (Production, Staging, etc.)

        Returns:
            Model version or None
        """
        if not self.enabled or not self._client:
            return None

        try:
            versions = self._client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
        except Exception as e:
            logger.warning(f"Failed to get model version: {e}")

        return None

    def get_model_uri(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: str = "Production"
    ) -> Optional[str]:
        """
        Get the URI to load a model.

        Args:
            model_name: Registered model name
            version: Specific version (optional)
            stage: Model stage if version not specified

        Returns:
            Model URI or None
        """
        if not self.enabled:
            return None

        try:
            if version:
                return f"models:/{model_name}/{version}"
            return f"models:/{model_name}/{stage}"
        except Exception as e:
            logger.warning(f"Failed to get model URI: {e}")

        return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment info dictionaries
        """
        if not self.enabled or not self._client:
            return []

        try:
            experiments = self._client.search_experiments()
            return [
                {
                    "name": exp.name,
                    "experiment_id": exp.experiment_id,
                    "artifact_location": exp.artifact_location,
                }
                for exp in experiments
            ]
        except Exception as e:
            logger.warning(f"Failed to list experiments: {e}")

        return []

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List runs for an experiment.

        Args:
            experiment_name: Experiment name (uses default if None)
            max_results: Maximum number of runs to return

        Returns:
            List of run info dictionaries
        """
        if not self.enabled or not self._client:
            return []

        try:
            exp_name = experiment_name or self.experiment_name
            experiment = mlflow.get_experiment_by_name(exp_name)

            if not experiment:
                return []

            runs = self._client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results
            )

            return [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                }
                for run in runs
            ]
        except Exception as e:
            logger.warning(f"Failed to list runs: {e}")

        return []


class InferenceTracker:
    """
    Context manager for tracking inference time.
    """

    def __init__(
        self,
        client: MLflowClientWrapper,
        model_name: str
    ):
        self.client = client
        self.model_name = model_name
        self.start_time = None
        self.success = True

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.success = exc_type is None

        self.client.log_inference(
            model_name=self.model_name,
            latency_ms=latency_ms,
            success=self.success
        )

        return False  # Don't suppress exceptions


# Global client instance (lazy initialization)
_mlflow_client: Optional[MLflowClientWrapper] = None


def get_mlflow_client(
    tracking_uri: Optional[str] = None
) -> MLflowClientWrapper:
    """
    Get or create the global MLflow client.

    Args:
        tracking_uri: MLflow tracking URI (optional)

    Returns:
        MLflowClientWrapper instance
    """
    global _mlflow_client

    if _mlflow_client is None:
        from app.config import settings
        uri = tracking_uri or getattr(settings, 'MLFLOW_TRACKING_URI', None)
        _mlflow_client = MLflowClientWrapper(tracking_uri=uri)

    return _mlflow_client
