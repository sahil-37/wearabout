"""
Prometheus metrics middleware for API monitoring
"""

import time
import logging
from typing import Callable
from collections import defaultdict
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Simple metrics collector for API monitoring.
    Thread-safe implementation using basic data structures.
    """

    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_latency_sum = defaultdict(float)
        self.request_latency_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.active_requests = 0
        self.model_inference_time = defaultdict(float)
        self.model_inference_count = defaultdict(int)

    def inc_request_count(self, method: str, endpoint: str, status: int):
        """Increment request counter"""
        key = f"{method}:{endpoint}:{status}"
        self.request_count[key] += 1

    def observe_latency(self, method: str, endpoint: str, latency: float):
        """Record request latency"""
        key = f"{method}:{endpoint}"
        self.request_latency_sum[key] += latency
        self.request_latency_count[key] += 1

    def inc_error_count(self, method: str, endpoint: str, error_type: str):
        """Increment error counter"""
        key = f"{method}:{endpoint}:{error_type}"
        self.error_count[key] += 1

    def observe_model_inference(self, model_name: str, duration: float):
        """Record model inference time"""
        self.model_inference_time[model_name] += duration
        self.model_inference_count[model_name] += 1

    def get_metrics(self) -> str:
        """
        Generate Prometheus-compatible metrics output
        """
        lines = []

        # Request count metrics
        lines.append("# HELP http_requests_total Total HTTP requests")
        lines.append("# TYPE http_requests_total counter")
        for key, count in self.request_count.items():
            method, endpoint, status = key.split(":")
            lines.append(
                f'http_requests_total{{method="{method}",endpoint="{endpoint}",status="{status}"}} {count}'
            )

        # Latency metrics
        lines.append("")
        lines.append("# HELP http_request_duration_seconds HTTP request latency")
        lines.append("# TYPE http_request_duration_seconds summary")
        for key, total in self.request_latency_sum.items():
            method, endpoint = key.split(":")
            count = self.request_latency_count[key]
            lines.append(
                f'http_request_duration_seconds_sum{{method="{method}",'
                f'endpoint="{endpoint}"}} {total:.6f}'
            )
            lines.append(
                f'http_request_duration_seconds_count{{method="{method}",'
                f'endpoint="{endpoint}"}} {count}'
            )

        # Error metrics
        lines.append("")
        lines.append("# HELP http_errors_total Total HTTP errors")
        lines.append("# TYPE http_errors_total counter")
        for key, count in self.error_count.items():
            method, endpoint, error_type = key.split(":")
            lines.append(
                f'http_errors_total{{method="{method}",endpoint="{endpoint}",error_type="{error_type}"}} {count}'
            )

        # Model inference metrics
        if self.model_inference_time:
            lines.append("")
            lines.append("# HELP model_inference_seconds Model inference time")
            lines.append("# TYPE model_inference_seconds summary")
            for model_name, total in self.model_inference_time.items():
                count = self.model_inference_count[model_name]
                lines.append(
                    f'model_inference_seconds_sum{{model="{model_name}"}} {total:.6f}'
                )
                lines.append(
                    f'model_inference_seconds_count{{model="{model_name}"}} {count}'
                )

        # Active requests (gauge)
        lines.append("")
        lines.append("# HELP http_requests_active Active HTTP requests")
        lines.append("# TYPE http_requests_active gauge")
        lines.append(f"http_requests_active {self.active_requests}")

        return "\n".join(lines)


# Global metrics collector instance
metrics = MetricsCollector()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics for all requests
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)

        # Track active requests
        metrics.active_requests += 1

        # Record start time
        start_time = time.perf_counter()

        # Get endpoint path (normalize dynamic paths)
        endpoint = self._normalize_path(request.url.path)
        method = request.method

        try:
            # Process request
            response = await call_next(request)

            # Record metrics
            duration = time.perf_counter() - start_time
            metrics.inc_request_count(method, endpoint, response.status_code)
            metrics.observe_latency(method, endpoint, duration)

            # Track errors
            if response.status_code >= 400:
                error_type = "client_error" if response.status_code < 500 else "server_error"
                metrics.inc_error_count(method, endpoint, error_type)

            return response

        except Exception:
            # Record error
            duration = time.perf_counter() - start_time
            metrics.inc_request_count(method, endpoint, 500)
            metrics.observe_latency(method, endpoint, duration)
            metrics.inc_error_count(method, endpoint, "exception")
            raise

        finally:
            metrics.active_requests -= 1

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to avoid high cardinality metrics.
        Replace dynamic segments with placeholders.
        """
        # Remove trailing slash
        path = path.rstrip("/") or "/"

        # Common patterns to normalize
        # e.g., /api/v1/products/123 -> /api/v1/products/{id}
        parts = path.split("/")
        normalized = []

        for part in parts:
            # Check if part looks like an ID (numeric or UUID-like)
            if part.isdigit():
                normalized.append("{id}")
            elif len(part) == 36 and "-" in part:  # UUID-like
                normalized.append("{uuid}")
            else:
                normalized.append(part)

        return "/".join(normalized)


async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """
    Endpoint to expose Prometheus metrics
    """
    return PlainTextResponse(
        content=metrics.get_metrics(),
        media_type="text/plain; charset=utf-8"
    )


def track_model_inference(model_name: str):
    """
    Decorator to track model inference time

    Usage:
        @track_model_inference("feature_extraction")
        def extract_features(image):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start_time
                metrics.observe_model_inference(model_name, duration)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start_time
                metrics.observe_model_inference(model_name, duration)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# Example: Add model inference tracking
class ModelMetrics:
    """Helper class for tracking model-specific metrics"""

    @staticmethod
    def record_inference(model_name: str, duration: float):
        """Record model inference time"""
        metrics.observe_model_inference(model_name, duration)

    @staticmethod
    def record_feature_extraction(duration: float):
        """Record feature extraction time"""
        metrics.observe_model_inference("feature_extraction", duration)

    @staticmethod
    def record_object_detection(duration: float):
        """Record object detection time"""
        metrics.observe_model_inference("object_detection", duration)

    @staticmethod
    def record_recommendation(duration: float):
        """Record recommendation generation time"""
        metrics.observe_model_inference("recommendation", duration)
