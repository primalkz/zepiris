"""HTTP client for calling the ML inference microservice.

The main application uses this client to send requests to the separate
ML inference container (running on port 8001 by default).

Example:
    import base64
    import cv2

    client = MLInferenceClient("http://localhost:8001")

    # Load image and convert to base64
    image_bgr = cv2.imread("photo.jpg")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_bytes = cv2.imencode(".jpg", image_rgb)[1].tobytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    nsfw_result = client.detect_nsfw(image_b64)
    client.close()
"""

from __future__ import annotations

from typing import Any

import httpx

from zepiris.schemas.ml_inference import (
    BlurDetectionResult,
    FaceEmbeddingResult,
    ImageQualityAssessmentResult,
    NSFWDetectionResult,
    SpoofDetectionResult,
)


def _upstream_error_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        upstream: Any = response.json()
    except Exception:
        upstream = response.text[:2000]
    return {
        "message": "ml_inference_request_failed",
        "upstream_status": response.status_code,
        "upstream": upstream,
    }


class MLInferenceClient:
    """HTTP client for calling the remote ML inference service.

    Accepts images in base64 format and sends them to the inference service
    as JSON for simpler HTTP communication.
    """

    def __init__(self, base_url: str) -> None:
        """Initialize ML inference client.

        Args:
            base_url: Base URL of the ML inference service, e.g. "http://localhost:8001"
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url)

    def _prepare_image_json(self, image_b64: str) -> dict:
        """Prepare JSON payload for an image in base64 format.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            dict: JSON payload with base64-encoded image
        """
        return {"image_b64": image_b64}

    def detect_nsfw(self, image_b64: str) -> NSFWDetectionResult:
        """Run NSFW detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            NSFWDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/nsfw_check", json=payload)
        response.raise_for_status()
        return NSFWDetectionResult(**response.json())

    def detect_spoof(self, image_b64: str) -> SpoofDetectionResult:
        """Run spoof detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            SpoofDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/spoof_check", json=payload)
        response.raise_for_status()
        return SpoofDetectionResult(**response.json())

    def detect_blur(self, image_b64: str) -> BlurDetectionResult:
        """Run blur detection on an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            BlurDetectionResult: Detection result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/blur_check", json=payload)
        response.raise_for_status()
        return BlurDetectionResult(**response.json())

    def embed_face(self, image_b64: str) -> FaceEmbeddingResult:
        """Generate face embedding from an image.

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            FaceEmbeddingResult: Embedding result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/face/embed", json=payload)
        response.raise_for_status()
        return FaceEmbeddingResult(**response.json())

    def assess_image_quality(self, image_b64: str) -> ImageQualityAssessmentResult:
        """Run combined image quality assessment (NSFW + spoof + blur).

        Args:
            image_b64: Image as base64-encoded string

        Returns:
            ImageQualityAssessmentResult: Combined assessment result
        """

        payload = self._prepare_image_json(image_b64)
        response = self.client.post("/v1/iqa/assess", json=payload)
        response.raise_for_status()
        return ImageQualityAssessmentResult(**response.json())

    def healthz(self) -> dict[str, str]:
        """Check service health.

        Returns:
            dict: Health status response
        """
        response = self.client.get("/healthz")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> MLInferenceClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit."""
        self.close()
