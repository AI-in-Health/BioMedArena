"""Adapter for torchxrayvision — chest X-ray classification (18 pathologies)."""

from __future__ import annotations

import asyncio
from typing import Any

from harness.adapter_base import AdapterBase

try:
    import torchxrayvision as xrv
    import torch
    import numpy as np
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class TorchXRVAdapter(AdapterBase):
    name = "torchxrv"
    modality = "imaging"
    description = "Chest X-ray pathology classification (18 findings) via torchxrayvision."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        if not _AVAILABLE:
            self.mark_unavailable("torchxrayvision not installed (pip install torchxrayvision)")
            return
        self._model = None

    def capabilities(self) -> list[str]:
        return ["chest_xray", "cxr_classification", "pathology_detection", "medical_imaging"]

    def _load_model(self):
        if self._model is None:
            self._model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self._model.eval()
        return self._model

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.available:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        ctx = context or {}
        image_path = ctx.get("image_path") or ctx.get("xray_path")
        if not image_path:
            return self.result(
                answer="No image provided. Set 'image_path' in context to a chest X-ray file.",
                confidence=0.1,
            )

        try:
            result = await asyncio.to_thread(self._predict, image_path)
            return result
        except Exception as exc:
            return self.result(answer=f"Prediction error: {exc}", confidence=0.0)

    def _predict(self, image_path: str) -> dict[str, Any]:
        import skimage.io

        model = self._load_model()
        img = xrv.datasets.normalize(
            skimage.io.imread(image_path), maxval=255, reshape=True
        )
        img = torch.from_numpy(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(img)

        pathologies = model.pathologies
        scores = preds[0].numpy()
        findings = {p: round(float(s), 3) for p, s in zip(pathologies, scores)}
        positive = {p: s for p, s in findings.items() if s > 0.5}

        if positive:
            summary_parts = [f"- {p}: {s:.1%}" for p, s in sorted(positive.items(), key=lambda x: -x[1])]
            answer = "**Chest X-ray findings (>50% probability):**\n" + "\n".join(summary_parts)
        else:
            answer = "No significant pathology detected (all findings <50% probability)."

        return self.result(
            answer=answer,
            evidence=[f"{p}: {s:.1%}" for p, s in positive.items()],
            confidence=max(scores) if len(scores) > 0 else 0.0,
            raw=findings,
        )
