"""MONAI TOOL_SPECS — medical imaging preprocessing ops.

Lightweight wrappers around MONAI transforms. Keeps the LLM-facing
surface small: load + normalise + metadata. Heavy workflows
(segmentation, sliding-window inference, DiceMetric) should go
through a future MONAIAdapter.
"""

from __future__ import annotations

from typing import Any


MONAI_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "medical_image_metadata",
            "description": (
                "Load a medical image (NIfTI / DICOM series dir / PNG) "
                "via MONAI and return shape, dtype, min/max/mean pixel "
                "statistics, and image spacing if available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Filesystem path to a .nii/.nii.gz/.dcm dir/.png",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "medical_image_normalize",
            "description": (
                "Apply intensity normalisation to a medical image and "
                "return the before/after statistics. Does not save the "
                "normalised image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["scale_intensity", "normalize_intensity"],
                        "default": "scale_intensity",
                    },
                },
                "required": ["path"],
            },
        },
    },
]


MONAI_TOOL_NAMES = {"medical_image_metadata", "medical_image_normalize"}


def _metadata_sync(path: str) -> dict[str, Any]:
    import numpy as np
    from monai.transforms import LoadImage
    loader = LoadImage(image_only=False)
    img, meta = loader(path)
    arr = np.asarray(img)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "spacing": list(meta.get("pixdim", [])[:5]) if isinstance(meta, dict) else [],
    }


def _normalize_sync(path: str, method: str = "scale_intensity") -> dict[str, Any]:
    import numpy as np
    from monai.transforms import LoadImage, ScaleIntensity, NormalizeIntensity
    img, _ = LoadImage(image_only=False)(path)
    arr = np.asarray(img, dtype=np.float32)
    before = {"min": float(arr.min()), "max": float(arr.max()),
                "mean": float(arr.mean())}
    t = ScaleIntensity() if method == "scale_intensity" else NormalizeIntensity()
    out = t(arr)
    out_arr = np.asarray(out)
    after = {"min": float(out_arr.min()), "max": float(out_arr.max()),
               "mean": float(out_arr.mean())}
    return {"method": method, "before": before, "after": after}


def handle_monai_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "medical_image_metadata":
            r = _metadata_sync(args["path"])
            return (
                f"shape={r['shape']} dtype={r['dtype']} "
                f"min={r['min']:.3f} max={r['max']:.3f} mean={r['mean']:.3f}"
            )
        if name == "medical_image_normalize":
            r = _normalize_sync(args["path"], method=args.get("method", "scale_intensity"))
            return (
                f"method={r['method']} before={r['before']} after={r['after']}"
            )
        return f"[unknown monai tool: {name}]"
    except Exception as exc:  # noqa: BLE001
        return f"[{name} error: {exc}]"
