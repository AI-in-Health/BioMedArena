"""Hugging Face source verification helpers for benchmark loaders."""

from __future__ import annotations

from typing import Any


def verify_hf_dataset_online(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> dict[str, Any] | None:
    """Return dataset metadata only if the Hub is reachable now.

    ``datasets.load_dataset`` can silently fall back to a local cache when
    the network is unavailable. For gated/current-version-sensitive
    benchmarks, that is too weak: we need to know the local run can see
    the upstream repository at run time.
    """
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(repo_id, revision=revision, token=token)
    except Exception:
        return None
    return {
        "repo_id": repo_id,
        "sha": getattr(info, "sha", None),
        "revision": revision or "main",
    }
