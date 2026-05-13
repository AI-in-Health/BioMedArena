"""Official-compatible BixBench capsule and scoring helpers.

The FutureHouse BixBench dataset publishes one JSONL row per question and a
matching ``CapsuleFolder-{uuid}.zip`` data capsule for each analysis capsule.
This module keeps BioMedArena's default lightweight MCQ adaptation available,
while exposing the pieces needed to reproduce the full agent-style setting:

* explicit capsule download and safe extraction;
* Docker-backed Python/R/bash sandbox execution with the capsule mounted;
* open-answer verification using BixBench ``eval_mode`` hints;
* MCQ scoring with optional opt-out handling and majority voting.

The official public harness is not vendored here.  The interfaces below are
intentionally small and auditable so an external official runner can call into
the same capsule cache and scoring surface when it is available.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from harness.eval.scoring import (
    extract_answer_from_response,
    extract_numeric_answer,
    normalize_text,
    score_exact_match,
    score_multiple_choice,
)


BIXBENCH_REPO_ID = "futurehouse/BixBench"
BIXBENCH_DEFAULT_REVISION = "main"
BIXBENCH_DEFAULT_CACHE = Path("data/cache/bixbench")
BIXBENCH_DEFAULT_DOCKER_IMAGE = "biomedarena/bixbench-sandbox:latest"


@dataclass(frozen=True)
class CapsuleInfo:
    """Resolved BixBench capsule file locations."""

    uuid: str
    filename: str
    archive_path: Path
    extract_dir: Path


@dataclass(frozen=True)
class SandboxResult:
    """Result of running one command in the BixBench sandbox."""

    command: str
    returncode: int
    stdout: str
    stderr: str
    backend: str
    capsule_dir: str
    work_dir: str


@dataclass
class BixBenchMCQConfig:
    """MCQ-regime options used by the official-compatible scorer."""

    majority_vote_n: int = 1
    allow_opt_out: bool = False
    opt_out_tokens: tuple[str, ...] = (
        "opt out",
        "abstain",
        "cannot answer",
        "insufficient information",
        "i don't know",
        "i do not know",
    )


@dataclass
class BixBenchSandboxConfig:
    """Sandbox configuration for capsule-backed analysis."""

    backend: str = "docker"
    image: str = BIXBENCH_DEFAULT_DOCKER_IMAGE
    docker_cmd: str | None = None
    timeout_s: int = 900
    network: str = "none"
    allow_local: bool = False
    extra_docker_args: list[str] = field(default_factory=list)


class BixBenchCapsuleManager:
    """Download and safely extract BixBench data capsules.

    Parameters
    ----------
    cache_dir:
        Local root for archives and extracted capsule folders.
    revision:
        HuggingFace revision/tag.  Use ``main`` or ``v1.5`` for the current
        205-row release; ``v1.0`` can be selected for legacy comparisons.
    token:
        Optional HuggingFace token.  The public dataset normally does not need
        one, but passing it is harmless and helps in mirrored/private setups.
    """

    def __init__(
        self,
        cache_dir: str | Path = BIXBENCH_DEFAULT_CACHE,
        *,
        revision: str | None = None,
        token: str | None = None,
        repo_id: str = BIXBENCH_REPO_ID,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.revision = revision or BIXBENCH_DEFAULT_REVISION
        self.token = token or _hf_token()
        self.repo_id = repo_id
        self.archive_dir = self.cache_dir / "archives" / self.revision
        self.extract_root = self.cache_dir / "capsules" / self.revision
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.extract_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def filename_for(capsule: str) -> str:
        """Return canonical ``CapsuleFolder-*.zip`` filename."""
        value = str(capsule or "").strip()
        if not value:
            raise ValueError("empty BixBench capsule id")
        name = Path(value).name
        if name.endswith(".zip") and name.startswith("CapsuleFolder-"):
            return name
        if value.startswith("CapsuleFolder-"):
            return f"{value}.zip" if not value.endswith(".zip") else value
        return f"CapsuleFolder-{value}.zip"

    @staticmethod
    def uuid_for(capsule: str) -> str:
        filename = BixBenchCapsuleManager.filename_for(capsule)
        return filename.removeprefix("CapsuleFolder-").removesuffix(".zip")

    def capsule_info(self, capsule: str, *, archive_path: str | Path | None = None) -> CapsuleInfo:
        filename = self.filename_for(capsule)
        uuid = self.uuid_for(filename)
        archive = Path(archive_path) if archive_path else self.archive_dir / filename
        return CapsuleInfo(
            uuid=uuid,
            filename=filename,
            archive_path=archive,
            extract_dir=self.extract_root / f"CapsuleFolder-{uuid}",
        )

    def download_capsule(self, capsule: str, *, force: bool = False) -> CapsuleInfo:
        """Download one capsule zip with ``huggingface_hub``.

        Large downloads are explicit by design.  Callers should expose a
        command such as ``biomedarena prepare-bixbench`` instead of doing this
        during normal benchmark loading.
        """
        info = self.capsule_info(capsule)
        if info.archive_path.exists() and not force:
            return info
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "BixBench capsule download requires huggingface_hub. "
                "Install with `pip install -e '.[eval]'`."
            ) from exc

        downloaded = hf_hub_download(
            repo_id=self.repo_id,
            filename=info.filename,
            repo_type="dataset",
            revision=self.revision,
            token=self.token,
            local_dir=str(self.archive_dir),
            local_dir_use_symlinks=False,
        )
        downloaded_path = Path(downloaded)
        if downloaded_path != info.archive_path and downloaded_path.exists():
            info.archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded_path, info.archive_path)
        return info

    def extract_capsule(self, capsule: str, *, force: bool = False) -> CapsuleInfo:
        """Ensure one capsule is downloaded and safely extracted."""
        info = self.download_capsule(capsule)
        if info.extract_dir.exists() and any(info.extract_dir.iterdir()) and not force:
            return info
        if force and info.extract_dir.exists():
            shutil.rmtree(info.extract_dir)
        info.extract_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(info.archive_path, info.extract_dir)
        return info

    def ensure_capsule(self, capsule: str, *, extract: bool = True) -> CapsuleInfo:
        """Download, and optionally extract, one capsule."""
        return self.extract_capsule(capsule) if extract else self.download_capsule(capsule)

    def ensure_for_tasks(
        self,
        tasks: Iterable[dict[str, Any]],
        *,
        extract: bool = True,
        limit: int | None = None,
    ) -> dict[str, CapsuleInfo]:
        """Download/extract unique capsules referenced by tasks."""
        resolved: dict[str, CapsuleInfo] = {}
        for task in tasks:
            capsule = _capsule_from_task(task)
            if not capsule:
                continue
            uuid = self.uuid_for(capsule)
            if uuid in resolved:
                continue
            resolved[uuid] = self.ensure_capsule(capsule, extract=extract)
            if limit is not None and len(resolved) >= limit:
                break
        return resolved


class BixBenchSandbox:
    """Run Python/R/bash commands with a BixBench capsule mounted.

    Docker is the default and recommended backend.  The local backend is
    intentionally opt-in because BixBench capsules are untrusted benchmark
    inputs and commands may execute arbitrary analysis code.
    """

    def __init__(self, config: BixBenchSandboxConfig | None = None) -> None:
        self.config = config or BixBenchSandboxConfig()

    def run(self, capsule_dir: str | Path, command: str) -> SandboxResult:
        capsule_path = Path(capsule_dir).resolve()
        if not capsule_path.exists():
            raise FileNotFoundError(f"BixBench capsule dir not found: {capsule_path}")
        if self.config.backend == "docker":
            return self._run_docker(capsule_path, command)
        if self.config.backend == "local":
            if not self.config.allow_local:
                raise RuntimeError("local BixBench sandbox backend requires allow_local=True")
            return self._run_local(capsule_path, command)
        raise ValueError(f"unsupported BixBench sandbox backend: {self.config.backend}")

    def _run_docker(self, capsule_path: Path, command: str) -> SandboxResult:
        docker_cmd = self.config.docker_cmd or resolve_docker_command()
        if docker_cmd is None:
            raise RuntimeError(
                "docker command not found. Install Docker Desktop or add docker to PATH."
            )
        with tempfile.TemporaryDirectory(prefix="bixbench-work-") as tmp:
            work_dir = Path(tmp).resolve()
            cmd = [
                docker_cmd, "run", "--rm",
                "--network", self.config.network,
                "-v", f"{capsule_path}:/capsule:ro",
                "-v", f"{work_dir}:/work",
                "-w", "/work",
                *self.config.extra_docker_args,
                self.config.image,
                "bash", "-lc", command,
            ]
            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
            )
            return SandboxResult(
                command=command,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                backend="docker",
                capsule_dir=str(capsule_path),
                work_dir=str(work_dir),
            )

    def _run_local(self, capsule_path: Path, command: str) -> SandboxResult:
        env = dict(os.environ)
        env["BIXBENCH_CAPSULE_DIR"] = str(capsule_path)
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(capsule_path),
            env=env,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_s,
            check=False,
        )
        return SandboxResult(
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            backend="local",
            capsule_dir=str(capsule_path),
            work_dir=str(capsule_path),
            )


def resolve_docker_command() -> str | None:
    """Return a Docker executable usable for BixBench sandbox runs.

    Docker Desktop on macOS can be installed without creating shell symlinks.
    In that case `docker` is absent from PATH, but the bundled CLI exists
    inside the app bundle.  Falling back to that path keeps local runs working
    without asking users to edit their shell profile.
    """

    found = shutil.which("docker")
    if found:
        return found
    macos_app_cli = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    if macos_app_cli.exists() and os.access(macos_app_cli, os.X_OK):
        return str(macos_app_cli)
    return None


def score_bixbench_mcq(
    predictions: str | list[str],
    expected_letter: str,
    *,
    config: BixBenchMCQConfig | None = None,
) -> dict[str, Any]:
    """Score BixBench MCQ adaptation with opt-out and majority voting."""
    cfg = config or BixBenchMCQConfig()
    if isinstance(predictions, str):
        raw_predictions = [predictions]
    else:
        raw_predictions = [str(item) for item in predictions]
    raw_predictions = raw_predictions[: max(1, cfg.majority_vote_n)]
    votes: list[str] = []
    opted_out = 0
    for raw in raw_predictions:
        text = str(raw or "")
        if cfg.allow_opt_out and _is_opt_out(text, cfg.opt_out_tokens):
            opted_out += 1
            continue
        votes.append(extract_answer_from_response(text, "multipleChoice"))
    if not votes:
        return {
            "correct": False,
            "method": "bixbench_mcq_majority",
            "selected": "",
            "votes": votes,
            "opted_out": opted_out,
        }
    counts = Counter(votes)
    selected = counts.most_common(1)[0][0]
    return {
        "correct": score_multiple_choice(selected, expected_letter),
        "method": "bixbench_mcq_majority",
        "selected": selected,
        "votes": votes,
        "opted_out": opted_out,
    }


async def score_bixbench_open_answer(
    task: dict[str, Any],
    prediction: str,
    *,
    judge_fn: Callable[[str, str, str], Awaitable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score one BixBench open-answer prediction.

    ``eval_mode`` values currently observed in the public dataset include
    ``str_verifier``, ``range_verifier``, and ``llm_verifier``.  String and
    range verification are deterministic; LLM verification delegates to the
    configured judge.
    """
    expected = str(task.get("answer", ""))
    context = task.get("context") or task.get("metadata") or {}
    eval_mode = str(context.get("eval_mode") or "").strip().lower()
    if "range" in eval_mode:
        result = _score_range_verifier(prediction, expected)
        result["method"] = "bixbench_range_verifier"
        return result
    if "str" in eval_mode:
        extracted = extract_answer_from_response(prediction, "exactMatch")
        correct = score_exact_match(prediction, expected) or score_exact_match(extracted, expected)
        return {
            "correct": correct,
            "method": "bixbench_str_verifier",
            "details": {"eval_mode": eval_mode, "extracted": extracted},
        }
    if judge_fn is None:
        # Deterministic fallback when judge is disabled/unavailable.
        correct = score_exact_match(prediction, expected)
        return {
            "correct": correct,
            "method": "bixbench_llm_verifier_fallback_exact",
            "details": {"eval_mode": eval_mode, "judge_invoked": False},
        }
    judge_result = await judge_fn(str(task.get("question", "")), expected, prediction)
    return {
        "correct": bool(judge_result.get("correct")),
        "method": "bixbench_llm_verifier",
        "details": {
            "eval_mode": eval_mode,
            "judge_invoked": True,
            "judge_raw": str(judge_result.get("reasoning", ""))[:500],
            "judge_error": bool(judge_result.get("error")),
        },
    }


def attach_capsule_paths(
    tasks: list[dict[str, Any]],
    *,
    cache_dir: str | Path = BIXBENCH_DEFAULT_CACHE,
    revision: str | None = None,
    require: bool = False,
) -> list[dict[str, Any]]:
    """Attach already-extracted capsule paths to task metadata.

    This function never downloads.  It is useful after
    ``biomedarena prepare-bixbench`` has materialized the cache.
    """
    manager = BixBenchCapsuleManager(cache_dir=cache_dir, revision=revision)
    for task in tasks:
        capsule = _capsule_from_task(task)
        if not capsule:
            continue
        info = manager.capsule_info(capsule)
        if info.extract_dir.exists():
            meta = task.setdefault("metadata", {})
            ctx = task.setdefault("context", {})
            meta["capsule_path"] = str(info.extract_dir)
            ctx["capsule_path"] = str(info.extract_dir)
        elif require:
            raise FileNotFoundError(
                f"Missing extracted BixBench capsule {info.extract_dir}. "
                "Run `biomedarena prepare-bixbench --extract` first."
            )
    return tasks


def _capsule_from_task(task: dict[str, Any]) -> str:
    meta = task.get("metadata") or {}
    ctx = task.get("context") or {}
    return str(
        meta.get("data_folder")
        or ctx.get("data_folder")
        or meta.get("capsule_uuid")
        or ctx.get("capsule_uuid")
        or ""
    )


def _safe_extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest = dest_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise RuntimeError(f"unsafe path in BixBench capsule zip: {member.filename}")
        zf.extractall(dest)


def _score_range_verifier(prediction: str, expected: str) -> dict[str, Any]:
    pred_num, reason = extract_numeric_answer(prediction)
    bounds = _parse_expected_range(expected)
    if pred_num is None:
        return {
            "correct": False,
            "details": {"reason": reason, "expected_range": bounds},
        }
    if bounds is not None:
        lo, hi = bounds
        return {
            "correct": lo <= pred_num <= hi,
            "details": {"predicted_number": pred_num, "expected_range": [lo, hi]},
        }
    exp_num, _ = extract_numeric_answer(expected)
    if exp_num is None:
        return {
            "correct": score_exact_match(prediction, expected),
            "details": {"predicted_number": pred_num, "expected": expected},
        }
    if exp_num == 0:
        correct = abs(pred_num) < 1e-6
    else:
        correct = abs(pred_num - exp_num) / abs(exp_num) <= 0.05
    return {
        "correct": correct,
        "details": {"predicted_number": pred_num, "expected_number": exp_num},
    }


def _parse_expected_range(value: str) -> tuple[float, float] | None:
    text = str(value or "")
    nums = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]
    if len(nums) >= 2 and ("(" in text or "[" in text or "," in text or "-" in text):
        lo, hi = nums[0], nums[1]
        return (min(lo, hi), max(lo, hi))
    return None


def _is_opt_out(text: str, tokens: tuple[str, ...]) -> bool:
    norm = normalize_text(text)
    return any(token in norm for token in tokens)


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Small utility for external official-runner interop."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
