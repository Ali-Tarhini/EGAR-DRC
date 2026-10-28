"""Build filtered experiment provenance."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from egar_drc.flow_config import DEFAULT_EGAR_DRC_CONFIG
from egar_drc.models.config import ModelConfig

# Restrict published decoding settings to known keys.
_PUBLIC_DECODING = frozenset({
    "top_p", "top_k", "seed", "frequency_penalty", "presence_penalty",
    "repetition_penalty", "min_p", "max_completion_tokens", "n",
})


def _sha256(path: Path) -> str | None:
    """Hash a local file without failing runs for unavailable input files."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _package_version(name: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _klayout_version(executable: str) -> str | None:
    """Probe the local executable with a bounded wait and no raw output retention."""
    try:
        result = subprocess.run(
            [executable, "-v"], capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"\bKLayout\s+(\d+\.\d+(?:\.\d+)?)\b", result.stdout)
    return match.group(1) if result.returncode == 0 and match else None


def _public_model_config(config: ModelConfig) -> dict[str, Any]:
    """Filter endpoint and provider settings for publication."""
    try:
        endpoint = urlsplit(config.base_url)
        origin = f"{endpoint.scheme}://{endpoint.hostname or ''}"
        if endpoint.port is not None:
            origin += f":{endpoint.port}"
    except ValueError:
        origin = "[REDACTED]"
    secret = config.api_key
    model = config.model.replace(secret, "[REDACTED]") if secret and secret != "not-required" else config.model
    return {
        "model": model,
        "base_url": origin,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "retry_attempts": config.retry_attempts,
        "max_concurrency": config.max_concurrency,
        "additional_parameters": {
            key: value if type(value) in (int, float, bool) else "[REDACTED]"
            for key, value in config.additional_parameters.items()
            if key in _PUBLIC_DECODING
        },
        "redacted_parameter_count": sum(key not in _PUBLIC_DECODING for key in config.additional_parameters),
    }


def build_run_metadata(
    config: ModelConfig,
    problems: list[Path],
    klayout_bin: str,
    flow: str,
    jobs: int,
    max_testcases: int,
) -> dict[str, Any]:
    """Snapshot this invocation's selected inputs, settings, and source identity.

    The caller rewrites run_metadata.json on each invocation, including resume;
    this is not a merged history of configurations for earlier result records.
    Source hashes include package Python, YAML, and text files, so documentation
    edits also change source identity. Unavailable files or version probes yield
    null values rather than blocking the experiment.

    Args:
        config: Validated provider configuration, never serialized directly.
        problems: Unique selected problem directories in scheduling order.
        klayout_bin: Resolved local executable to probe for its version.
        flow: Selected flow name.
        jobs: Maximum simultaneous problem workers.
        max_testcases: Requested testcase budget, clamped like flow input.

    Returns:
        Filtered provenance with selected ids, content hashes, and versions.
    """
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(package_root).as_posix(): _sha256(path)
        for path in sorted(package_root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".txt"}
    }
    flow_config = asdict(DEFAULT_EGAR_DRC_CONFIG)
    flow_config["max_testcases"] = max(int(max_testcases), 1)
    return {
        "schema_version": 1,
        "package": {"name": "egar-drc", "version": _package_version("egar-drc")},
        "source": {"package": "egar_drc", "sha256": sources},
        "descriptors": {
            name: digest
            for name, digest in sources.items()
            if Path(name).match("agents/*/*_agent.yaml")
        },
        "problems": [{"id": problem.name, "spec_sha256": _sha256(problem / "spec.yaml")} for problem in problems],
        "model_config": _public_model_config(config),
        "execution": {"flow": flow, "jobs": jobs},
        "flow_config": flow_config,
        "versions": {
            "python": platform.python_version(),
            "klayout": _klayout_version(klayout_bin),
            "packages": {
                name: _package_version(name)
                for name in ("langchain", "langchain-openai", "langgraph", "pydantic", "PyYAML")
            },
        },
    }