"""Dataset access and flow-input construction for Rule2DRC evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from egar_drc.flow_config import DEFAULT_EGAR_DRC_CONFIG


def resolve_repository_path(repository_path: str | None = None) -> Path:
    """Resolve the Rule2DRC repository root.

    Args:
        repository_path: Optional explicit repository path override.

    Returns:
        Resolved repository root path.
    """
    raw_path = str(
        repository_path or os.environ.get("RULE2DRC_REPOSITORY_PATH", "")
    ).strip()
    if not raw_path:
        raise ValueError("Set RULE2DRC_REPOSITORY_PATH or pass repository_path explicitly.")
    return Path(raw_path).expanduser().resolve()


def find_problem_dir(problems_root: Path, problem: str) -> Path:
    """Resolve a single problem directory by exact id or numeric prefix.

    Args:
        problems_root: Directory containing problem subdirectories.
        problem: Exact problem id (``0001_physical``) or numeric prefix (``0001``).

    Returns:
        The resolved problem directory.

    Raises:
        FileNotFoundError: When no problem directory matches the request.
        ValueError: When a numeric prefix matches more than one directory.
    """
    requested = problem.strip()
    if not requested:
        raise ValueError("problem must be a non-empty string.")

    exact_path = problems_root / requested
    if exact_path.is_dir():
        return exact_path

    prefix_matches = sorted(path for path in problems_root.glob(f"{requested}_*") if path.is_dir())
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ValueError(f"Ambiguous problem prefix '{requested}': {[path.name for path in prefix_matches]}")

    raise FileNotFoundError(f"Problem '{requested}' not found under {problems_root}.")


def iter_problem_dirs(problems_root: Path, problems: list[str] | None = None) -> list[Path]:
    """List problem directories, optionally filtered by requested ids.

    Args:
        problems_root: Directory containing problem subdirectories.
        problems: Optional list of exact ids or numeric prefixes.

    Returns:
        Unique directories in selector order, or sorted directories without selectors.
    """
    if not problems_root.is_dir():
        raise FileNotFoundError(f"Problems root not found: {problems_root}.")

    if problems:
        resolved = (find_problem_dir(problems_root, problem).resolve() for problem in problems)
        return list(dict.fromkeys(resolved))

    return sorted(path for path in problems_root.iterdir() if path.is_dir() and (path / "spec.yaml").is_file())


def load_spec(problem_dir: Path) -> dict[str, Any]:
    """Load and parse a problem ``spec.yaml`` file.

    Args:
        problem_dir: Problem directory containing ``spec.yaml``.

    Returns:
        Parsed spec mapping.
    """
    spec = yaml.safe_load((problem_dir / "spec.yaml").read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid spec.yaml in {problem_dir}.")

    return spec


def _format_layer_context(layers: dict[str, Any]) -> str:
    """Render the layer map as a readable prompt context string.

    Args:
        layers: Mapping of layer name to ``[layer_number, datatype]``.

    Returns:
        Human-readable layer context text.
    """
    lines = []
    for name, value in layers.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lines.append(f"- {name}: layer {value[0]}, datatype {value[1]}")
        else:
            lines.append(f"- {name}: {value}")

    return "\n".join(lines)


def build_flow_input(
    problem_dir: Path,
    repository_root: Path,
    max_code_generation_compile_iterations: int = (
        DEFAULT_EGAR_DRC_CONFIG.max_code_generation_compile_iterations
    ),
    max_code_generation_retry_iterations: int = DEFAULT_EGAR_DRC_CONFIG.max_code_generation_retry_iterations,
    max_code_correction_compile_iterations: int = (
        DEFAULT_EGAR_DRC_CONFIG.max_code_correction_compile_iterations
    ),
    max_code_correction_iterations: int = DEFAULT_EGAR_DRC_CONFIG.max_code_correction_iterations,
    max_testcase_generation_prepare_iterations: int = (
        DEFAULT_EGAR_DRC_CONFIG.max_testcase_generation_prepare_iterations
    ),
    max_testcase_correction_prepare_iterations: int = (
        DEFAULT_EGAR_DRC_CONFIG.max_testcase_correction_prepare_iterations
    ),
    max_testcase_correction_iterations: int = DEFAULT_EGAR_DRC_CONFIG.max_testcase_correction_iterations,
    max_critic_review_iterations: int = DEFAULT_EGAR_DRC_CONFIG.max_critic_review_iterations,
    max_testcases: int = DEFAULT_EGAR_DRC_CONFIG.max_testcases,
    klayout_bin: str = "klayout",
    testcase_suite_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Build a Rule2DRC flow input payload from a problem spec.

    Prompt-facing spec text omits gold-deck and benchmark-data paths. Repository
    and problem paths remain runtime wiring, not gold correctness feedback.

    Args:
        problem_dir: Problem directory containing ``spec.yaml``.
        repository_root: Repository root passed through to the flow state.
        max_code_generation_compile_iterations: Compilation attempts allowed
            before the first successful compilation.
        max_code_generation_retry_iterations: Fresh code-generation cycles allowed
            after generation compile attempts are exhausted.
        max_code_correction_compile_iterations: Compilation attempts allowed
            for each correction cycle after compilation succeeds.
        max_code_correction_iterations: Critic-requested code-correction cycle budget.
        max_testcase_generation_prepare_iterations: Testcase generator preparation attempts.
        max_testcase_correction_prepare_iterations: Testcase correction preparation attempts.
        max_testcase_correction_iterations: Critic-requested testcase-correction cycle budget.
        max_critic_review_iterations: Critic testcase/review budget.
        max_testcases: Maximum initial GDS testcases generated by the agent.
        klayout_bin: KLayout executable used by the in-flow compilation check.
        testcase_suite_cache_root: Optional persistent per-flow GDS suite cache.

    Returns:
        Input payload for the EGAR-DRC graph.
    """
    spec = load_spec(problem_dir)
    layers = spec.get("layers", {}) or {}
    rules = spec.get("rules", []) or []
    categories = list(spec.get("categories", []) or [])
    thresholds = spec.get("thresholds", {}) or {}
    prompt_spec = dict(spec)
    prompt_spec.pop("deck", None)
    prompt_spec.pop("data_dir", None)

    return {
        "problem_id": str(spec.get("id", problem_dir.name)),
        "repository_path": str(repository_root),
        "problem_dir": str(problem_dir),
        "klayout_bin": str(klayout_bin),
        "testcase_suite_cache_root": str(testcase_suite_cache_root) if testcase_suite_cache_root else "",
        "title": str(spec.get("title", "")),
        "rule_description": str(spec.get("nl_description", "")),
        "rules": rules,
        "categories": categories,
        "layers": layers,
        "thresholds": thresholds,
        "spec_yaml": yaml.safe_dump(prompt_spec, sort_keys=False),
        "layer_context": _format_layer_context(layers),
        "max_code_generation_compile_iterations": max(int(max_code_generation_compile_iterations), 1),
        "max_code_generation_retry_iterations": max(int(max_code_generation_retry_iterations), 1),
        "max_code_correction_compile_iterations": max(int(max_code_correction_compile_iterations), 1),
        "max_code_correction_iterations": max(int(max_code_correction_iterations), 1),
        "max_testcase_generation_prepare_iterations": max(int(max_testcase_generation_prepare_iterations), 1),
        "max_testcase_correction_prepare_iterations": max(int(max_testcase_correction_prepare_iterations), 1),
        "max_testcase_correction_iterations": max(int(max_testcase_correction_iterations), 1),
        "max_critic_review_iterations": max(int(max_critic_review_iterations), 1),
        "max_testcases": max(int(max_testcases), 1),
        "loop_metrics": {},
        "llm_request_metrics": [],
    }
