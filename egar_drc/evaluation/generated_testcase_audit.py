"""Post-flow gold-reference audit for generated Rule2DRC testcases.

This module is evaluation-only. Its outputs must never be inserted into flow
state or agent prompts.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from egar_drc.evaluation.dataset import load_spec
from egar_drc.evaluation.generated_testcases import prepare_testcase_suite
from egar_drc.evaluation.klayout_scoring import evaluate_deck, read_labels_csv


def _inventory_by_name(inventory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate inventory filenames and SHA-256 digests and index its records."""
    records: dict[str, dict[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Testcase inventory records must be objects.")
        filename = item.get("filename")
        digest = item.get("sha256")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".gds")
            or filename in records
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Testcase inventory requires unique GDS basenames and SHA-256 digests.")
        records[filename] = item
    return records


def _hash_mismatches(generated_dir: Path, inventory: list[dict[str, Any]]) -> list[str]:
    """Return missing, unexpected, or modified GDS files against the inventory."""
    records = _inventory_by_name(inventory)
    mismatches = set(path.name for path in generated_dir.glob("*.gds")) - set(records)
    for filename, record in records.items():
        path = generated_dir / filename
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(generated_dir.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            mismatches.add(filename)
    return sorted(mismatches)


def _classify_score(
    inventory: list[dict[str, Any]],
    score: dict[str, Any],
    critic_rejected_filenames: list[str],
) -> dict[str, Any]:
    """Classify verified gold outcomes and mark missing evidence unverifiable."""
    filenames = set(_inventory_by_name(inventory))
    precheck = score.get("precheck") or {}
    failed_names: set[str] = set()
    for key in ("klayout_failed_cases", "malformed_rdb_cases", "labels_missing_rows"):
        for value in precheck.get(key, []):
            failed_names.add(Path(str(value.get("path", "") if isinstance(value, dict) else value)).name)
    invalid_labels = bool(precheck.get("labels_missing") or precheck.get("labels_extra"))
    correct: set[str] = set()
    incorrect: set[str] = set()
    outcomes = score.get("case_results") or {}
    for relative_path, outcome in outcomes.items():
        filename = Path(relative_path).name
        if filename not in filenames or filename in failed_names or invalid_labels:
            continue
        expected = outcome.get("expected") if isinstance(outcome, dict) else None
        observed = outcome.get("observed") if isinstance(outcome, dict) else None
        if (
            not isinstance(expected, dict)
            or not expected
            or not isinstance(observed, dict)
            or expected.keys() != observed.keys()
            or any(value not in (0, 1) for value in (*expected.values(), *observed.values()))
        ):
            continue
        if expected == observed:
            correct.add(filename)
        else:
            incorrect.add(filename)
    unverifiable = filenames - correct - incorrect
    rejected = {Path(value.strip()).name for value in critic_rejected_filenames if value.strip()}
    true_rejections = rejected & incorrect
    false_rejections = rejected & correct
    verified_rejections = len(true_rejections) + len(false_rejections)
    verified_count = len(correct) + len(incorrect)
    return {
        "n_cases": len(filenames),
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "n_unverifiable": len(unverifiable),
        "accuracy": len(correct) / verified_count if verified_count else None,
        "correct_filenames": sorted(correct),
        "incorrect_filenames": sorted(incorrect),
        "unverifiable_filenames": sorted(unverifiable),
        "critic_rejection": {
            "rejected_filenames": sorted(rejected),
            "true_rejections": sorted(true_rejections),
            "false_rejections": sorted(false_rejections),
            "unverifiable_rejections": sorted(rejected - correct - incorrect),
            "missed_incorrect": sorted(incorrect - rejected),
            "precision": len(true_rejections) / verified_rejections if verified_rejections else None,
            "recall": len(true_rejections) / len(incorrect) if incorrect else None,
        },
    }


def _resolve_gold_deck(problem_dir: Path, repository_root: Path) -> Path:
    """Resolve a problem's gold deck while enforcing repository containment.

    Args:
        problem_dir: Problem directory containing the specification.
        repository_root: Rule2DRC repository root.

    Returns:
        Resolved gold deck path.
    """
    spec = load_spec(problem_dir)
    deck_value = str(spec.get("deck", "")).strip()
    if not deck_value:
        raise ValueError(f"Problem has no gold deck path: {problem_dir}")

    deck_path = (repository_root / deck_value).resolve()
    deck_path.relative_to(repository_root.resolve())
    if not deck_path.is_file():
        raise FileNotFoundError(deck_path)

    return deck_path


def audit_generated_testcases(
    *,
    problem_dir: Path,
    repository_root: Path,
    generator_scripts: list[str],
    categories: list[str],
    max_cases: int | None,
    expected_inventory: list[dict[str, Any]],
    critic_rejected_filenames: list[str],
    output_dir: Path,
    klayout_bin: str,
) -> dict[str, Any]:
    """Regenerate testcases and compare their labels with the gold deck.

    Gold artifacts are read only after the agentic flow has terminated. The
    result is written to evaluation artifacts and is never returned to an agent.

    Args:
        problem_dir: Source problem directory.
        repository_root: Rule2DRC repository root.
        generator_scripts: Initial and supplemental generator source strings.
        categories: Exact output categories.
        max_cases: Optional case limit, used only for an initial generator.
        expected_inventory: Hash inventory recorded during the final flow test.
        critic_rejected_filenames: Cases rejected by the critic during flow.
        output_dir: Persistent audit artifact directory.
        klayout_bin: Resolved KLayout executable.

    Returns:
        Testcase correctness and reproducibility metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rule2drc_gold_audit_") as temp_dir:

        suite_root = Path(temp_dir) / "suite"
        suite_problem_dir = prepare_testcase_suite(
            source_problem_dir=problem_dir,
            suite_repository_root=suite_root,
            generator_scripts=generator_scripts,
            categories=categories,
            max_cases=max_cases,
        )
        spec = load_spec(problem_dir)
        generated_dir = suite_root / Path(str(spec.get("data_dir", ""))) / "generated"
        inventory = json.loads((generated_dir / "inventory.json").read_text(encoding="utf-8"))
        if not isinstance(inventory, list):
            raise ValueError("Generated testcase inventory must be a list.")
        hash_mismatches = _hash_mismatches(generated_dir, inventory)

        gold_deck = _resolve_gold_deck(problem_dir, repository_root)
        score = evaluate_deck(
            suite_problem_dir,
            gold_deck,
            suite_root,
            output_dir,
            klayout_bin,
        )

    expected_by_name = _inventory_by_name(expected_inventory)
    actual_by_name = _inventory_by_name(inventory)
    reproducibility_available = bool(expected_by_name)
    reproducible = (expected_by_name == actual_by_name and not hash_mismatches) if reproducibility_available else None
    payload: dict[str, Any] = {
        "evaluation_only": True,
        "gold_exposed_to_agents": False,
        "reproducibility_available": reproducibility_available,
        "reproducible": reproducible,
        **_classify_score(inventory, score, critic_rejected_filenames),
        "expected_inventory": expected_inventory,
        "regenerated_inventory": inventory,
        "score": score,
    }
    (output_dir / "generated_testcase_audit.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    return payload


def audit_cached_testcase_suite(
    *,
    problem_dir: Path,
    repository_root: Path,
    suite_repository_root: Path,
    expected_inventory: list[dict[str, Any]],
    critic_rejected_filenames: list[str],
    output_dir: Path,
    klayout_bin: str,
) -> dict[str, Any]:
    """Audit a persisted generated testcase suite without rerunning its generator.

    Hash and label checks establish integrity of cached evidence, not whether
    rerunning the generator would reproduce it. Gold results remain evaluation
    artifacts and must not feed flow state or agent prompts.

    Args:
        problem_dir: Source problem directory used to resolve the gold deck.
        repository_root: Original Rule2DRC repository root.
        suite_repository_root: Cached isolated suite repository root.
        expected_inventory: Optional inventory captured during the original flow.
        critic_rejected_filenames: Cases rejected by the critic during the flow.
        output_dir: Persistent audit artifact directory.
        klayout_bin: Resolved KLayout executable.

    Returns:
        Testcase correctness and provenance metrics for the cached suite.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(problem_dir)
    suite_problem_dir = suite_repository_root / "problems" / problem_dir.name
    generated_dir = suite_repository_root / Path(str(spec.get("data_dir", ""))) / "generated"
    inventory_path = generated_dir / "inventory.json"
    if not suite_problem_dir.is_dir() or not inventory_path.is_file():
        raise FileNotFoundError(f"Invalid cached testcase suite: {suite_repository_root}")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, list):
        raise ValueError("Cached testcase inventory must be a list.")

    gold_deck = _resolve_gold_deck(problem_dir, repository_root)
    expected_by_name = _inventory_by_name(expected_inventory)
    actual_by_name = _inventory_by_name(inventory)
    hash_mismatches = _hash_mismatches(generated_dir, inventory)
    labels_path = generated_dir.parent / "labels.csv"
    labels_match: bool | None = None
    if all(isinstance(item.get("expected"), dict) for item in inventory):
        try:
            labels = read_labels_csv(labels_path, list(spec.get("categories", [])))
            expected_labels = {
                (generated_dir / item["filename"]).relative_to(suite_repository_root).as_posix(): item["expected"]
                for item in inventory
            }
            labels_match = labels == expected_labels
        except (OSError, ValueError):
            labels_match = False
    score = (
        # Untrusted cached bytes or labels cannot support a gold classification.
        {"precheck": {"cached_hash_mismatches": hash_mismatches, "cached_labels_match": labels_match}}
        if hash_mismatches or labels_match is False else
        evaluate_deck(suite_problem_dir, gold_deck, suite_repository_root, output_dir, klayout_bin)
    )
    payload: dict[str, Any] = {
        "evaluation_only": True,
        "gold_exposed_to_agents": False,
        "suite_source": "persisted_cache",
        "suite_repository_root": str(suite_repository_root),
        "reproducibility_available": False,
        "reproducible": None,
        "cached_hash_integrity_available": True,
        "cached_hash_integrity": not hash_mismatches,
        "cached_hash_mismatches": hash_mismatches,
        "cached_labels_match_inventory": labels_match,
        "inventory_matches_expected": expected_by_name == actual_by_name if expected_by_name else None,
        **_classify_score(inventory, score, critic_rejected_filenames),
        "expected_inventory": expected_inventory,
        "cached_inventory": inventory,
        "score": score,
    }
    (output_dir / "generated_testcase_audit.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return payload
