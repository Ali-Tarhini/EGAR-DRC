"""Materialize LLM-generated Rule2DRC GDS testcase suites."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def prepare_cached_testcase_suite(
    *,
    source_problem_dir: Path,
    cache_root: Path,
    generator_scripts: list[str],
    categories: list[str],
    max_cases: int | None,
    excluded_filenames: list[str] | None = None,
    timeout_seconds: int = 120,
) -> Path:
    """Create or reuse an immutable generated-testcase suite.

    The key covers the resolved problem path, ordered scripts and categories,
    case limit, and normalized exclusions. Candidate-deck revisions can reuse
    the suite without rerunning generators. Spec contents, evaluator code, and
    toolchain versions are not keyed; this cache is not an integrity check or a
    guarantee of reproducibility across environments.

    Args:
        source_problem_dir: Problem directory used only to read ``spec.yaml``.
        cache_root: Flow-local persistent root for cached suites.
        generator_scripts: Initial and supplemental generator source strings.
        categories: Exact label columns required by the problem spec.
        max_cases: Optional maximum number of GDS files from the first generator.
        excluded_filenames: Generated cases to omit after validation.
        timeout_seconds: Timeout for each generated script.

    Returns:
        The copied problem directory in the cached isolated suite.
    """
    cache_inputs = {
        "problem_dir": str(source_problem_dir.resolve()),
        "generator_scripts": generator_scripts,
        "categories": categories,
        "max_cases": int(max_cases) if max_cases is not None else None,
        "excluded_filenames": sorted({str(value).strip() for value in excluded_filenames or [] if str(value).strip()}),
    }
    cache_key = hashlib.sha256(json.dumps(cache_inputs, sort_keys=True).encode("utf-8")).hexdigest()
    suite_repository_root = cache_root / cache_key
    suite_problem_dir = suite_repository_root / "problems" / source_problem_dir.name
    generated_dir = generated_testcase_directory(
        suite_repository_root=suite_repository_root,
        source_problem_dir=source_problem_dir,
    )
    if (
        (suite_problem_dir / "spec.yaml").is_file()
        and (generated_dir / "inventory.json").is_file()
        and (generated_dir.parent / "labels.csv").is_file()
    ):
        return suite_problem_dir

    shutil.rmtree(suite_repository_root, ignore_errors=True)

    return prepare_testcase_suite(
        source_problem_dir=source_problem_dir,
        suite_repository_root=suite_repository_root,
        generator_scripts=generator_scripts,
        categories=categories,
        max_cases=max_cases,
        excluded_filenames=excluded_filenames,
        timeout_seconds=timeout_seconds,
    )


def generated_testcase_directory(*, suite_repository_root: Path, source_problem_dir: Path) -> Path:
    """Resolve the generated-GDS directory for an isolated testcase suite.

    Args:
        suite_repository_root: Root of the generated isolated repository.
        source_problem_dir: Problem directory containing the original specification.

    Returns:
        Generated testcase directory within the isolated suite.

    Raises:
        ValueError: If the configured data directory is unsafe.
    """
    spec = yaml.safe_load((source_problem_dir / "spec.yaml").read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid spec.yaml in {source_problem_dir}.")

    data_dir = Path(str(spec.get("data_dir", "")))
    if data_dir.is_absolute() or ".." in data_dir.parts:
        raise ValueError("spec.yaml data_dir must be a repository-relative path.")

    return suite_repository_root / data_dir / "generated"


def prepare_testcase_suite(
    *,
    source_problem_dir: Path,
    suite_repository_root: Path,
    generator_scripts: list[str],
    categories: list[str],
    max_cases: int | None,
    excluded_filenames: list[str] | None = None,
    timeout_seconds: int = 120,
) -> Path:
    """Build an isolated generated-testcase suite from a problem specification.

    Args:
        source_problem_dir: Problem directory used only to read ``spec.yaml``.
        suite_repository_root: Temporary repository root for the isolated suite.
        generator_scripts: Initial and critic-provided Python ``pya`` scripts.
        categories: Exact label columns required by the problem spec.
        max_cases: Optional maximum number of GDS files from the first generator.
        excluded_filenames: Generated cases to omit after all scripts and append
            contracts have been validated.
        timeout_seconds: Timeout for each generated script.

    Returns:
        The copied problem directory in ``suite_repository_root``.

    Raises:
        ValueError: If generated files or manifest labels are invalid.
        RuntimeError: If a testcase generator cannot be executed.
    """
    spec_path = source_problem_dir / "spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid spec.yaml in {source_problem_dir}.")

    suite_problem_dir = suite_repository_root / "problems" / source_problem_dir.name
    suite_problem_dir.mkdir(parents=True, exist_ok=True)
    suite_spec = dict(spec)
    # Generated-suite execution needs the schema, not access to the gold deck.
    suite_spec.pop("deck", None)
    (suite_problem_dir / "spec.yaml").write_text(yaml.safe_dump(suite_spec, sort_keys=False), encoding="utf-8")

    data_dir = Path(str(spec.get("data_dir", "")))
    if data_dir.is_absolute() or ".." in data_dir.parts:
        raise ValueError("spec.yaml data_dir must be a repository-relative path.")

    gds_dir = suite_repository_root / data_dir
    generated_dir = gds_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index, script in enumerate(generator_scripts):

        prior_records = _normalize_manifest(generated_dir, categories) if index > 0 else []
        prior_hashes = {
            str(record["filename"]): _file_sha256(generated_dir / str(record["filename"])) for record in prior_records
        }
        _run_generator(
            script=script,
            script_path=suite_problem_dir / "generated_testcases" / f"generator_{index:02d}.py",
            out_dir=generated_dir,
            timeout_seconds=timeout_seconds,
        )
        current_records = _normalize_manifest(generated_dir, categories)
        if index == 0:
            if not current_records:
                raise ValueError("Initial testcase generator must produce at least one testcase.")

            if max_cases is not None and len(current_records) > max_cases:
                raise ValueError(
                    f"Initial testcase generator produced {len(current_records)} cases; limit is {max_cases}."
                )

            expected_filenames = [f"case_{case_index:04d}.gds" for case_index in range(len(current_records))]
            actual_filenames = [str(record["filename"]) for record in current_records]
            if actual_filenames != expected_filenames:
                raise ValueError("Initial testcase filenames must be contiguous starting at case_0000.gds.")

            _validate_manifest_file_set(generated_dir=generated_dir, records=current_records)
        else:
            _validate_supplemental_append(
                generated_dir=generated_dir,
                prior_records=prior_records,
                prior_hashes=prior_hashes,
                current_records=current_records,
            )

    records = _normalize_manifest(generated_dir, categories)
    # Preserve pre-exclusion provenance; labels below describe only retained cases.
    inventory = [{**record, "sha256": _file_sha256(generated_dir / str(record["filename"]))} for record in records]
    (generated_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if excluded_filenames:
        records = _exclude_testcases(
            generated_dir=generated_dir,
            records=records,
            excluded_filenames=excluded_filenames,
        )

    _merge_labels(
        labels_path=gds_dir / "labels.csv",
        repository_root=suite_repository_root,
        generated_dir=generated_dir,
        records=records,
        categories=categories,
    )

    return suite_problem_dir


def describe_generated_testcases(generated_dir: Path, categories: list[str]) -> list[dict[str, Any]]:
    """Return validated generated testcase metadata including content hashes.

    Args:
        generated_dir: Directory containing generated GDS files and manifest.
        categories: Exact expected-label keys.

    Returns:
        Normalized manifest rows augmented with ``sha256``.
    """
    return [
        {**record, "sha256": _file_sha256(generated_dir / str(record["filename"]))}
        for record in _normalize_manifest(generated_dir, categories)
    ]


def _exclude_testcases(
    *,
    generated_dir: Path,
    records: list[dict[str, Any]],
    excluded_filenames: list[str],
) -> list[dict[str, Any]]:
    """Remove critic-rejected cases after generator contracts are validated.

    Args:
        generated_dir: Generated artifact directory.
        records: Validated manifest records.
        excluded_filenames: Basenames selected by the critic for exclusion.

    Returns:
        Remaining manifest records.

    Raises:
        ValueError: If an exclusion is unknown or would remove every testcase.
    """
    excluded = {str(value).strip() for value in excluded_filenames if str(value).strip()}
    known = {str(record["filename"]) for record in records}
    unknown = excluded - known
    if unknown:
        raise ValueError(f"Cannot exclude unknown generated testcases: {sorted(unknown)}")

    remaining = [record for record in records if str(record["filename"]) not in excluded]
    if not remaining:
        raise ValueError("Critic exclusions removed every generated testcase.")

    for filename in excluded:
        (generated_dir / filename).unlink()

    (generated_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in remaining),
        encoding="utf-8",
    )
    _validate_manifest_file_set(generated_dir=generated_dir, records=remaining)

    return remaining


def _run_generator(
    *,
    script: str,
    script_path: Path,
    out_dir: Path,
    timeout_seconds: int,
) -> None:
    """Execute one generated Python testcase script.

    Args:
        script: Python source returned by an agent.
        script_path: Destination source file path.
        out_dir: Script-provided ``OUT_DIR``.
        timeout_seconds: Execution timeout.

    Raises:
        RuntimeError: If the generator returns an error or times out.
    """
    if not script.strip():
        raise ValueError("Generated testcase script is empty.")

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script.strip() + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.update({"OUT_DIR": str(out_dir)})
    try:
        completed = subprocess.run(
            [sys.executable, script_path.name],
            cwd=str(script_path.parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        raise RuntimeError(f"Testcase generator timed out after {timeout_seconds}s.\n{output}") from exc

    transcript = completed.stdout or ""
    (script_path.with_suffix(".log")).write_text(transcript, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Testcase generator failed with exit code {completed.returncode}.\n{transcript}")


def _file_sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for a generated GDS file.

    Args:
        path: Generated file to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_supplemental_append(
    *,
    generated_dir: Path,
    prior_records: list[dict[str, Any]],
    prior_hashes: dict[str, str],
    current_records: list[dict[str, Any]],
) -> None:
    """Validate that a critic script appended exactly one immutable testcase.

    Args:
        generated_dir: Directory containing generated testcase artifacts.
        prior_records: Normalized manifest before the supplemental script.
        prior_hashes: Digests of all previously generated GDS files.
        current_records: Normalized manifest after the supplemental script.

    Raises:
        ValueError: If prior artifacts changed or the append contract is broken.
    """
    if current_records[: len(prior_records)] != prior_records:
        raise ValueError("Supplemental testcase generator changed or removed existing manifest rows.")

    if len(current_records) != len(prior_records) + 1:
        raise ValueError("Supplemental testcase generator must append exactly one manifest row.")

    for filename, digest in prior_hashes.items():

        if _file_sha256(generated_dir / filename) != digest:
            raise ValueError(f"Supplemental testcase generator modified existing GDS file: {filename}")

    expected_filename = f"case_{len(prior_records):04d}.gds"
    appended_filename = str(current_records[-1]["filename"])
    if appended_filename != expected_filename:
        raise ValueError(f"Supplemental testcase generator must append {expected_filename}, got {appended_filename}.")

    _validate_manifest_file_set(generated_dir=generated_dir, records=current_records)


def _validate_manifest_file_set(*, generated_dir: Path, records: list[dict[str, Any]]) -> None:
    """Require a one-to-one mapping between manifest rows and generated GDS files.

    Args:
        generated_dir: Directory containing generated testcase artifacts.
        records: Current normalized manifest records.

    Raises:
        ValueError: If a GDS file is unmanifested or a manifest row has no GDS.
    """
    manifest_filenames = {str(record["filename"]) for record in records}
    generated_filenames = {path.name for path in generated_dir.glob("*.gds") if path.is_file()}
    if generated_filenames != manifest_filenames:
        raise ValueError("Every generated GDS file must have exactly one matching manifest row.")


def _normalize_manifest(generated_dir: Path, categories: list[str]) -> list[dict[str, Any]]:
    """Validate and normalize generated manifest records.

    Exact category keys and integer bits are required: filling missing labels
    with zero would turn absent evidence into asserted negative examples.

    Args:
        generated_dir: Generated GDS and manifest directory.
        categories: Exact expected-label keys.

    Returns:
        Sorted normalized testcase records.

    Raises:
        ValueError: If the manifest is missing or violates the testcase contract.
    """
    manifest_path = generated_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError("Generated testcase suite is missing manifest.jsonl.")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):

        if not line.strip():
            continue

        try:
            payload = json.loads(line)

        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid manifest JSON at line {line_number}: {exc.msg}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Manifest line {line_number} must be a JSON object.")

        filename = str(payload.get("filename", "")).replace("\\", "/").lstrip("./")
        if not filename or Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"Manifest line {line_number} contains an invalid filename.")

        if not re.fullmatch(r"case_\d{4}\.gds", filename):
            raise ValueError(f"Manifest line {line_number} must use a case_XXXX.gds filename.")

        if filename in seen or not (generated_dir / filename).is_file():
            raise ValueError(f"Manifest references duplicate or missing GDS file: {filename}")

        expected = payload.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(categories):
            raise ValueError("Each manifest expected mapping must contain exactly the problem categories.")

        if any(
            isinstance(expected[category], bool)
            or not isinstance(expected[category], int)
            or expected[category] not in (0, 1)
            for category in categories
        ):
            raise ValueError("Each manifest expected bit must be the integer 0 or 1.")

        intent = str(payload.get("intent", "")).strip()
        if not intent:
            raise ValueError("Each manifest row must include a non-empty testcase intent.")

        records.append(
            {
                "filename": filename,
                "expected": {category: int(expected[category]) for category in categories},
                "intent": intent,
            }
        )
        seen.add(filename)

    records.sort(key=lambda item: str(item["filename"]))

    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    if not records:
        raise ValueError("No valid generated GDS testcases were produced.")

    return records


def _merge_labels(
    *,
    labels_path: Path,
    repository_root: Path,
    generated_dir: Path,
    records: list[dict[str, Any]],
    categories: list[str],
) -> None:
    """Write labels for generated testcases without reading benchmark labels.

    Args:
        labels_path: Generated-suite ``labels.csv`` path.
        repository_root: Root used for repository-relative label paths.
        generated_dir: Directory containing generated GDS files.
        records: Normalized manifest records.
        categories: Ordered category names.
    """
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8", newline="") as labels_file:

        writer = csv.DictWriter(labels_file, fieldnames=["filename", *categories])
        writer.writeheader()
        for record in records:
            gds_path = generated_dir / str(record["filename"])
            writer.writerow(
                {
                    "filename": gds_path.resolve().relative_to(repository_root.resolve()).as_posix(),
                    **{category: record["expected"][category] for category in categories},
                }
            )
