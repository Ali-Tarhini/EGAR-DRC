"""Validate and atomically materialize benchmark rows without network access."""

from __future__ import annotations

import csv
import io
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


def _relative_path(value: Any) -> Path:
    """Return a safe relative path, rejecting POSIX and Windows escapes."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*path.parts)


def _problem_id(row: Mapping[str, Any]) -> str:
    """Read an identifier that is a single safe directory name."""
    problem_id = row.get("problem_id")
    if not isinstance(problem_id, str) or len(_relative_path(problem_id).parts) != 1 or any(
        ord(character) < 32 or character == ":" for character in problem_id
    ):
        raise ValueError(f"Invalid problem_id: {problem_id!r}")
    return problem_id


def _write(root: Path, relative: Path, content: bytes) -> None:
    """Write bytes once within the private staging directory."""
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(content)


def _text(row: Mapping[str, Any], key: str) -> str:
    """Require a nonempty text field without rewriting its contents."""
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key}")
    return value


def _case_path(row: Mapping[str, Any], data_dir: Path) -> Path:
    """Resolve either testcase path representation into repository coordinates."""
    raw_path = row.get("gds_path")
    if raw_path:
        relative = _relative_path(raw_path)
    else:
        relative = _relative_path(row.get("filename"))
        if len(relative.parts) == 1:
            split_hint = row.get("split_hint")
            if split_hint not in {"pass", "fail"}:
                raise ValueError("A bare testcase filename requires split_hint pass or fail")
            relative = Path(split_hint) / relative
    if relative.parts[0] != data_dir.parts[0]:
        relative = data_dir / relative
    if not relative.is_relative_to(data_dir) or relative.suffix != ".gds":
        raise ValueError(f"Testcase must be a .gds file under {data_dir}: {relative}")
    return relative


def materialize_dataset(
    tasks: Iterable[Mapping[str, Any]],
    testcases: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    repository: str,
    requested_revision: str,
    resolved_revision: str,
    fingerprints: Mapping[str, str | None] | None = None,
) -> Path:
    """Publish validated benchmark files at a new destination.

    Args:
        tasks: Rows containing problem_id, spec_yaml, gold_drc and labels_csv.
        testcases: Rows containing problem_id, a path and binary GDS content.
        output_dir: New directory to publish; existing destinations are refused.
        repository: Hugging Face dataset repository identifier for the manifest.
        requested_revision: User-selected branch, tag or commit.
        resolved_revision: Immutable snapshot commit used by the loader.
        fingerprints: Optional fingerprints of the two loaded splits.

    Returns:
        Absolute path usable as the runner's --repository argument.

    Raises:
        ValueError: If rows, references or destination paths are unsafe or incomplete.
        FileExistsError: If the destination already exists.
    """
    destination = output_dir.expanduser().absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("Output path must not contain symlinks")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; choose a new --output-dir: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        problems: dict[str, tuple[Path, set[str]]] = {}
        actual: dict[str, set[str]] = {}
        for task in tasks:
            problem_id = _problem_id(task)
            if problem_id in problems:
                raise ValueError(f"Duplicate task: {problem_id}")
            spec_text = _text(task, "spec_yaml")
            spec = yaml.safe_load(spec_text)
            if not isinstance(spec, dict):
                raise ValueError(f"Invalid YAML mapping for {problem_id}")
            problem_dir = Path("problems") / problem_id
            data_dir = problem_dir / "data" / "gds"
            if _relative_path(spec.get("data_dir")) != data_dir:
                raise ValueError(f"Unexpected data_dir for {problem_id}")
            gold_path = problem_dir / "gold" / f"{problem_id}.drc"
            if "deck" in spec and _relative_path(spec["deck"]) != gold_path:
                raise ValueError(f"Unexpected gold deck path for {problem_id}")
            categories = spec.get("categories")
            if not isinstance(categories, list) or not categories or any(
                not isinstance(category, str) or not category for category in categories
            ) or len(set(categories)) != len(categories) or "filename" in categories:
                raise ValueError(f"Invalid categories for {problem_id}")
            labels_text = _text(task, "labels_csv")
            reader = csv.DictReader(io.StringIO(labels_text, newline=""))
            if reader.fieldnames is None or set(reader.fieldnames) != {"filename", *categories} or (
                len(reader.fieldnames) != len(categories) + 1
            ):
                raise ValueError(f"Label columns do not match categories for {problem_id}")
            expected: set[str] = set()
            for label in reader:
                filename = _relative_path(label.get("filename")).as_posix()
                if not Path(filename).is_relative_to(data_dir) or not filename.endswith(".gds"):
                    raise ValueError(f"Label outside data_dir for {problem_id}: {filename}")
                if filename in expected:
                    raise ValueError(f"Duplicate label for {problem_id}: {filename}")
                if None in label or any(label[category] not in {"0", "1"} for category in categories):
                    raise ValueError(f"Invalid binary labels for {problem_id}: {filename}")
                expected.add(filename)
            if not expected:
                raise ValueError(f"No labels for {problem_id}")
            problems[problem_id] = (data_dir, expected)
            actual[problem_id] = set()
            _write(staging, problem_dir / "spec.yaml", spec_text.encode("utf-8"))
            _write(staging, gold_path, _text(task, "gold_drc").encode("utf-8"))
            _write(staging, data_dir / "labels.csv", labels_text.encode("utf-8"))
        if not problems:
            raise ValueError("No tasks found")
        testcase_count = 0
        for testcase in testcases:
            problem_id = _problem_id(testcase)
            if problem_id not in problems:
                raise ValueError(f"Orphan testcase: {problem_id}")
            data_dir, expected = problems[problem_id]
            relative = _case_path(testcase, data_dir)
            filename = relative.as_posix()
            if filename not in expected or filename in actual[problem_id]:
                raise ValueError(f"Unlabelled or duplicate testcase for {problem_id}: {filename}")
            content = testcase.get("gds")
            if content is None:
                content = testcase.get("gds_bytes")
            if not isinstance(content, (bytes, bytearray, memoryview)) or not content:
                raise ValueError(f"Missing binary GDS for {filename}")
            _write(staging, relative, bytes(content))
            actual[problem_id].add(filename)
            testcase_count += 1
        for problem_id, (_, expected) in problems.items():
            if actual[problem_id] != expected:
                raise ValueError(f"Missing testcases for {problem_id}: {sorted(expected - actual[problem_id])}")
        manifest = {
            "dataset_repository": repository,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "split": "test",
            "task_count": len(problems),
            "testcase_count": testcase_count,
            "fingerprints": dict(fingerprints or {}),
        }
        _write(staging, Path("dataset_manifest.json"), (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination appeared during preparation: {destination}")
        staging.rename(destination)
    return destination