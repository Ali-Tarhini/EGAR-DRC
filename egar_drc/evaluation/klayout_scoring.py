"""KLayout DRC scoring for Rule2DRC agentic-flow evaluation.

This module mirrors the scoring logic of the Rule2DRC benchmark evaluator
(``evaluate/klayout_eval.py``) so that decks produced by the agentic flow are
scored identically to the other Rule2DRC methods. The only intentional
difference is that testcase paths are keyed against an explicit repository root
instead of the process working directory, which keeps scoring correct when the
evaluator is called from a runtime, MCP tool, or worker thread.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

_KLAYOUT_TIMEOUT_SECONDS = 120
_EMPTY_LAYOUT_CACHE_ROOT = Path(tempfile.gettempdir()) / "rule2drc_empty_layouts"
_EMPTY_LAYOUT_CACHE_LOCK = Lock()


def run_klayout(klayout_bin: str, deck: Path, gds: Path, rdb: Path, timeout_s: int = _KLAYOUT_TIMEOUT_SECONDS) -> bool:
    """Run a KLayout DRC deck against a single GDS file in batch mode.

    Args:
        klayout_bin: KLayout executable name or path.
        deck: Path to the Ruby DRC deck.
        gds: Path to the GDS testcase.
        rdb: Path where the KLayout report database is written.
        timeout_s: Hard timeout for the KLayout process in seconds.

    Returns:
        True when KLayout exits cleanly and writes the report database.
    """
    rdb.parent.mkdir(parents=True, exist_ok=True)
    rdb.unlink(missing_ok=True)
    command = [
        klayout_bin,
        "-b",
        "-r",
        str(deck),
        "-rd",
        f"input={gds.resolve()}",
        "-rd",
        f"output={rdb.resolve()}",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    return completed.returncode == 0 and rdb.exists()


def compile_check_deck(
    deck_path: Path,
    gds: Path,
    klayout_bin: str = "klayout",
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Run a DRC deck once in KLayout to check that it executes without error.

    This is a compilation/runtime smoke check only: it verifies the deck runs
    under KLayout on a single GDS testcase. It does not compare results against
    any labelled ground truth, so it does not reveal correctness of the deck.

    Args:
        deck_path: Path to the Ruby DRC deck.
        gds: Path to a GDS file used only to execute the deck.
        klayout_bin: KLayout executable name or path.
        timeout_s: Hard timeout for the KLayout process in seconds.

    Returns:
        Mapping with ``ok`` (bool), ``returncode`` (int), and ``error`` (str).
    """
    with tempfile.TemporaryDirectory(prefix="rule2drc_compile_") as temp_dir:

        rdb = Path(temp_dir) / "compile_check.lyrdb"
        command = [
            klayout_bin,
            "-b",
            "-r",
            str(deck_path),
            "-rd",
            f"input={gds.resolve()}",
            "-rd",
            f"output={rdb.resolve()}",
        ]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=int(timeout_s),
                check=False,
            )

        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": -1, "error": f"KLayout timed out after {timeout_s}s."}

        except FileNotFoundError:
            return {"ok": False, "returncode": -1, "error": f"KLayout binary not found: {klayout_bin}"}

        ok = completed.returncode == 0 and rdb.exists()
        error = ""
        if not ok:
            error = (completed.stderr or completed.stdout or "").strip()

        return {"ok": ok, "returncode": completed.returncode, "error": error}


def compile_check_deck_without_testcases(
    deck_path: Path,
    klayout_bin: str = "klayout",
    timeout_s: int = 60,
) -> dict[str, Any]:
    """Smoke-check a DRC deck using a KLayout-created empty GDS layout.

    The candidate deck requires ``$input`` and ``$output`` runtime variables, so
    a valid empty GDS is created first. No generated or benchmark testcase is
    read by this check.

    Args:
        deck_path: Candidate Ruby DRC deck.
        klayout_bin: KLayout executable name or path.
        timeout_s: Timeout for each KLayout process.

    Returns:
        Mapping with ``ok``, ``returncode``, and diagnostic ``error`` fields.
    """
    # Cache the smoke-check fixture by executable path.
    cache_key = hashlib.sha256(str(Path(klayout_bin).expanduser()).encode("utf-8")).hexdigest()[:16]
    empty_gds = _EMPTY_LAYOUT_CACHE_ROOT / f"{cache_key}.gds"
    with _EMPTY_LAYOUT_CACHE_LOCK:

        if not empty_gds.is_file():
            _EMPTY_LAYOUT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            generator = _EMPTY_LAYOUT_CACHE_ROOT / f"{cache_key}.rb"
            generator.write_text(
                "layout = RBA::Layout.new\n" 'layout.create_cell("TOP")\n' "layout.write($output)\n",
                encoding="utf-8",
            )
            try:
                completed = subprocess.run(
                    [klayout_bin, "-b", "-r", str(generator), "-rd", f"output={empty_gds}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=int(timeout_s),
                    check=False,
                )

            except subprocess.TimeoutExpired:
                return {"ok": False, "returncode": -1, "error": f"Empty-layout creation timed out after {timeout_s}s."}

            except FileNotFoundError:
                return {"ok": False, "returncode": -1, "error": f"KLayout binary not found: {klayout_bin}"}

            if completed.returncode != 0 or not empty_gds.is_file():
                error = (completed.stderr or completed.stdout or "KLayout did not create the empty GDS.").strip()

                return {"ok": False, "returncode": completed.returncode, "error": error}

    return compile_check_deck(
        deck_path=deck_path,
        gds=empty_gds,
        klayout_bin=klayout_bin,
        timeout_s=timeout_s,
    )


def parse_rdb_counts(rdb: Path) -> dict[str, int]:
    """Parse per-category violation counts from a KLayout report database.

    Args:
        rdb: Path to a KLayout ``.lyrdb`` report database.

    Returns:
        Mapping of category name to violation-item count.
    """
    counts: dict[str, int] = defaultdict(int)
    if not rdb.exists():
        return dict(counts)

    root = ET.parse(str(rdb)).getroot()

    # Form A: <category name="..."><item>...</item></category>.
    # ---------------------------------------------------------
    for category in root.findall(".//category"):

        name = category.get("name") or ""
        if name:
            counts[name] += len(category.findall("./item"))

    # Form B: <items><item><category>NAME</category>...</item></items>.
    # -----------------------------------------------------------------
    for item in root.findall(".//items/item"):

        name = (item.findtext("category") or "").strip()
        if name:
            counts[name] += 1

    return dict(counts)


def read_labels_csv(labels_csv: Path, categories: list[str]) -> dict[str, dict[str, int]]:
    """Read ground-truth labels keyed by repository-relative GDS path.

    Args:
        labels_csv: Path to the problem ``labels.csv`` file.
        categories: Category names to extract from each row.

    Returns:
        Mapping of filename to a category-to-bit dictionary.
    """
    ground_truth: dict[str, dict[str, int]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as labels_file:

        reader = csv.DictReader(labels_file)
        for row_number, row in enumerate(reader, start=2):

            filename = str(row.get("filename", "")).strip()
            if not filename:
                raise ValueError(f"Missing filename in {labels_csv} row {row_number}.")

            if filename in ground_truth:
                raise ValueError(f"Duplicate filename in {labels_csv}: {filename}")

            labels: dict[str, int] = {}
            for category in categories:

                raw_value = str(row.get(category, "")).strip()
                if raw_value not in {"0", "1"}:
                    raise ValueError(f"Label {category!r} for {filename} must be 0 or 1, got {raw_value!r}.")

                labels[category] = int(raw_value)

            ground_truth[filename] = labels

    return ground_truth


def f1_from_counts(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute confusion-matrix metrics for one category.

    Benchmark parity requires zero precision/recall/F1 for zero denominators.
    Thus an all-negative, perfectly matched category still has F1 zero and
    cannot satisfy the evaluator's strict F1-equals-one success criterion.

    Args:
        y_true: Ground-truth bits.
        y_pred: Predicted bits.

    Returns:
        Mapping with tp/fp/fn/tn, precision, recall, f1, and accuracy.
    """
    true_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    false_positive = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    false_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    true_negative = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positive + true_negative) / max(
        1, (true_positive + true_negative + false_positive + false_negative)
    )

    return {
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
        "tn": float(true_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _read_csv_header(path: Path) -> list[str]:
    """Return the header row of a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Header column names.
    """
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return next(csv.reader(csv_file))


def _write_predicted_labels_csv(out_csv: Path, rows: list[tuple[str, dict[str, int]]], categories: list[str]) -> None:
    """Write predicted labels in the Rule2DRC predicted-labels format.

    Args:
        out_csv: Output CSV path.
        rows: Rows of (filename, category-to-bit) predictions.
        categories: Ordered category names.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as csv_file:

        writer = csv.writer(csv_file)
        writer.writerow(["filename", *categories])
        for filename, predictions in rows:
            writer.writerow([filename, *[predictions.get(category, 0) for category in categories]])


def evaluate_deck(
    problem_dir: Path,
    deck_path: Path,
    repository_root: Path,
    out_dir: Path | None = None,
    klayout_bin: str = "klayout",
) -> dict[str, Any]:
    """Evaluate a DRC deck against a problem's labelled GDS testcases.

    The success criterion matches the Rule2DRC benchmark: every category must
    reach an F1 of 1.0, the spec categories must match the label columns, the
    deck must not emit unknown categories, and no per-file mismatch may remain.

    Args:
        problem_dir: Problem directory containing ``spec.yaml``.
        deck_path: Path to the DRC deck under evaluation.
        repository_root: Repository root used to key labels by relative path.
        out_dir: Optional directory for report databases and result files.
        klayout_bin: KLayout executable name or path.

    Returns:
        Result payload mirroring the Rule2DRC evaluator output shape.
    """
    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="rule2drc_eval_") as temp_dir:
            return evaluate_deck(problem_dir, deck_path, repository_root, Path(temp_dir), klayout_bin)

    out_dir.mkdir(parents=True, exist_ok=True)
    repository_root = repository_root.resolve()

    spec = yaml.safe_load((problem_dir / "spec.yaml").read_text(encoding="utf-8"))
    problem_id = spec.get("id", problem_dir.name)
    title = spec.get("title", problem_id)
    categories = list(spec["categories"])
    spec_categories = set(categories)

    gds_dir = repository_root / spec["data_dir"]
    labels_csv = gds_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    header = _read_csv_header(labels_csv)
    label_columns = {column for column in header if column != "filename"}
    precheck: dict[str, list[str]] = {}
    if missing := sorted(spec_categories - label_columns):
        precheck["labels_missing"] = missing

    if extra := sorted(label_columns - spec_categories):
        precheck["labels_extra"] = extra

    ground_truth = read_labels_csv(labels_csv, categories)
    files = sorted(path for path in gds_dir.rglob("*.gds") if path.is_file())
    if not files:
        payload = {
            "id": problem_id,
            "title": title,
            "deck": str(deck_path),
            "gds_dir": str(gds_dir),
            "success": False,
            "n_cases": 0,
            "compile_rate": 0.0,
            "precheck": {"no_gds_files": [str(gds_dir)]},
            "mismatches": {},
        }
        (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        return payload

    lyrdb_dir = out_dir / "lyrdb"
    predicted_rows: list[tuple[str, dict[str, int]]] = []
    compiled = 0
    truth_by_category: dict[str, list[int]] = {category: [] for category in categories}
    pred_by_category: dict[str, list[int]] = {category: [] for category in categories}
    unknown_deck_categories: set[str] = set()
    per_file_mismatches: dict[str, list[tuple[str, int, int]]] = {}
    case_results: dict[str, dict[str, dict[str, int]]] = {}
    evaluated_paths: set[str] = set()

    for file_index, gds in enumerate(files):

        relative_gds = gds.resolve().relative_to(repository_root).as_posix()
        evaluated_paths.add(relative_gds)
        rdb = (lyrdb_dir / relative_gds).with_suffix(".lyrdb")
        compiled_ok = run_klayout(klayout_bin, deck_path, gds, rdb)
        if not compiled_ok:
            precheck.setdefault("klayout_failed_cases", []).append(relative_gds)
            if file_index == 0:
                precheck["klayout_execution"] = ["KLayout did not produce a report database."]

        counts: dict[str, int] = {}
        if compiled_ok:
            try:
                counts = parse_rdb_counts(rdb)
            except ET.ParseError as exc:
                compiled_ok = False
                precheck.setdefault("malformed_rdb_cases", []).append(
                    {"path": relative_gds, "error": str(exc)}
                )
        compiled += int(compiled_ok)
        unknown_deck_categories.update(set(counts.keys()) - spec_categories)

        row = ground_truth.get(relative_gds)
        # Exclude fallback results from gold-audit evidence.
        if row is None:
            precheck.setdefault("labels_missing_rows", []).append(relative_gds)
            row = {category: 0 for category in categories}

        predictions: dict[str, int] = {}
        for category in categories:

            truth_bit = int(row.get(category, 0))
            pred_bit = 1 if counts.get(category, 0) > 0 else 0
            predictions[category] = pred_bit
            truth_by_category[category].append(truth_bit)
            pred_by_category[category].append(pred_bit)
            if truth_bit != pred_bit:
                per_file_mismatches.setdefault(relative_gds, []).append((category, truth_bit, pred_bit))

        predicted_rows.append((relative_gds, predictions))
        case_results[relative_gds] = {
            "expected": {category: int(row.get(category, 0)) for category in categories},
            "observed": predictions,
        }

    extra_label_rows = sorted(set(ground_truth) - evaluated_paths)
    if extra_label_rows:
        precheck["labels_extra_rows"] = extra_label_rows

    _write_predicted_labels_csv(out_dir / "predicted_labels.csv", predicted_rows, categories)

    success = True
    per_category_f1: dict[str, float] = {}
    for category in categories:
        category_f1 = f1_from_counts(truth_by_category[category], pred_by_category[category])["f1"]
        per_category_f1[category] = category_f1
        success = success and (category_f1 == 1.0)

    if unknown_deck_categories:
        precheck["deck_unknown"] = sorted(unknown_deck_categories)

    # Require valid execution and exact matches alongside category F1.
    if precheck or per_file_mismatches:
        success = False

    payload = {
        "id": problem_id,
        "title": title,
        "deck": str(deck_path),
        "gds_dir": str(gds_dir),
        "success": bool(success),
        "n_cases": len(files),
        "compile_rate": compiled / len(files),
        "per_category_f1": per_category_f1,
        "precheck": precheck,
        "mismatches": {key: [list(item) for item in value] for key, value in per_file_mismatches.items()},
        "case_results": case_results,
    }
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return payload
