"""Resumable, post-generation evaluation with content-verified step artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from egar_drc.evaluation.dataset import find_problem_dir, load_spec, resolve_repository_path
from egar_drc.evaluation.generated_testcase_audit import audit_cached_testcase_suite
from egar_drc.evaluation.klayout_scoring import evaluate_deck
from egar_drc.evaluation.metrics import latest_records, summarize

STEPS = ("benchmark", "testcases", "metrics", "plots")
LOGGER = logging.getLogger(__name__)


def read_json(path: Path) -> Any:
    """Read one JSON artifact, propagating malformed or missing input errors."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Atomically replace a JSON artifact after serializing finite values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    """Stream a file's SHA-256 without retaining large GDS files in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_steps(steps: list[str] | tuple[str, ...]) -> list[str]:
    """Expand all and dependency-order unique steps; plots require metrics."""
    unknown = set(steps) - {*STEPS, "all"}
    if unknown:
        raise ValueError(f"Unknown evaluation steps: {sorted(unknown)}")
    chosen = set(STEPS if "all" in steps else steps)
    if "plots" in chosen:
        chosen.add("metrics")
    return [step for step in STEPS if step in chosen]


class RunEvaluation:
    """Own canonical per-run results and an independently resumable manifest.

    Args:
        run_dir: Generation output directory containing results.jsonl.
        repository: Dataset override, required only for gold-based steps.
        klayout_bin: KLayout executable used for benchmark and testcase audits.
    """

    def __init__(self, run_dir: Path, repository: str | None = None, klayout_bin: str = "klayout") -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.output_dir = self.run_dir / "evaluation"
        self.repository = repository
        self.klayout_bin = klayout_bin
        self.manifest_path = self.output_dir / "manifest.json"
        self.records = self._load_records()
        metadata_path = self.run_dir / "run_metadata.json"
        self.metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {
            "schema_version": 1, "steps": {},
        }
        if self.manifest.get("schema_version") != 1 or not isinstance(self.manifest.get("steps"), dict):
            raise ValueError(f"Invalid evaluation manifest: {self.manifest_path}")

    def _load_records(self) -> list[dict[str, Any]]:
        """Read complete generation records and enrich them by dataset ID."""
        records: list[dict[str, Any]] = []
        path = self.run_dir / "results.jsonl"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not record.get("id"):
                raise ValueError(f"Invalid generation record at {path}:{number}")
            problem_id = str(record["id"])
            if Path(problem_id).name != problem_id or problem_id in {".", ".."}:
                raise ValueError(f"Invalid problem ID: {problem_id}")
            artifact = self.run_dir / problem_id / "flow_result.json"
            flow = read_json(artifact) if artifact.exists() else {}
            if not isinstance(flow, dict):
                raise ValueError(f"Invalid flow artifact: {artifact}")
            records.append({**flow, **record})
        return latest_records(records)

    def _deck(self, record: dict[str, Any]) -> Path:
        """Prefer a portable run-local deck over its historical absolute path."""
        local = self.run_dir / str(record["id"]) / f"{record['id']}.drc"
        return local if local.is_file() else Path(str(record.get("deck", local))).expanduser().resolve()

    def _dataset(self) -> Path:
        """Resolve the explicit dataset input without trusting artifact paths."""
        return resolve_repository_path(self.repository)

    def _inputs(self, step: str) -> dict[str, str]:
        """Fingerprint evidence, evaluator code, and dependencies for reuse."""
        paths = {self.run_dir / "results.jsonl"}
        paths.add(self.run_dir / "run_metadata.json")
        paths.update(self.run_dir / str(record["id"]) / "flow_result.json" for record in self.records)
        paths.update(Path(__file__).parent.glob("*.py"))
        if step in {"benchmark", "testcases"}:
            repository = self._dataset()
            executable = shutil.which(self.klayout_bin)
            if executable is None:
                raise FileNotFoundError(f"KLayout executable not found: {self.klayout_bin}")
            paths.add(Path(executable).resolve())
            for record in self.records:
                problem = find_problem_dir(repository / "problems", str(record["id"]))
                paths.add(problem / "spec.yaml")
                expected_specs = {str(item["id"]): item.get("spec_sha256")
                                  for item in self.metadata.get("problems", [])}
                expected = expected_specs.get(str(record["id"]))
                if expected and file_hash(problem / "spec.yaml") != expected:
                    raise ValueError(f"Dataset specification differs from generation: {record['id']}")
                spec = load_spec(problem)
                data = (repository / str(spec["data_dir"])).resolve()
                data.relative_to(repository)
                if step == "benchmark":
                    paths.add(self._deck(record))
                    paths.update(path for path in data.rglob("*") if path.is_file())
                else:
                    gold = (repository / str(spec["deck"])).resolve()
                    gold.relative_to(repository)
                    paths.add(gold)
                    cache = self.run_dir / str(record["id"]) / "testcase_suite_cache"
                    paths.update(path for path in cache.rglob("*") if path.is_file())
        else:
            paths.add(self.manifest_path)
            paths.update(self.output_dir / name for name in ("benchmark.json", "testcases.json"))
            if step == "plots":
                paths.update(self.output_dir / name for name in ("results.json", "metrics.json"))
        inputs = {str(path): file_hash(path) if path.is_file() else "missing" for path in sorted(paths)
                  if path != self.manifest_path}
        if step in {"metrics", "plots"}:
            # Detect stale dependencies without hashing the step's own output.
            dependencies = {
                key: {"entry": self.manifest["steps"].get(key), "current_inputs": self._inputs(key)}
                for key in ("benchmark", "testcases")
                if self.manifest["steps"].get(key, {}).get("status") == "completed"
            }
            inputs["upstream_steps"] = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()).hexdigest()
        return inputs

    def _current(self, step: str, inputs: dict[str, str]) -> bool:
        """Require matching inputs and output hashes, not just a completion flag."""
        entry = self.manifest["steps"].get(step, {})
        outputs = entry.get("outputs", {})
        return bool(
            entry.get("status") == "completed" and entry.get("inputs") == inputs and outputs
            and all((self.output_dir / name).is_file() and file_hash(self.output_dir / name) == digest
                    for name, digest in outputs.items())
        )

    def run(self, steps: list[str] | tuple[str, ...] = ("metrics",), force: bool = False) -> dict[str, Any]:
        """Run missing or stale steps and record their outcomes."""
        failures: list[str] = []
        for step in selected_steps(steps):
            LOGGER.info("EVALUATION run=%s step=%s", self.run_dir.name, step)
            try:
                inputs = self._inputs(step)
                if not force and self._current(step, inputs):
                    LOGGER.info("REUSED run=%s step=%s", self.run_dir.name, step)
                    continue
                outputs = self._execute(step)
                entry = {
                    "status": "completed", "inputs": inputs,
                    "outputs": {str(path.relative_to(self.output_dir)): file_hash(path) for path in outputs},
                }
                if step in {"benchmark", "testcases"}:
                    rows = read_json(outputs[0])["results"]
                    # Track artifact completion separately from row coverage.
                    entry["coverage"] = {
                        "records": len(rows),
                        "completed": sum("error" not in item.get("score", {}) for item in rows)
                                     if step == "benchmark" else sum(item["status"] == "completed" for item in rows),
                        "errors": sum("error" in item.get("score", {}) for item in rows)
                                  if step == "benchmark" else sum(item["status"] == "error" for item in rows),
                        "skipped": sum(item.get("status") == "skipped" for item in rows),
                    }
            except Exception as exc:
                failures.append(step)
                entry = {"status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)}}
                LOGGER.error("FAILED run=%s step=%s detail=%s", self.run_dir.name, step, exc)
            entry["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.manifest["steps"][step] = entry
            write_json(self.manifest_path, self.manifest)
        if failures:
            raise RuntimeError(f"Evaluation steps failed: {', '.join(failures)}; see {self.manifest_path}")
        return self.manifest

    def _execute(self, step: str) -> list[Path]:
        """Dispatch a bounded set of public evaluation operations."""
        if step == "benchmark":
            return self._benchmark()
        if step == "testcases":
            return self._testcases()
        if step == "metrics":
            return self._metrics()
        destination = self.output_dir / "plots"
        # Isolate optional plotting dependencies.
        subprocess.run(
            [sys.executable, "-m", "egar_drc.evaluation.plotting", str(self.run_dir), "--output", str(destination)],
            check=True,
        )
        outputs = sorted(path for path in destination.rglob("*") if path.is_file())
        if not outputs:
            raise RuntimeError("Plotting produced no artifacts")
        return outputs

    def _benchmark(self) -> list[Path]:
        """Re-score final decks without changing generation artifacts."""
        repository = self._dataset()
        results: list[dict[str, Any]] = []
        for record in self.records:
            problem_id = str(record["id"])
            try:
                problem = find_problem_dir(repository / "problems", problem_id)
                deck = self._deck(record)
                if not deck.is_file():
                    raise FileNotFoundError(deck)
                score = evaluate_deck(problem, deck, repository, self.output_dir / "benchmark" / problem_id, self.klayout_bin)
                results.append({"id": problem_id, "score": score})
            except Exception as exc:
                results.append({"id": problem_id, "score": {"error": {"type": type(exc).__name__, "message": str(exc)}}})
        path = self.output_dir / "benchmark.json"
        write_json(path, {"schema_version": 1, "results": results})
        return [path, *sorted(item for item in (self.output_dir / "benchmark").rglob("*") if item.is_file())]

    def _testcases(self) -> list[Path]:
        """Audit exact cached suites, explicitly skipping unavailable evidence."""
        repository = self._dataset()
        results: list[dict[str, Any]] = []
        for record in self.records:
            problem_id = str(record["id"])
            inventory = record.get("generated_testcases")
            if not isinstance(inventory, list) or not inventory:
                results.append({"id": problem_id, "status": "skipped", "reason": "No recorded generated suite"})
                continue
            problem = find_problem_dir(repository / "problems", problem_id)
            spec = load_spec(problem)
            cache = self.run_dir / problem_id / "testcase_suite_cache"
            suites = [path.parent for path in sorted(cache.glob("*/problems"))
                      if (path / problem.name).is_dir()]
            matching = [suite for suite in suites
                        if (suite / str(spec["data_dir"]) / "generated" / "inventory.json").is_file()
                        and read_json(suite / str(spec["data_dir"]) / "generated" / "inventory.json") == inventory]
            if not matching:
                results.append({"id": problem_id, "status": "skipped", "reason": "No matching persisted suite"})
                continue
            try:
                audit = audit_cached_testcase_suite(
                    problem_dir=problem, repository_root=repository, suite_repository_root=matching[0],
                    expected_inventory=inventory, critic_rejected_filenames=[],
                    output_dir=self.output_dir / "testcases" / problem_id, klayout_bin=self.klayout_bin,
                )
                results.append({"id": problem_id, "status": "completed", "audit": audit})
            except Exception as exc:
                results.append({"id": problem_id, "status": "error", "error": str(exc)})
        path = self.output_dir / "testcases.json"
        write_json(path, {"schema_version": 1, "evaluation_only": True, "results": results})
        return [path, *sorted(item for item in (self.output_dir / "testcases").rglob("*") if item.is_file())]

    def _metrics(self) -> list[Path]:
        """Publish canonical records using verified stage-2 results."""
        records = [dict(record) for record in self.records]
        for filename, key in (("benchmark.json", "score"), ("testcases.json", "testcase_audit")):
            path = self.output_dir / filename
            step = "benchmark" if key == "score" else "testcases"
            entry = self.manifest["steps"].get(step, {})
            if not path.exists():
                continue
            if entry.get("status") != "completed":
                raise ValueError(f"Unverified {step} artifact; select {step} before metrics")
            if not self._current(step, self._inputs(step)):
                raise ValueError(f"Stale {step} results; select {step} before metrics")
            additions = {str(item["id"]): item for item in read_json(path)["results"]}
            for record in records:
                item = additions.get(str(record["id"]))
                if item is not None:
                    record[key] = item["score"] if key == "score" else item
                    if key == "score" and "error" in record:
                        record["generation_error"] = record.pop("error")
        result_path = self.output_dir / "results.json"
        metrics_path = self.output_dir / "metrics.json"
        write_json(result_path, {"schema_version": 1, "results": records})
        write_json(metrics_path, summarize(records))
        return [result_path, metrics_path]


def main() -> None:
    """Evaluate selected operations for one completed generation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", nargs="+", choices=(*STEPS, "all"), default=["benchmark", "testcases", "metrics"])
    parser.add_argument("--repository")
    parser.add_argument("--klayout-bin", default="klayout")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RunEvaluation(args.run_dir, args.repository, args.klayout_bin).run(args.steps, args.force)


if __name__ == "__main__":
    main()