"""Evidence-based metrics shared by evaluation artifacts and plotting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


def _count(value: Any) -> int | None:
    """Return a nonnegative integer count, or None for unavailable evidence."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0 or int(value) != value:
        return None
    return int(value)


def _sum_known(values: Iterable[int | None]) -> int | None:
    """Sum observed counts, returning None when none are available."""
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def latest_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last supplied record per dataset ID, retaining anonymous rows."""
    latest: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("id") is None or record.get("id") == "":
            anonymous.append(record)
        else:
            latest[str(record["id"])] = record
    return list(latest.values()) + anonymous


def outcome(record: dict[str, Any]) -> str:
    """Classify only explicit benchmark scores, never flow acceptance."""
    if record.get("error"):
        return "error"
    score = record.get("score")
    if not isinstance(score, dict) or score.get("error"):
        return "error" if isinstance(score, dict) and score.get("error") else "unscored"
    if score.get("success") is True:
        return "success"
    if score.get("success") is False:
        return "fail"
    return "unscored"


def record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Extract observed per-rule loop and execution counts for CSV and charts."""
    metrics = record.get("loop_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    def attempts(phase: str) -> int | None:
        """Read one phase's explicitly recorded attempt count."""
        value = metrics.get(phase)
        return _count(value.get("attempts")) if isinstance(value, dict) else None

    def origin_count(phase: str, origin: str) -> int | None:
        """Count origins only when the attempt history is available."""
        value = metrics.get(phase)
        history = value.get("attempt_records") if isinstance(value, dict) else None
        if not isinstance(history, list) or any(not isinstance(item, dict) or "origin" not in item for item in history):
            return None
        return sum(item["origin"] == origin for item in history)

    result: dict[str, Any] = {
        "id": str(record.get("id", "")),
        "benchmark_outcome": {"success": "PASS", "fail": "FAIL", "error": "ERROR", "unscored": "UNSCORED"}[outcome(record)],
        "flow_status": record.get("flow_status") or "unknown",
        "generation_llm_calls": attempts("code_generation"),
        "generation_compilation_attempts": origin_count("compilation", "initial_code_generation"),
        "code_correction_iterations": attempts("code_correction"),
        "code_correction_compilation_iterations": origin_count("compilation", "code_correction"),
        "testcase_correction_iterations": attempts("testcase_correction"),
        "critic_review_iterations": _sum_known(attempts(phase) for phase in ("testcase_adjudication", "fixed_testcase_adjudication")),
        "testcase_preparation_iterations": origin_count("testcase_preparation", "initial_testcases"),
        "testcase_correction_preparation_iterations": origin_count("testcase_preparation", "testcase_correction"),
        "total_compilation_iterations": attempts("compilation"),
        "code_generation_retry_iterations": _count(record.get("code_regeneration_iteration")),
    }
    generated = record.get("generated_testcases")
    result["generated_cases_total"] = len(generated) if isinstance(generated, list) else _count(record.get("generated_testcase_count"))
    candidates = record.get("code_candidates")
    candidates_known = isinstance(candidates, list) and all(
        isinstance(candidate, dict) and isinstance(candidate.get("compiled"), bool)
        for candidate in candidates
    )
    compiled = [candidate for candidate in candidates if candidate["compiled"]] if candidates_known else None
    selection = record.get("final_selection")
    selected_id = selection.get("selected_candidate_id") if isinstance(selection, dict) else None
    result["code_compile_nb"] = len(compiled) if compiled is not None else None
    result["code_compiled_not_selected_nb"] = (
        sum(candidate["candidate_id"] != selected_id for candidate in compiled)
        if compiled is not None and selected_id is not None
        and all(candidate.get("candidate_id") is not None for candidate in compiled) else None
    )
    executions = record.get("execution_attempts")
    executions = executions if isinstance(executions, list) else None
    result["execution_count"] = len(executions) if executions is not None else None
    for prefix, execution in (("initial", executions[0] if executions else None), ("last_execution", executions[-1] if executions else None)):
        execution = execution if isinstance(execution, dict) else {}
        cases = _count(execution.get("n_cases"))
        failures = _count(execution.get("n_failing_cases"))
        result[f"{prefix}_testcases_nb"] = cases
        result[f"{prefix}_testcases_fails_nb"] = failures
        result[f"{prefix}_testcases_fails_percentage"] = (
            round(100 * failures / cases, 2)
            if cases and failures is not None and failures <= cases else None
        )
    return result


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build JSON-compatible metrics from latest, flow-enriched evaluation records.

    Benchmark success_rate uses only explicit pass/fail scores, whereas
    cohort_success_rate uses all records. Testcase accuracy similarly excludes
    unverifiable cases, whose count remains separate from incorrect cases.

    Args:
        records: Records enriched by dataset ID. Last occurrence of an ID wins.

    Returns:
        Schema-versioned benchmark, coverage, token, request, loop and execution
        metrics. Null means unavailable; totals with partial coverage are observed
        subtotals, not estimates. Errors are excluded from scored performance.

    """
    rows = latest_records(records)
    outcomes = [outcome(record) for record in rows]
    successful = outcomes.count("success")
    failed = outcomes.count("fail")
    scored = successful + failed
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    token_values: dict[str, list[int | None]] = {field: [] for field in token_fields}
    roles: dict[str, dict[str, list[int | None]]] = {}
    role_requests: dict[str, list[bool | None]] = {}
    request_counts: list[int | None] = []
    failed_requests: list[int | None] = []
    request_outcomes_known = 0
    loop_counts: dict[str, list[int | None]] = {}
    execution_counts: list[int | None] = []
    testcase_statuses: dict[str, int] = {}
    testcase_counts: dict[str, list[int | None]] = {
        key: [] for key in ("n_cases", "n_correct", "n_incorrect", "n_unverifiable")
    }
    for record in rows:
        testcase = record.get("testcase_audit")
        if isinstance(testcase, dict):
            status = str(testcase.get("status", "unknown"))
            testcase_statuses[status] = testcase_statuses.get(status, 0) + 1
            audit = testcase.get("audit")
            if status == "completed" and isinstance(audit, dict):
                for field in testcase_counts:
                    testcase_counts[field].append(_count(audit.get(field)))
        requests = record.get("llm_request_metrics")
        requests = requests if isinstance(requests, list) else None
        valid_requests = [item for item in requests or [] if isinstance(item, dict)]
        request_counts.append(len(requests) if requests is not None else None)
        known = [item for item in valid_requests if isinstance(item.get("success"), bool)]
        request_outcomes_known += len(known)
        failed_requests.append(sum(item["success"] is False for item in known) if known or requests == [] else None)
        for request in valid_requests:
            role = str(request.get("agent") or request.get("role") or "unknown")
            usage = request.get("token_usage")
            usage = usage if isinstance(usage, dict) else {}
            bucket = roles.setdefault(role, {field: [] for field in token_fields})
            role_requests.setdefault(role, []).append(request.get("success") if isinstance(request.get("success"), bool) else None)
            for field in token_fields:
                bucket[field].append(_count(usage.get(field)))
        usage = record.get("token_usage")
        usage = usage if isinstance(usage, dict) else {}
        for field in token_fields:
            value = _count(usage.get(field))
            if value is None and requests is not None:
                value = _sum_known(_count((item.get("token_usage") or {}).get(field)) for item in valid_requests if isinstance(item.get("token_usage"), dict))
                if requests == []:
                    value = 0
            token_values[field].append(value)
        loops = record.get("loop_metrics")
        for phase, values in (loops.items() if isinstance(loops, dict) else []):
            loop_counts.setdefault(str(phase), []).append(_count(values.get("attempts")) if isinstance(values, dict) else None)
        execution_counts.append(record_summary(record)["execution_count"])
    return {
        "schema_version": 1,
        "benchmark": {
            "completed": scored, "successful": successful, "failed": failed,
            "errors": outcomes.count("error"), "unscored": outcomes.count("unscored"),
            "success_rate": successful / scored if scored else None,
            "cohort_success_rate": successful / len(rows) if rows else None,
        },
        "coverage": {
            "records": len(rows), "scored": scored,
            "token_records": {field: sum(value is not None for value in values) for field, values in token_values.items()},
            "request_records": sum(value is not None for value in request_counts),
            "request_outcomes": request_outcomes_known,
            "role_token_requests": {role: {field: sum(value is not None for value in values) for field, values in fields.items()} for role, fields in roles.items()},
            "execution_records": sum(value is not None for value in execution_counts),
        },
        "tokens": {**{field: _sum_known(values) for field, values in token_values.items()},
                   "by_role": {role: {field: _sum_known(values) for field, values in fields.items()} for role, fields in roles.items()}},
        "requests": {"total": _sum_known(request_counts), "failed": _sum_known(failed_requests),
                 "by_role": {role: {"total": len(values), "outcomes_known": sum(value is not None for value in values),
                        "failed": sum(value is False for value in values) if any(value is not None for value in values) else None}
                     for role, values in role_requests.items()}},
        "loops": {phase: {"attempts": _sum_known(values), "records": sum(value is not None for value in values)} for phase, values in loop_counts.items()},
        "executions": {"total": _sum_known(execution_counts)},
        "testcases": {
            "status_counts": testcase_statuses,
            **{field: _sum_known(values) for field, values in testcase_counts.items()},
            "accuracy": (
                sum(value or 0 for value in testcase_counts["n_correct"])
                / (sum(value or 0 for value in testcase_counts["n_correct"])
                   + sum(value or 0 for value in testcase_counts["n_incorrect"]))
                if any(testcase_counts["n_correct"]) or any(testcase_counts["n_incorrect"]) else None
            ),
        },
    }


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    """Load canonical results without rereading or overriding enriched artifacts.

    Args:
        run_dir: Run root, or its evaluation directory.

    Returns:
        Latest records from evaluation/results.json (or results.json when passed
        the evaluation directory directly).
    """
    path = run_dir / "evaluation" / "results.json"
    if not path.is_file():
        path = run_dir / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("results"), list):
        raise ValueError(f"Invalid canonical evaluation results: {path}")
    if any(not isinstance(record, dict) for record in payload["results"]):
        raise ValueError(f"Invalid result record: {path}")
    return latest_records(payload["results"])