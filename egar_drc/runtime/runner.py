"""Run bounded problem workers with separate generation and scoring stages."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from egar_drc.evaluation.dataset import build_flow_input, iter_problem_dirs, resolve_repository_path
from egar_drc.evaluation.klayout_scoring import evaluate_deck
from egar_drc.flow_config import DEFAULT_EGAR_DRC_CONFIG
from egar_drc.models.config import ModelConfig
from egar_drc.models.runtime import ModelRuntime
from egar_drc.orchestration.code_generation_baseline import EgarDrcCodeGenerationBaselineFlow
from egar_drc.orchestration.code_only_repair import EgarDrcCodeOnlyRepairFlow
from egar_drc.orchestration.egar_drc import EgarDrcFlow
from egar_drc.orchestration.single_sample_baseline import EgarDrcSingleSampleBaselineFlow
from egar_drc.runtime.logging_setup import run_logger, sanitize_detail
from egar_drc.runtime.provenance import build_run_metadata

FLOW_TYPES: dict[str, type[EgarDrcFlow]] = {
    "egar-drc": EgarDrcFlow,
    "code_only_repair": EgarDrcCodeOnlyRepairFlow,
    "code_generation_baseline": EgarDrcCodeGenerationBaselineFlow,
    "single_sample_baseline": EgarDrcSingleSampleBaselineFlow,
}


def _load_completed(path: Path) -> set[str]:
    """Find recorded problem IDs for resume, including failures."""
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id"):
            completed.add(str(record["id"]))
    return completed


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a record, leaving ordering and concurrent-write locking to callers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def _log_agent_failures(
    logger: logging.Logger, problem_id: str, metrics: list[dict[str, Any]], api_key: str,
) -> None:
    """Log filtered role and model failures after the flow."""
    for metric in metrics:
        failures = [
            (f"model_call={index}", call)
            for index, call in enumerate(metric.get("model_calls", []), start=1)
            if call.get("success") is False or call.get("error")
        ]
        if metric.get("success") is False or metric.get("error"):
            failures.insert(0, ("role", metric))
        for source, failure in failures:
            error = failure.get("error", {})
            detail = error.get("message", "no error details") if isinstance(error, dict) else error
            error_type = error.get("type", "unknown") if isinstance(error, dict) else "unknown"
            logger.error(
                "AGENT_FAILURE problem=%s agent=%s request=%s source=%s type=%s detail=%s",
                problem_id, sanitize_detail(metric.get("agent", "unknown"), api_key),
                metric.get("request_index", "unknown"), source,
                sanitize_detail(error_type, api_key), sanitize_detail(detail, api_key),
            )


async def run_experiment(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run selected problems with invocation-owned progress handlers.

    Returns:
        Records produced in this invocation, excluding ids skipped by resume.
        Persisted summaries also include records from earlier invocations.
    """
    output_dir = Path(args.output_dir).expanduser().resolve()
    with run_logger(output_dir, getattr(args, "log_level", "INFO")) as logger:
        return await _run_experiment(args, logger)


async def _run_experiment(args: argparse.Namespace, logger: logging.Logger) -> list[dict[str, Any]]:
    """Run concurrent problem workers with a shared model runtime."""
    run_started = time.perf_counter()
    repository = resolve_repository_path(args.repository)
    problems = iter_problem_dirs(repository / "problems", args.problem or None)
    if args.limit:
        problems = problems[: args.limit]
    klayout_bin = shutil.which(args.klayout_bin)
    if klayout_bin is None:
        raise FileNotFoundError(f"KLayout executable not found: {args.klayout_bin}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = output_dir / "results.jsonl"
    completed = _load_completed(result_path) if args.resume else set()
    pending = [problem for problem in problems if problem.name not in completed]
    semaphore = asyncio.Semaphore(args.jobs)
    write_lock = asyncio.Lock()
    model_config = ModelConfig.from_yaml(args.model_config)
    runtime = ModelRuntime.from_config(model_config)
    requested_max_testcases = getattr(args, "max_testcases", None)
    effective_max_testcases = requested_max_testcases if requested_max_testcases is not None else (
        DEFAULT_EGAR_DRC_CONFIG.max_testcases
    )
    done = 0
    passed = 0
    failed = 0
    errors = 0

    async def run_one(problem_dir: Path) -> dict[str, Any]:
        """Generate, score, and record one problem outcome."""
        nonlocal done, passed, failed, errors
        async with semaphore:
            started_at = time.perf_counter()
            logger.info("START problem=%s", problem_dir.name)
            problem_output = output_dir / problem_dir.name
            run_id: str | None = None
            record: dict[str, Any] = {
                "schema_version": 1,
                "id": problem_dir.name,
                "success": False,
            }
            try:
                problem_output.mkdir(parents=True, exist_ok=True)
                state = build_flow_input(
                    problem_dir,
                    repository,
                    max_testcases=effective_max_testcases,
                    klayout_bin=klayout_bin,
                    testcase_suite_cache_root=problem_output / "testcase_suite_cache",
                )
                run_id = uuid4().hex
                state["run_id"] = run_id
                result = await flow.invoke(state)
                logger.info(
                    "FLOW_DONE problem=%s status=%s seconds=%.3f",
                    problem_dir.name, sanitize_detail(result.get("flow_status", ""), model_config.api_key),
                    time.perf_counter() - started_at,
                )
                _log_agent_failures(
                    logger, problem_dir.name, result.get("llm_request_metrics", []), model_config.api_key,
                )
                record.update({
                    "flow_status": result.get("flow_status", ""),
                    "token_usage": result.get("token_usage_total", {}),
                    "llm_request_metrics": result.get("llm_request_metrics", []),
                    "loop_metrics": result.get("loop_metrics", {}),
                    "final_selection": result.get("final_selection", {}),
                })
                # Preserve generation evidence before scoring.
                (problem_output / "flow_result.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                deck = str(result.get("final_code", "")).strip()
                deck_path = problem_output / f"{problem_dir.name}.drc"
                deck_path.write_text(deck + ("\n" if deck else ""), encoding="utf-8")
                record["deck"] = str(deck_path)
                _append_jsonl(problem_output / "flow_record.jsonl", record)
                # Keep external scoring off the event loop.
                score = await asyncio.to_thread(
                    evaluate_deck,
                    problem_dir,
                    deck_path,
                    repository,
                    problem_output / "evaluation",
                    klayout_bin,
                )
                record.update({
                    "success": bool(score.get("success", False)),
                    "score": score,
                })
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                logger.error(
                    "PROBLEM_ERROR problem=%s type=%s detail=%s", problem_dir.name,
                    type(exc).__name__, sanitize_detail(exc, model_config.api_key),
                )
            finally:
                if run_id is not None:
                    await runtime.release_run(run_id)
            record["duration_seconds"] = round(time.perf_counter() - started_at, 6)
            async with write_lock:
                _append_jsonl(result_path, record)
                done += 1
                status = "ERROR" if "error" in record else "PASS" if record["success"] else "FAIL"
                passed += status == "PASS"
                failed += status == "FAIL"
                errors += status == "ERROR"
                logger.info(
                    "RESULT problem=%s done=%d/%d status=%s seconds=%.3f tokens=%d passed=%d",
                    problem_dir.name, done, len(pending), status, record["duration_seconds"],
                    record.get("token_usage", {}).get("total_tokens", 0) or 0, passed,
                )
            return record

    workers: list[asyncio.Task[dict[str, Any]]] = []

    async def settle_and_close() -> None:
        """Finish worker cleanup before closing shared resources."""
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await runtime.close()

    try:
        flow_name = getattr(args, "flow", "egar-drc")
        flow = FLOW_TYPES[flow_name](runtime)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = await asyncio.to_thread(
            build_run_metadata, model_config, problems, klayout_bin,
            flow_name, args.jobs, effective_max_testcases,
        )
        public_config = metadata["model_config"]
        logger.info(
            "RUN_START flow=%s model=%s endpoint=%s selected=%d pending=%d resumed=%d jobs=%d provider_limit=%d",
            flow_name, sanitize_detail(public_config["model"], model_config.api_key),
            sanitize_detail(public_config["base_url"], model_config.api_key), len(problems),
            len(pending), len(problems) - len(pending), args.jobs, model_config.max_concurrency,
        )
        # Refresh invocation metadata on resume.
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
        )
        workers.extend(asyncio.create_task(run_one(problem)) for problem in pending)
        outcomes = await asyncio.shield(asyncio.gather(*workers, return_exceptions=True))
        records: list[dict[str, Any]] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
            records.append(outcome)
        all_records = _read_records(result_path)
        elapsed_seconds = round(time.perf_counter() - run_started, 6)
        summary = build_summary(all_records, elapsed_seconds=elapsed_seconds)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        (output_dir / "eval_results.json").write_text(
            json.dumps({**summary, "results": all_records}, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "RUN_DONE completed=%d passed=%d failed=%d errors=%d pass_rate=%s seconds=%.3f total_completed=%d total_passed=%d",
            done, passed, failed, errors, f"{passed / done:.3f}" if done else "n/a",
            elapsed_seconds, summary["completed"], summary["successful"],
        )
        return records
    finally:
        # Shield cleanup from repeated cancellation.
        cleanup = asyncio.create_task(settle_and_close())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                cancellation = exc
        cleanup.result()
        if cancellation is not None:
            raise cancellation


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read the latest valid result per ID in sorted order."""
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and record.get("id"):
            records[str(record["id"])] = record
    return [records[key] for key in sorted(records)]


def build_summary(
    records: list[dict[str, Any]], *, elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate outcomes and problem durations with optional invocation timing.

    Args:
        records: Recorded outcomes, including historical records on resume.
        elapsed_seconds: Current invocation elapsed time through record collection,
            before summary persistence and shared runtime cleanup. None means
            unavailable; elapsed time cannot be inferred from problem durations.

    Returns:
        Historical record aggregates and current invocation wall time, if known.
    """
    completed = len(records)
    successful = sum(bool(record.get("success", False)) for record in records)
    durations = [float(record.get("duration_seconds", 0.0) or 0.0) for record in records]
    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    token_usage = {
        key: sum(int(record.get("token_usage", {}).get(key, 0) or 0) for record in records) for key in token_keys
    }
    return {
        "schema_version": 1,
        "completed": completed,
        "successful": successful,
        "success_rate": successful / completed if completed else None,
        "wall_time_seconds_total": round(elapsed_seconds, 6) if elapsed_seconds is not None else None,
        "problem_duration_seconds_total": round(sum(durations), 6),
        "problem_duration_seconds_mean": round(sum(durations) / completed, 6) if completed else None,
        "token_usage": token_usage,
        "tokens_per_success": token_usage["total_tokens"] / successful if successful else None,
    }


def _parser() -> argparse.ArgumentParser:
    """Expose worker, selection, resume, and scoring settings for one invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--flow", choices=tuple(FLOW_TYPES), default="egar-drc")
    parser.add_argument("--repository", default=None)
    parser.add_argument("--problem", action="append", default=[])
    parser.add_argument("--output-dir", default="results/run")
    parser.add_argument("--jobs", type=int, default=8, choices=range(1, 201), metavar="[1-200]")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-testcases", type=int, default=None)
    parser.add_argument("--klayout-bin", default="klayout")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main() -> None:
    """Print separate counts for new invocation records and persisted totals."""
    args = _parser().parse_args()
    records = asyncio.run(run_experiment(args))
    successes = sum(bool(record.get("success")) for record in records)
    total = build_summary(_read_records(Path(args.output_dir).expanduser().resolve() / "results.jsonl"))
    print(json.dumps({
        "invocation": {"completed": len(records), "successful": successes},
        "total": {"completed": total["completed"], "successful": total["successful"]},
    }, indent=2))


if __name__ == "__main__":
    main()
