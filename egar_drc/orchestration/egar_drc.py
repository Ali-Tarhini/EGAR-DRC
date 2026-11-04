"""Explicit graph orchestrator for EGAR-DRC."""

import asyncio
import json
import operator
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, cast
from uuid import uuid4

from egar_drc.agents import (
    CodeCorrectionAgent,
    CodeGenerationAgent,
    TestcaseAdjudicationAgent,
    TestcaseCorrectionAgent,
    TestcaseGenerationAgent,
)
from egar_drc.evaluation.generated_testcases import (
    describe_generated_testcases,
    generated_testcase_directory,
    prepare_cached_testcase_suite,
)
from egar_drc.evaluation.klayout_scoring import (
    compile_check_deck_without_testcases,
    evaluate_deck,
)
from egar_drc.flow_config import DEFAULT_EGAR_DRC_CONFIG
from egar_drc.models.runtime import ModelRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


# State contracts and reducers.
def _append_unique(left: list[Any], right: list[Any]) -> list[Any]:
    """Merge history snapshots without duplicating records."""
    merged = list(left)
    for record in right:
        if record not in merged:
            merged.append(record)
    return merged


def _merge_dict(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Merge partial dictionary updates, preferring new values for shared keys."""
    return {**dict(left), **dict(right)}


def _sum_token_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Add token-usage deltas emitted by concurrent agent nodes."""
    keys = set(left) | set(right)
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in keys}


def _take_right(_left: Any, right: Any) -> Any:
    """Resolve incidental scalar branch updates by keeping the newest value."""
    return right


class EgarDrcState(TypedDict, total=False):
    """State contract for one EGAR-DRC invocation."""

    run_id: str
    role_generations: Annotated[dict[str, int], _merge_dict]
    problem_id: str
    repository_path: str
    problem_dir: str
    klayout_bin: str
    testcase_suite_cache_root: str
    title: str
    rule_description: str
    rules: list[dict[str, Any]]
    categories: list[str]
    layers: dict[str, Any]
    thresholds: dict[str, Any]
    spec_yaml: str
    layer_context: str
    max_testcases: int
    testcase_generator: str
    generated_code: str
    final_code: str
    compilation_report: dict[str, Any]
    testcase_preparation_report: dict[str, Any]
    test_execution_report: dict[str, Any]
    critic_review: dict[str, Any]
    pending_code_correction: bool
    pending_testcase_correction: bool
    testcase_update_instruction: str
    testcase_feedback: dict[str, Any]
    testcase_correction_failure: dict[str, Any]
    code_correction_reason: str
    code_correction_compile_recovery: dict[str, Any] | None
    code_correction_memory_reset_pending: bool
    code_origin: str
    testcase_origin: str
    share_message: Annotated[dict[str, Any], _merge_dict]
    flow_status: Annotated[str, _take_right]
    next_node: str
    testcase_generation_iteration: int
    testcase_correction_iteration: int
    testcase_generation_prepare_iteration: int
    testcase_correction_prepare_iteration: int
    max_testcase_generation_prepare_iterations: int
    max_testcase_correction_prepare_iterations: int
    max_testcase_correction_iterations: int
    code_generation_iteration: int
    code_correction_iteration: int
    code_revision_iteration: int
    max_code_correction_iterations: int
    max_code_generation_compile_iterations: int
    max_code_generation_retry_iterations: int
    max_code_correction_compile_iterations: int
    code_generation_compile_iteration: int
    code_generation_failure_iteration: int
    code_correction_compile_iteration: int
    code_regeneration_iteration: int
    code_regeneration_pending: bool
    compile_iteration: int
    critic_review_iteration: int
    max_critic_review_iterations: int
    compilation_succeeded: bool
    testcase_generation_succeeded: bool
    critic_request_succeeded: bool
    generated_testcases: list[dict[str, Any]]
    active_generated_testcases: list[dict[str, Any]]
    loop_metrics: Annotated[dict[str, Any], _merge_dict]
    llm_request_metrics: Annotated[list[dict[str, Any]], _append_unique]
    testcase_adjudication_metrics: Annotated[dict[str, Any], _merge_dict]
    token_usage_total: Annotated[dict[str, int], _sum_token_usage]
    last_token_usage: dict[str, int]
    flow_events: Annotated[list[dict[str, Any]], _append_unique]
    code_candidates: Annotated[list[dict[str, Any]], _append_unique]
    testcase_script_versions: Annotated[list[dict[str, Any]], _append_unique]
    testcase_suite_versions: Annotated[list[dict[str, Any]], _append_unique]
    execution_attempts: Annotated[list[dict[str, Any]], _append_unique]
    best_code_candidate: dict[str, Any]
    final_selection: dict[str, Any]


def _event(
    state: EgarDrcState,
    event_phase: str,
    event_status: str,
    **payload: Any,
) -> list[dict[str, Any]]:
    """Append one serializable flow event."""
    events = [dict(record) for record in state.get("flow_events", [])]
    event = {
        **payload,
        "event_index": len(events) + 1,
        "phase": event_phase,
        "status": event_status,
    }
    events.append(event)
    return events


def _loop_metric(
    state: EgarDrcState,
    *,
    phase: str,
    started_at: float,
    success: bool,
    outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one deterministic or agent-node attempt metric."""
    metrics = dict(state.get("loop_metrics", {}))
    phase_metrics = dict(metrics.get(phase, {}))
    durations = [float(value) for value in phase_metrics.get("durations_seconds", [])]
    duration = round(time.perf_counter() - started_at, 6)
    durations.append(duration)
    attempts = int(phase_metrics.get("attempts", 0)) + 1
    successes = int(phase_metrics.get("successes", 0)) + int(success)
    attempt_records = [dict(record) for record in phase_metrics.get("attempt_records", [])]
    attempt_records.append(
        {
            "attempt": attempts,
            "success": bool(success),
            "duration_seconds": duration,
            **dict(outcome or {}),
        }
    )
    phase_metrics.update(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "durations_seconds": durations,
            "total_duration_seconds": round(sum(durations), 6),
            "attempt_records": attempt_records,
        }
    )
    metrics[phase] = phase_metrics
    return metrics


def _case_pass_count(report: dict[str, Any]) -> int:
    """Count exactly matching generated testcase results."""
    case_results = report.get("case_results", {})
    if not isinstance(case_results, dict):
        return 0
    passing = 0
    for case_result in case_results.values():
        if not isinstance(case_result, dict):
            continue
        if isinstance(case_result.get("expected"), dict) and case_result.get("expected") == case_result.get("observed"):
            passing += 1
    return passing


def _execution_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact execution summary fields."""
    n_cases = int(report.get("n_cases", 0) or 0)
    n_passing = int(report.get("n_passing_cases", _case_pass_count(report)) or 0)
    per_category_f1 = report.get("per_category_f1", {})
    per_category_f1 = (
        {str(category): float(score) for category, score in per_category_f1.items()}
        if isinstance(per_category_f1, dict)
        else {}
    )
    return {
        "success": bool(report.get("success", False)),
        "phase": str(report.get("phase", "")),
        "n_cases": n_cases,
        "n_passing_cases": n_passing,
        "n_failing_cases": max(n_cases - n_passing, 0),
        "testcase_success_rate": n_passing / n_cases if n_cases else None,
        "per_category_f1": per_category_f1,
        "macro_f1": sum(per_category_f1.values()) / len(per_category_f1) if per_category_f1 else None,
        "error": str(report.get("error", "")),
    }


class EgarDrcFlow:
    """Refine candidate decks and testcases using execution evidence."""

    assistant_id = "egar_drc"

    def __init__(self, runtime: ModelRuntime) -> None:
        """Initialize agent node instances."""
        self.runtime = runtime
        self._testcase_generation_agent = TestcaseGenerationAgent(runtime)
        self._code_generation_agent = CodeGenerationAgent(runtime)
        self._testcase_adjudication_agent = TestcaseAdjudicationAgent(runtime)
        self._code_correction_agent = CodeCorrectionAgent(runtime)
        self._testcase_correction_agent = TestcaseCorrectionAgent(runtime)
        self._graph = self.make_graph()

    async def invoke(self, state: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        """Await the graph with budget-derived limits, unlike sync Runnable.invoke."""
        state = {**asdict(DEFAULT_EGAR_DRC_CONFIG), **state, "run_id": state.get("run_id") or uuid4().hex}
        testcase_budget = int(state.get("max_testcase_generation_prepare_iterations", 5))
        testcase_correction_prepare_budget = int(
            state.get("max_testcase_correction_prepare_iterations", testcase_budget)
        )
        testcase_correction_budget = int(state.get("max_testcase_correction_iterations", 5))
        generation_compile_budget = int(state.get("max_code_generation_compile_iterations", 15))
        correction_budget = int(state.get("max_code_correction_iterations", 5))
        correction_compile_budget = int(state.get("max_code_correction_compile_iterations", 15))
        regeneration_budget = int(state.get("max_code_generation_retry_iterations", 5))
        critic_budget = int(state.get("max_critic_review_iterations", 3))
        # Allow enough LangGraph steps for the configured retries.
        required_limit = (
            20
            + testcase_budget * 3
            + testcase_correction_budget * testcase_correction_prepare_budget * 4
            + generation_compile_budget * (regeneration_budget + 1) * 3
            + correction_budget * correction_compile_budget * 4
            + critic_budget * 8
        )
        graph_config = dict(config or {})
        # Preserve higher caller limits.
        graph_config["recursion_limit"] = max(int(str(graph_config.get("recursion_limit", 0))), required_limit)
        return cast(dict[str, Any], await self._graph.ainvoke(state, config=cast(RunnableConfig, graph_config)))

    # Graph construction.
    def make_graph(self) -> Any:
        """Build the explicit EGAR-DRC graph."""
        builder: StateGraph = StateGraph(EgarDrcState)
        builder.add_node("initialize", cast(Any, self.initialize))
        builder.add_node("generate_code", cast(Any, self._code_generation_agent.invoke))
        builder.add_node("generate_testcases", cast(Any, self._testcase_generation_agent.invoke))
        builder.add_node("compile_code", cast(Any, self.compile_code))
        builder.add_node("prepare_testcases", cast(Any, self.prepare_testcases))
        builder.add_node("restore_testcases", cast(Any, self.restore_testcases))
        builder.add_node("execute_testcases", cast(Any, self.execute_testcases))
        builder.add_node("adjudicate_testcases", cast(Any, self._testcase_adjudication_agent.invoke))
        builder.add_node("correct_code", cast(Any, self._code_correction_agent.invoke))
        builder.add_node("correct_testcases", cast(Any, self._testcase_correction_agent.invoke))
        builder.add_node("finalize", cast(Any, self.finalize))

        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate_code")
        builder.add_edge("generate_code", "compile_code")
        builder.add_conditional_edges(
            "compile_code",
            self.route_after_compile,
            {
                "generate_code": "generate_code",
                "correct_code": "correct_code",
                "correct_testcases": "correct_testcases",
                "generate_testcases": "generate_testcases",
                "adjudicate_testcases": "adjudicate_testcases",
                "execute_testcases": "execute_testcases",
                "finalize": "finalize",
            },
        )
        builder.add_edge("generate_testcases", "prepare_testcases")
        builder.add_conditional_edges(
            "prepare_testcases",
            self.route_after_prepare,
            {
                "generate_testcases": "generate_testcases",
                "correct_testcases": "correct_testcases",
                "restore_testcases": "restore_testcases",
                "execute_testcases": "execute_testcases",
                "adjudicate_testcases": "adjudicate_testcases",
            },
        )
        builder.add_edge("restore_testcases", "execute_testcases")
        builder.add_edge("execute_testcases", "adjudicate_testcases")
        builder.add_conditional_edges(
            "adjudicate_testcases",
            self.route_after_critic,
            {
                "correct_code": "correct_code",
                "correct_testcases": "correct_testcases",
                "execute_testcases": "execute_testcases",
                "adjudicate_testcases": "adjudicate_testcases",
                "finalize": "finalize",
            },
        )
        builder.add_edge("correct_code", "compile_code")
        builder.add_edge("correct_testcases", "prepare_testcases")
        builder.add_edge("finalize", END)
        return builder.compile()

    # Deterministic nodes.
    async def initialize(self, state: EgarDrcState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Initialize default flow state before parallel branches start."""
        del config
        initialized = self._with_defaults(dict(state))
        initialized["flow_status"] = "initialized"
        initialized["flow_events"] = _event(initialized, "flow", "started")
        return cast(dict[str, Any], initialized)

    async def compile_code(self, state: EgarDrcState) -> dict[str, Any]:
        """Compile the current Ruby deck and record a code candidate."""
        started_at = time.perf_counter()
        final_code = str(state.get("generated_code", state.get("final_code", ""))).strip()
        klayout_bin = str(state.get("klayout_bin", "klayout")).strip() or "klayout"
        if not final_code:
            report = {"ok": False, "success": False, "phase": "compilation", "error": "Empty Ruby deck."}
        else:
            with tempfile.TemporaryDirectory(prefix="egar_drc_compile_") as temp_dir:
                deck_path = Path(temp_dir) / "candidate.drc"
                deck_path.write_text(final_code + "\n", encoding="utf-8")
                report = await asyncio.to_thread(compile_check_deck_without_testcases, deck_path, klayout_bin)
                report = {**report, "success": bool(report.get("ok", False)), "phase": "compilation"}

        origin = str(state.get("code_origin", "initial_code_generation")) or "initial_code_generation"
        succeeded = bool(report.get("success", False))
        compile_iteration = int(state.get("compile_iteration", 0)) + 1
        code_generation_compile_iteration = int(state.get("code_generation_compile_iteration", 0))
        code_correction_compile_iteration = int(state.get("code_correction_compile_iteration", 0))
        if origin == "code_correction":
            code_correction_compile_iteration += 1
        else:
            code_generation_compile_iteration += 1
        code_regeneration_iteration = int(state.get("code_regeneration_iteration", 0))
        code_regeneration_pending = False
        flow_status = "compiled" if succeeded else "compile_failed"
        # Keep generation and correction compile windows separate.
        if not succeeded:
            if origin == "code_correction" and code_correction_compile_iteration >= int(
                state.get("max_code_correction_compile_iterations", 15)
            ):
                flow_status = "code_correction_compile_budget_exhausted"
            elif origin != "code_correction" and code_generation_compile_iteration >= int(
                state.get("max_code_generation_compile_iterations", 15)
            ):
                code_regeneration_iteration += 1
                code_generation_compile_iteration = 0
                code_regeneration_pending = code_regeneration_iteration <= int(
                    state.get("max_code_generation_retry_iterations", 5)
                )
                flow_status = (
                    "initial_code_regeneration_pending"
                    if code_regeneration_pending
                    else "initial_code_generation_failed"
                )

        recovery: dict[str, Any] | None = None
        if flow_status == "code_correction_compile_budget_exhausted":
            # Preserve the latest compiling deck for recovery.
            previous_compiling_candidate = next(
                (
                    dict(candidate)
                    for candidate in reversed(state.get("code_candidates", []))
                    if isinstance(candidate, dict) and bool(candidate.get("compiled", False))
                ),
                dict(state.get("best_code_candidate", {})),
            )
            recovery = {
                "previous_compiling_code": str(previous_compiling_candidate.get("code", "")),
                "previous_compilation_report": previous_compiling_candidate.get("compilation_report", {}),
                "failed_corrected_code": final_code,
                "compilation_report": report,
            }

        update: EgarDrcState = {
            "final_code": final_code,
            "generated_code": final_code,
            "compilation_report": report,
            "compilation_succeeded": succeeded,
            "compile_iteration": compile_iteration,
            "code_generation_compile_iteration": code_generation_compile_iteration,
            "code_generation_failure_iteration": int(state.get("code_generation_failure_iteration", 0))
            + int(flow_status == "initial_code_generation_failed"),
            "code_correction_compile_iteration": code_correction_compile_iteration,
            "code_regeneration_iteration": code_regeneration_iteration,
            "code_regeneration_pending": code_regeneration_pending,
            "code_correction_compile_recovery": recovery,
            "share_message": {"compilation": report},
            "flow_status": flow_status,
            "loop_metrics": _loop_metric(
                state,
                phase="compilation",
                started_at=started_at,
                success=succeeded,
                outcome={"origin": origin, "error": str(report.get("error", ""))},
            ),
            "flow_events": _event(state, "compilation", "succeeded" if succeeded else "failed", origin=origin),
        }
        update.update(self._record_code_candidate(cast(EgarDrcState, {**state, **update}), origin=origin))
        return cast(dict[str, Any], update)

    async def prepare_testcases(self, state: EgarDrcState) -> dict[str, Any]:
        """Prepare the current full testcase script and record its version."""
        started_at = time.perf_counter()
        script = str(state.get("testcase_generator", "")).strip()
        problem_dir = Path(str(state.get("problem_dir", ""))).resolve()
        cache_root_value = str(state.get("testcase_suite_cache_root", "")).strip()
        generated_testcases: list[dict[str, Any]] = []
        active_generated_testcases: list[dict[str, Any]] = []
        origin = str(state.get("testcase_origin", "initial_testcases")) or "initial_testcases"
        if not script or not problem_dir.is_dir():
            report = {
                "success": False,
                "phase": "testcase_generation",
                "error": "Missing problem directory or testcase generator script.",
            }
        else:
            with tempfile.TemporaryDirectory(prefix="egar_drc_prepare_testcases_") as temp_dir:
                cache_root = Path(cache_root_value).resolve() if cache_root_value else Path(temp_dir) / "suite_cache"
                try:
                    suite_problem_dir = await asyncio.to_thread(
                        prepare_cached_testcase_suite,
                        source_problem_dir=problem_dir,
                        cache_root=cache_root,
                        generator_scripts=[script],
                        categories=list(state.get("categories", [])),
                        max_cases=max(int(state.get("max_testcases", 8)), 1) if origin == "initial_testcases" else None,
                        excluded_filenames=[],
                    )
                    suite_root = suite_problem_dir.parents[1]
                    generated_dir = generated_testcase_directory(
                        suite_repository_root=suite_root,
                        source_problem_dir=problem_dir,
                    )
                    inventory = json.loads((generated_dir / "inventory.json").read_text(encoding="utf-8"))
                    generated_testcases = inventory if isinstance(inventory, list) else []
                    active_generated_testcases = describe_generated_testcases(
                        generated_dir, list(state.get("categories", []))
                    )
                    report = {
                        "success": True,
                        "phase": "testcase_generation",
                        "n_cases": len(active_generated_testcases),
                    }
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    report = {
                        "success": False,
                        "phase": "testcase_generation",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        succeeded = bool(report.get("success", False))
        testcase_generation_prepare_iteration = int(state.get("testcase_generation_prepare_iteration", 0))
        testcase_correction_prepare_iteration = int(state.get("testcase_correction_prepare_iteration", 0))
        if origin == "testcase_correction":
            testcase_correction_prepare_iteration += 1
        else:
            testcase_generation_prepare_iteration += 1
        update: EgarDrcState = {
            "testcase_generator": script,
            "testcase_generation_succeeded": succeeded,
            "testcase_generation_prepare_iteration": testcase_generation_prepare_iteration,
            "testcase_correction_prepare_iteration": testcase_correction_prepare_iteration,
            "generated_testcases": generated_testcases,
            "active_generated_testcases": active_generated_testcases,
            "testcase_preparation_report": report,
            "share_message": {"testcase_generation": report},
            "flow_status": "testcase_generator_ready" if succeeded else "testcase_generation_failed",
            "loop_metrics": _loop_metric(
                state,
                phase="testcase_preparation",
                started_at=started_at,
                success=succeeded,
                outcome={
                    "origin": origin,
                    "n_cases": int(report.get("n_cases", 0) or 0),
                    "error": str(report.get("error", "")),
                },
            ),
            "flow_events": _event(state, "testcase_preparation", "succeeded" if succeeded else "failed", origin=origin),
            "testcase_script_versions": [dict(record) for record in state.get("testcase_script_versions", [])],
            "testcase_suite_versions": [dict(record) for record in state.get("testcase_suite_versions", [])],
        }
        self._record_testcase_script_version(cast(EgarDrcState, update), origin=origin)
        self._record_testcase_suite_version(cast(EgarDrcState, update), origin=origin, report=report)
        return cast(dict[str, Any], update)

    async def restore_testcases(self, state: EgarDrcState) -> dict[str, Any]:
        """Restore the latest prepared testcase script after correction failure."""
        del self
        prepared_scripts = [
            dict(record)
            for record in state.get("testcase_script_versions", [])
            if bool(record.get("prepared", False)) and str(record.get("script", "")).strip()
        ]
        prepared_suites = [
            dict(record) for record in state.get("testcase_suite_versions", []) if bool(record.get("prepared", False))
        ]
        if not prepared_scripts:
            return {
                "flow_status": "testcase_correction_failed_without_fallback",
                "flow_events": _event(state, "testcase_fallback", "unavailable"),
            }
        # Restore the last successfully prepared suite.
        script_record = prepared_scripts[-1]
        suite_record = prepared_suites[-1] if prepared_suites else {}
        failure_report = dict(state.get("testcase_preparation_report", {}))
        return {
            "testcase_generator": str(script_record.get("script", "")),
            "testcase_origin": "testcase_fallback",
            "testcase_generation_succeeded": True,
            "generated_testcases": list(suite_record.get("generated_testcases", [])),
            "active_generated_testcases": list(suite_record.get("active_generated_testcases", [])),
            "share_message": {
                "testcase_correction_failure": {
                    "error": str(failure_report.get("error", "")),
                    "failed_report": failure_report,
                    "restored_script_version": int(script_record.get("version", 0) or 0),
                }
            },
            "testcase_correction_failure": {
                "error": str(failure_report.get("error", "")),
                "failed_report": failure_report,
                "restored_script_version": int(script_record.get("version", 0) or 0),
            },
            "flow_status": "testcase_fallback_restored",
            "flow_events": _event(
                state,
                "testcase_fallback",
                "restored",
                script_version=int(script_record.get("version", 0) or 0),
            ),
        }

    async def execute_testcases(self, state: EgarDrcState) -> dict[str, Any]:
        """Execute the compiled deck against the current generated testcase suite."""
        started_at = time.perf_counter()
        final_code = str(state.get("final_code", state.get("generated_code", ""))).strip()
        script = str(state.get("testcase_generator", "")).strip()
        klayout_bin = str(state.get("klayout_bin", "klayout")).strip() or "klayout"
        problem_dir = Path(str(state.get("problem_dir", ""))).resolve()
        repository_path = Path(str(state.get("repository_path", ""))).resolve()
        cache_root_value = str(state.get("testcase_suite_cache_root", "")).strip()
        generated_testcases: list[dict[str, Any]] = []
        active_generated_testcases: list[dict[str, Any]] = []
        if not final_code or not script or not problem_dir.is_dir() or not repository_path.is_dir():
            report = {
                "success": False,
                "phase": "deck_evaluation",
                "error": "Missing deck, testcase generator, problem directory, or repository path.",
            }
        else:
            with tempfile.TemporaryDirectory(prefix="egar_drc_execute_testcases_") as temp_dir:
                cache_root = Path(cache_root_value).resolve() if cache_root_value else Path(temp_dir) / "suite_cache"
                try:
                    suite_problem_dir = await asyncio.to_thread(
                        prepare_cached_testcase_suite,
                        source_problem_dir=problem_dir,
                        cache_root=cache_root,
                        generator_scripts=[script],
                        categories=list(state.get("categories", [])),
                        max_cases=(
                            max(int(state.get("max_testcases", 8)), 1)
                            if str(state.get("testcase_origin", "initial_testcases")) == "initial_testcases"
                            else None
                        ),
                        excluded_filenames=[],
                    )
                    suite_root = suite_problem_dir.parents[1]
                    generated_dir = generated_testcase_directory(
                        suite_repository_root=suite_root,
                        source_problem_dir=problem_dir,
                    )
                    inventory = json.loads((generated_dir / "inventory.json").read_text(encoding="utf-8"))
                    generated_testcases = inventory if isinstance(inventory, list) else []
                    active_generated_testcases = describe_generated_testcases(
                        generated_dir, list(state.get("categories", []))
                    )
                    deck_path = Path(temp_dir) / "candidate.drc"
                    deck_path.write_text(final_code + "\n", encoding="utf-8")
                    report = await asyncio.to_thread(
                        evaluate_deck,
                        suite_problem_dir,
                        deck_path,
                        suite_root,
                        Path(temp_dir) / "evaluation",
                        klayout_bin,
                    )
                    report = {**report, "phase": "deck_evaluation"}
                except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    report = {
                        "success": False,
                        "phase": "deck_evaluation",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        n_passing = _case_pass_count(report)
        n_cases = int(report.get("n_cases", 0) or len(active_generated_testcases))
        report = {
            **report,
            "success": bool(report.get("success", False)),
            "n_cases": n_cases,
            "n_passing_cases": n_passing,
            "n_failing_cases": max(n_cases - n_passing, 0),
            "generated_testcases": generated_testcases,
            "active_generated_testcases": active_generated_testcases,
        }
        attempts = [dict(record) for record in state.get("execution_attempts", [])]
        attempts.append(
            {
                "execution_index": len(attempts) + 1,
                "candidate_id": self._current_candidate_id(state),
                "testcase_script_version": len(state.get("testcase_script_versions", [])),
                **_execution_summary(report),
            }
        )
        update: EgarDrcState = {
            "test_execution_report": report,
            "generated_testcases": generated_testcases,
            "active_generated_testcases": active_generated_testcases,
            "execution_attempts": attempts,
            "share_message": {
                "test_execution": {"test_execution_report": report, "generated_testcases": generated_testcases}
            },
            "flow_status": "test_execution_complete",
            "loop_metrics": _loop_metric(
                state,
                phase="test_execution",
                started_at=started_at,
                success=bool(report.get("success", False)),
                outcome=_execution_summary(report),
            ),
            "flow_events": _event(state, "test_execution", "completed", **_execution_summary(report)),
        }
        update.update(self._update_best_candidate_from_execution(state, report))
        return cast(dict[str, Any], update)

    async def finalize(self, state: EgarDrcState) -> dict[str, Any]:
        """Select the final candidate and close the flow."""
        status = str(state.get("flow_status", "finished"))
        terminal_statuses = {
            "accepted",
            "critic_inconclusive",
            "initial_code_generation_failed",
            "critic_budget_exhausted",
            "code_correction_budget_exhausted",
            "code_correction_compile_budget_exhausted",
            "testcase_correction_budget_exhausted",
        }
        if status not in terminal_statuses:
            has_compiled_candidate = any(
                isinstance(candidate, dict) and bool(candidate.get("compiled", False))
                for candidate in state.get("code_candidates", [])
            )
            if not bool(state.get("compilation_succeeded", False)) and not has_compiled_candidate:
                status = "initial_code_generation_failed"
            else:
                review = state.get("critic_review", {})
                code_instruction = (
                    str(review.get("code_correction_instruction", "")).strip() if isinstance(review, dict) else ""
                )
                testcase_instruction = (
                    str(review.get("testcase_script_update_instruction", "")).strip()
                    if isinstance(review, dict)
                    else ""
                )
                if code_instruction and int(state.get("code_correction_iteration", 0)) >= int(
                    state.get("max_code_correction_iterations", 5)
                ):
                    status = "code_correction_budget_exhausted"
                elif testcase_instruction and int(state.get("testcase_correction_iteration", 0)) >= int(
                    state.get("max_testcase_correction_iterations", 5)
                ):
                    status = "testcase_correction_budget_exhausted"
                elif int(state.get("critic_review_iteration", 0)) >= int(state.get("max_critic_review_iterations", 3)):
                    status = "critic_budget_exhausted"
                else:
                    status = "critic_budget_exhausted"
        best = dict(state.get("best_code_candidate", {}))
        selected_code = str(state.get("final_code", state.get("generated_code", "")))
        # Use the historical best when the active deck is not accepted.
        if status != "accepted" and best.get("code"):
            selected_code = str(best["code"])
        selected_id = (
            self._current_candidate_id(state)
            if status == "accepted"
            else str(best.get("candidate_id", self._current_candidate_id(state)))
        )
        selected_generated_testcase_metrics = (
            _execution_summary(state.get("test_execution_report", {}))
            if status == "accepted"
            else best.get("execution_report", {})
        )
        selected_generated_testcase_metrics = (
            dict(selected_generated_testcase_metrics) if isinstance(selected_generated_testcase_metrics, dict) else {}
        )
        result = {
            "final_code": selected_code,
            "generated_code": selected_code,
            "flow_status": status,
            "selected_generated_testcase_metrics": selected_generated_testcase_metrics,
            "final_selection": {
                "status": status,
                "selected_candidate_id": selected_id,
                "selected_reason": (
                    "accepted_current_candidate" if status == "accepted" else "best_generated_testcase_score"
                ),
                "best_code_candidate": best,
            },
            "flow_events": _event(state, "flow", "finished", final_status=status, selected_candidate_id=selected_id),
        }
        return result

    # Budget-aware routing.
    @staticmethod
    def route_after_compile(state: EgarDrcState) -> Literal[
        "generate_code",
        "correct_code",
        "correct_testcases",
        "generate_testcases",
        "adjudicate_testcases",
        "execute_testcases",
        "finalize",
    ]:
        """Route after deterministic compilation."""
        phase = str(state.get("code_origin", "initial_code_generation"))
        if bool(state.get("compilation_succeeded", False)):
            if phase == "code_correction":
                if bool(state.get("pending_testcase_correction", False)):
                    return "correct_testcases"
                return (
                    "execute_testcases"
                    if bool(state.get("testcase_generation_succeeded", False))
                    else "adjudicate_testcases"
                )
            return "generate_testcases"
        if phase == "code_correction":
            # Return exhausted compile retries to adjudication.
            if int(state.get("code_correction_compile_iteration", 0)) < int(
                state.get("max_code_correction_compile_iterations", 15)
            ):
                return "correct_code"
            if int(state.get("critic_review_iteration", 0)) < int(state.get("max_critic_review_iterations", 3)):
                return "adjudicate_testcases"
            return "finalize"
        if int(state.get("code_regeneration_iteration", 0)) > int(state.get("max_code_generation_retry_iterations", 5)):
            return "finalize"
        if bool(state.get("code_regeneration_pending", False)):
            return "generate_code"
        if int(state.get("code_generation_compile_iteration", 0)) < int(
            state.get("max_code_generation_compile_iterations", 15)
        ):
            return "generate_code"
        return "generate_code"

    @staticmethod
    def route_after_prepare(state: EgarDrcState) -> Literal[
        "generate_testcases",
        "correct_testcases",
        "restore_testcases",
        "execute_testcases",
        "adjudicate_testcases",
    ]:
        """Route after deterministic testcase preparation."""
        phase = str(state.get("testcase_origin", "initial_testcases"))
        if bool(state.get("testcase_generation_succeeded", False)):
            if phase == "testcase_correction":
                return "execute_testcases" if bool(state.get("compilation_succeeded", False)) else "adjudicate_testcases"
            return "execute_testcases"
        if phase == "testcase_correction":
            # Restore a prepared suite when correction retries are exhausted.
            if int(state.get("testcase_correction_prepare_iteration", 0)) < int(
                state.get("max_testcase_correction_prepare_iterations", 5)
            ):
                return "correct_testcases"
            prepared_versions = state.get("testcase_script_versions", [])
            if any(bool(record.get("prepared", False)) for record in prepared_versions):
                return "restore_testcases"
            return "adjudicate_testcases"
        if int(state.get("testcase_generation_prepare_iteration", 0)) < int(
            state.get("max_testcase_generation_prepare_iterations", 5)
        ):
            return "generate_testcases"
        return "adjudicate_testcases"

    @staticmethod
    def route_after_critic(state: EgarDrcState) -> Literal[
        "correct_code",
        "correct_testcases",
        "execute_testcases",
        "adjudicate_testcases",
        "finalize",
    ]:
        """Route after critic instructions."""
        if not bool(state.get("critic_request_succeeded", True)):
            # Bound failed adjudication attempts by the review budget.
            if int(state.get("critic_review_iteration", 0)) < int(state.get("max_critic_review_iterations", 3)):
                return "adjudicate_testcases"
            return "finalize"
        review = state.get("critic_review", {})
        code_instruction = (
            str(review.get("code_correction_instruction", "")).strip() if isinstance(review, dict) else ""
        )
        testcase_instruction = (
            str(review.get("testcase_script_update_instruction", "")).strip() if isinstance(review, dict) else ""
        )
        if not code_instruction and not testcase_instruction:
            report = state.get("test_execution_report", {})
            n_cases = int(report.get("n_cases", 0) or 0) if isinstance(report, dict) else 0
            n_passing = int(report.get("n_passing_cases", 0) or 0) if isinstance(report, dict) else 0
            if n_cases > n_passing and int(state.get("critic_review_iteration", 0)) < int(
                state.get("max_critic_review_iterations", 3)
            ):
                return "adjudicate_testcases"
            return "finalize"
        if int(state.get("critic_review_iteration", 0)) >= int(state.get("max_critic_review_iterations", 3)):
            return "finalize"
        if code_instruction:
            # Keep correction counters with their owning agents.
            if int(state.get("code_correction_iteration", 0)) > int(state.get("max_code_correction_iterations", 5)):
                return "finalize"
            return "correct_code"
        if testcase_instruction:
            if int(state.get("testcase_correction_iteration", 0)) >= int(
                state.get("max_testcase_correction_iterations", 5)
            ):
                return "finalize"
            return "correct_testcases"
        return "execute_testcases"

    # State defaults and evidence history.
    @staticmethod
    def _with_defaults(state: dict[str, Any]) -> EgarDrcState:
        """Return state with all flow-owned defaults installed."""
        defaults: dict[str, Any] = {
            "testcase_generation_iteration": 0,
            "testcase_correction_iteration": 0,
            "testcase_generation_prepare_iteration": 0,
            "testcase_correction_prepare_iteration": 0,
            "code_generation_iteration": 0,
            "code_correction_iteration": 0,
            "code_revision_iteration": 0,
            "code_generation_compile_iteration": 0,
            "code_generation_failure_iteration": 0,
            "code_correction_compile_iteration": 0,
            "code_correction_compile_recovery": None,
            "code_correction_memory_reset_pending": False,
            "code_regeneration_iteration": 0,
            "code_regeneration_pending": False,
            "compile_iteration": 0,
            "critic_review_iteration": 0,
            "compilation_succeeded": False,
            "testcase_generation_succeeded": False,
            "critic_request_succeeded": True,
            "pending_code_correction": False,
            "pending_testcase_correction": False,
            "generated_testcases": [],
            "active_generated_testcases": [],
            "share_message": {},
            "flow_status": "initialized",
            "code_origin": "initial_code_generation",
            "testcase_origin": "initial_testcases",
            "loop_metrics": {},
            "llm_request_metrics": [],
            "testcase_adjudication_metrics": {},
            "testcase_preparation_report": {},
            "testcase_correction_failure": {},
            "flow_events": [],
            "code_candidates": [],
            "testcase_script_versions": [],
            "testcase_suite_versions": [],
            "execution_attempts": [],
            "best_code_candidate": {},
            "final_selection": {},
            "role_generations": {},
        }
        return cast(EgarDrcState, {**defaults, **asdict(DEFAULT_EGAR_DRC_CONFIG), **state})

    @staticmethod
    def _current_candidate_id(state: EgarDrcState) -> str:
        """Return the candidate matching the active deck, including recovery."""
        candidates = state.get("code_candidates", [])
        if not candidates:
            return ""
        code = state.get("final_code", state.get("generated_code"))
        if code is not None:
            # Match restored source to its latest recorded candidate.
            return next(
                (str(candidate.get("candidate_id", "")) for candidate in reversed(candidates)
                 if str(candidate.get("code", "")).strip() == str(code).strip()),
                "",
            )
        return str(candidates[-1].get("candidate_id", ""))

    @staticmethod
    def _record_code_candidate(state: EgarDrcState, *, origin: str) -> EgarDrcState:
        """Return a new candidate delta using accumulated revision history."""
        candidates = [dict(record) for record in state.get("code_candidates", [])]
        candidate = {
            "candidate_id": f"code_{len(candidates) + 1:04d}",
            "origin": origin,
            "code_generation_iteration": int(state.get("code_generation_iteration", 0)),
            "code_correction_iteration": int(state.get("code_correction_iteration", 0)),
            "compile_iteration": int(state.get("compile_iteration", 0)),
            "compiled": bool(state.get("compilation_succeeded", False)),
            "code": str(state.get("final_code", state.get("generated_code", ""))),
            "compilation_report": state.get("compilation_report", {}),
        }
        update: EgarDrcState = {"code_candidates": [candidate]}
        # A compiling deck seeds the fallback even before any suite is executable.
        if candidate["compiled"] and not state.get("best_code_candidate"):
            update["best_code_candidate"] = {**candidate, "n_cases": 0, "n_passing_cases": 0}
        return update

    @staticmethod
    def _record_testcase_script_version(state: EgarDrcState, *, origin: str) -> None:
        """Record one full testcase-generator script version."""
        versions = [dict(record) for record in state.get("testcase_script_versions", [])]
        versions.append(
            {
                "version": len(versions) + 1,
                "origin": origin,
                "prepared": bool(state.get("testcase_generation_succeeded", False)),
                "n_cases": len(state.get("active_generated_testcases", [])),
                "script": str(state.get("testcase_generator", "")),
            }
        )
        state["testcase_script_versions"] = versions

    @staticmethod
    def _record_testcase_suite_version(state: EgarDrcState, *, origin: str, report: dict[str, Any]) -> None:
        """Record deterministic testcase-suite preparation metadata."""
        versions = [dict(record) for record in state.get("testcase_suite_versions", [])]
        versions.append(
            {
                "version": len(versions) + 1,
                "origin": origin,
                "prepared": bool(report.get("success", False)),
                "n_cases": int(report.get("n_cases", 0) or 0),
                "report": report,
                "generated_testcases": state.get("generated_testcases", []),
                "active_generated_testcases": state.get("active_generated_testcases", []),
            }
        )
        state["testcase_suite_versions"] = versions

    @staticmethod
    def _update_best_candidate_from_execution(state: EgarDrcState, report: dict[str, Any]) -> EgarDrcState:
        """Retain the compiled candidate with the greatest observed pass count."""
        candidate_id = EgarDrcFlow._current_candidate_id(state)
        if not candidate_id:
            return {}
        n_cases = int(report.get("n_cases", 0) or 0)
        n_passing = int(report.get("n_passing_cases", _case_pass_count(report)) or 0)
        best = dict(state.get("best_code_candidate", {}))
        if n_cases > 0 and (not best.get("execution_report") or n_passing > int(best.get("n_passing_cases", -1))):
            matching = next(
                (
                    dict(candidate)
                    for candidate in state.get("code_candidates", [])
                    if candidate.get("candidate_id") == candidate_id
                ),
                {},
            )
            if not matching.get("compiled", False):
                return {}
            return {"best_code_candidate": {
                **matching,
                "n_cases": n_cases,
                "n_passing_cases": n_passing,
                "execution_report": _execution_summary(report),
            }}
        return {}
