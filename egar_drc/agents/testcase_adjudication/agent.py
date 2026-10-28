"""Joint adjudication of EGAR-DRC decks and generated testcase evidence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from egar_drc.agents.base import RoleAgent
from egar_drc.agents.code_generation.agent import load_klayout_docs
from egar_drc.agents.flow_node_helpers import (
    append_flow_event,
    append_llm_metric,
    append_loop_metric,
    graph_token_usage_update,
    has_passing_execution,
    owned_update,
)
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TestcaseAdjudicationOutput(BaseModel):
    """Structured output contract for joint code/testcase adjudication."""

    code_correction_instruction: str = Field(default="")
    testcase_script_update_instruction: str = Field(default="")


class TestcaseAdjudicationAgent(RoleAgent):
    """Adjudicate disagreements between generated DRC code and testcases."""

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Initialize the agent from its local descriptor."""
        descriptor_path = Path(__file__).with_name("testcase_adjudication_agent.yaml")
        super().__init__(runtime, descriptor_path, tools=tools)

    @staticmethod
    def _compact_generated_testcases(testcases: object) -> list[dict[str, Any]]:
        """Keep only testcase facts relevant to semantic adjudication.

        Args:
            testcases: Generated testcase manifest rows from flow state.

        Returns:
            Compact filename, expected-label, and intent rows.
        """
        if not isinstance(testcases, list):
            return []
        compact_rows: list[dict[str, Any]] = []
        for testcase in testcases:
            if not isinstance(testcase, dict):
                continue
            compact_rows.append(
                {
                    "filename": str(testcase.get("filename", "")),
                    "expected": testcase.get("expected", {}),
                    "intent": str(testcase.get("intent", "")),
                }
            )
        return compact_rows

    @classmethod
    def _compact_test_execution_report(cls, report: object, testcases: object) -> dict[str, Any]:
        """Summarize deterministic execution without paths or duplicate results.

        Args:
            report: Full evaluator output from flow state.
            testcases: Generated testcase manifest rows used to resolve intents.

        Returns:
            Test summary plus mismatch-only expected/observed evidence.
        """
        if not isinstance(report, dict):
            return {"available": False}

        manifests = {str(row.get("filename", "")): row for row in cls._compact_generated_testcases(testcases)}
        case_results = report.get("case_results", {})
        mismatches = report.get("mismatches", {})
        mismatch_evidence: list[dict[str, Any]] = []
        if isinstance(mismatches, dict) and isinstance(case_results, dict):
            for path in sorted(mismatches):
                filename = Path(str(path)).name
                case_result = case_results.get(path, {})
                if not isinstance(case_result, dict):
                    continue
                manifest = manifests.get(filename, {})
                mismatch_evidence.append(
                    {
                        "filename": filename,
                        "intent": str(manifest.get("intent", "")),
                        "expected": case_result.get("expected", {}),
                        "observed": case_result.get("observed", {}),
                    }
                )

        return {
            "available": bool(report.get("success", False)),
            "phase": str(report.get("phase", "")),
            "n_cases": int(report.get("n_cases", 0)),
            "n_passing_cases": int(report.get("n_passing_cases", 0)),
            "n_failing_cases": int(report.get("n_failing_cases", 0)),
            "per_category_f1": report.get("per_category_f1", {}),
            "error": str(report.get("error", "")),
            "mismatches": mismatch_evidence,
        }

    @classmethod
    def _compact_review_payload(cls, state: dict[str, Any]) -> dict[str, Any]:
        """Build the complete compact evidence payload for critic follow-ups.

        Args:
            state: Flow state containing the current deck and evaluation state.

        Returns:
            Minimal deterministic facts needed to reassess the deck.
        """
        testcases = state.get("generated_testcases", [])
        compilation = state.get("compilation_report", {})
        return {
            "generated_code": str(state.get("generated_code", "")),
            "compilation": {
                "success": (
                    bool(compilation.get("success", compilation.get("ok", False)))
                    if isinstance(compilation, dict)
                    else False
                ),
                "error": str(compilation.get("error", "")) if isinstance(compilation, dict) else "",
            },
            "test_execution": cls._compact_test_execution_report(state.get("test_execution_report", {}), testcases),
            "testcase_preparation": state.get("testcase_preparation_report", {}),
            "testcase_generator": str(state.get("testcase_generator", "")),
            "testcase_correction_failure": state.get("testcase_correction_failure", {}),
        }

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract specification and compact evidence from EGAR-DRC state.

        Args:
            state: Flow state carrying rule specification and generated code.

        Returns:
            Prompt-rendering input payload.
        """
        compact = self._compact_review_payload(state)
        return {
            "problem_id": str(state.get("problem_id", "")).strip(),
            "title": str(state.get("title", "")).strip(),
            "rule_description": str(state.get("rule_description", "")).strip(),
            "rules": state.get("rules", []),
            "categories": state.get("categories", []),
            "layers": state.get("layers", {}),
            "thresholds": state.get("thresholds", {}),
            "generated_code": compact["generated_code"],
            "testcase_generator": compact["testcase_generator"],
            "compilation_report": compact["compilation"],
            "test_execution_report": compact["test_execution"],
            "klayout_docs": load_klayout_docs(),
        }

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Format the latest generated deck and deterministic test result.

        Args:
            state: Flow state carrying a newly tested code revision.

        Returns:
            Review request for the existing critic conversation.
        """
        handoff = state.get("share_message", {})
        if not isinstance(handoff, dict):
            return None
        recovery = state.get("code_correction_compile_recovery")
        if isinstance(recovery, dict):
            return (
                "Your previous code-correction recommendation exhausted its compilation budget. "
                "Reassess the failed corrected deck and its compiler diagnostics. Return a new complete "
                "adjudication JSON object. You may recommend code correction, testcase correction, both, or neither.\n\n"
                + json.dumps(
                    {
                        "failed_corrected_code": str(recovery.get("failed_corrected_code", "")),
                        "compilation_report": recovery.get("compilation_report", {}),
                    },
                    indent=2,
                    ensure_ascii=True,
                    default=str,
                )
            )
        return (
            "Re-adjudicate the newly generated deck and generated testcases using the latest "
            "deterministic result. Return the complete adjudication JSON object again.\n\n"
            + json.dumps(self._compact_review_payload(state), indent=2, ensure_ascii=True, default=str)
        )

    def output_model(self) -> type[TestcaseAdjudicationOutput] | None:
        """Return the structured output schema for criticism."""
        return TestcaseAdjudicationOutput

    async def invoke(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Request code or suite correction, accepting only passing execution.

        Args:
            state: Current flow state.

        Returns:
            Updated state containing the critic decision and routing flags.
        """
        started_at = time.perf_counter()
        ai_result: dict[str, Any] = {}
        try:
            ai_result = await self._invoke_model(state)
            parsed_output = ai_result["parsed_output"]
            if not isinstance(parsed_output, TestcaseAdjudicationOutput):
                raise ValueError("Parsed output model mismatch for TestcaseAdjudicationOutput.")
        except Exception as exc:
            ai_result = getattr(exc, "update", ai_result)
            iteration = int(state.get("critic_review_iteration", 0)) + 1
            return {
                **graph_token_usage_update(ai_result),
                "critic_review_iteration": iteration,
                "critic_request_succeeded": False,
                "pending_code_correction": False,
                "pending_testcase_correction": False,
                "critic_review": {
                    "code_correction_instruction": "",
                    "testcase_script_update_instruction": "",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "flow_status": "critic_request_failed",
                "llm_request_metrics": append_llm_metric(
                    state, agent="testcase_adjudication", started_at=started_at, success=False, update=ai_result, error=exc
                ),
                "loop_metrics": append_loop_metric(
                    state,
                    phase="testcase_adjudication",
                    started_at=started_at,
                    success=False,
                    outcome={"error": str(exc)},
                ),
                "flow_events": append_flow_event(state, "testcase_adjudication", "failed", error=str(exc)),
            }

        critic_review = parsed_output.model_dump()
        code_instruction = parsed_output.code_correction_instruction.strip()
        testcase_instruction = parsed_output.testcase_script_update_instruction.strip()
        needs_revision = bool(code_instruction)
        final_code = str(state.get("generated_code", "")).strip()
        critic_review_iteration = int(state.get("critic_review_iteration", 0)) + 1
        code_revision_iteration = int(state.get("code_revision_iteration", 0))
        # Track correction rounds separately from compiler retries.
        code_correction_iteration = int(state.get("code_correction_iteration", 0)) + int(bool(code_instruction))

        if not code_instruction and not testcase_instruction and has_passing_execution(state):
            flow_status = "accepted"
        elif not code_instruction and not testcase_instruction:
            flow_status = "critic_inconclusive"
        else:
            flow_status = "critic_revision_pending"

        adjudication_metrics = dict(state.get("testcase_adjudication_metrics", {}))
        adjudication_metrics.update(
            {
                "reviews": int(adjudication_metrics.get("reviews", 0)) + 1,
                "code_updates_requested": int(adjudication_metrics.get("code_updates_requested", 0))
                + int(bool(code_instruction)),
                "testcase_updates_requested": int(adjudication_metrics.get("testcase_updates_requested", 0))
                + int(bool(testcase_instruction)),
            }
        )

        update = {
            **state,
            "critic_review": critic_review,
            "critic_review_iteration": critic_review_iteration,
            "code_correction_iteration": code_correction_iteration,
            "code_revision_iteration": code_revision_iteration,
            "pending_code_correction": bool(code_instruction),
            "pending_testcase_correction": bool(testcase_instruction),
            "testcase_adjudication_metrics": adjudication_metrics,
            "needs_revision": needs_revision,
            "final_code": final_code,
            "flow_status": flow_status,
            "share_message": {"critic_feedback": critic_review},
        }
        recovery = state.get("code_correction_compile_recovery")
        if isinstance(recovery, dict):
            restored_code = str(recovery.get("previous_compiling_code", "")).strip()
            # Restore a compiling deck before starting fresh correction history.
            restored_report = recovery.get("previous_compilation_report", {})
            update["code_correction_compile_iteration"] = 0
            update["code_correction_compile_recovery"] = None
            update["code_correction_memory_reset_pending"] = bool(code_instruction)
            if restored_code:
                update["generated_code"] = restored_code
                update["final_code"] = restored_code
                update["compilation_succeeded"] = True
                update["compilation_report"] = restored_report if isinstance(restored_report, dict) else {}
        return {
            **owned_update(
                update,
                (
                    "critic_review",
                    "critic_review_iteration",
                    "code_correction_iteration",
                    "code_revision_iteration",
                    "pending_code_correction",
                    "pending_testcase_correction",
                    "testcase_adjudication_metrics",
                    "needs_revision",
                    "code_correction_compile_iteration",
                    "code_correction_compile_recovery",
                    "code_correction_memory_reset_pending",
                    "generated_code",
                    "final_code",
                    "compilation_succeeded",
                    "compilation_report",
                ),
            ),
            **graph_token_usage_update(ai_result),
            "critic_request_succeeded": True,
            "share_message": {"critic_feedback": critic_review},
            "flow_status": flow_status,
            "llm_request_metrics": append_llm_metric(
                state,
                agent="testcase_adjudication",
                started_at=started_at,
                success=True,
                update=ai_result,
            ),
            "loop_metrics": append_loop_metric(state, phase="testcase_adjudication", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "testcase_adjudication", "succeeded", critic_review=critic_review),
        }
