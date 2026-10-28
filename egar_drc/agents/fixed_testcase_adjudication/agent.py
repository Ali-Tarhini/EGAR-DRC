"""Code-only adjudication against a fixed generated testcase suite."""

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
from egar_drc.agents.testcase_adjudication.agent import TestcaseAdjudicationAgent
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class FixedTestcaseAdjudicationOutput(BaseModel):
    """Structured code-only correction decision."""

    code_correction_instruction: str = Field(default="")


class FixedTestcaseAdjudicationAgent(RoleAgent):
    """Adjudicate code against fixed testcases without requesting suite repair."""

    descriptor_filename = "fixed_testcase_adjudication_agent.yaml"

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Load the role-specific descriptor and injected client."""
        super().__init__(runtime, Path(__file__).with_name(self.descriptor_filename), tools=tools)

    def _review_payload(self, state: dict[str, Any]) -> dict[str, Any]:
        """Keep the deterministic evidence for the fixed-testcase role."""
        return TestcaseAdjudicationAgent._compact_review_payload(state)

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Map specification and role-appropriate evidence into the prompt."""
        compact = self._review_payload(state)
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
        """Re-adjudicate the latest deck, including exhausted repair diagnostics."""
        if not isinstance(state.get("share_message", {}), dict):
            return None
        recovery = state.get("code_correction_compile_recovery")
        if isinstance(recovery, dict):
            return (
                "Your previous code-correction recommendation exhausted its compilation budget. "
                "Reassess the failed corrected deck and its compiler diagnostics. Return a new complete "
                "code-only adjudication JSON object. You may recommend code correction or no correction.\n\n"
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
            "Re-adjudicate the newly generated deck using the latest deterministic result. "
            "Return the complete code-only adjudication JSON object again.\n\n"
            + json.dumps(self._review_payload(state), indent=2, ensure_ascii=True, default=str)
        )

    def output_model(self) -> type[FixedTestcaseAdjudicationOutput]:
        """Return the single-instruction response schema."""
        return FixedTestcaseAdjudicationOutput

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Publish a code decision, preserving upstream recovery and counters."""
        started_at = time.perf_counter()
        iteration = int(state.get("critic_review_iteration", 0)) + 1
        result: dict[str, Any] = {}
        try:
            result = await self._invoke_model(state)
            parsed = result["parsed_output"]
            if not isinstance(parsed, FixedTestcaseAdjudicationOutput):
                raise ValueError("Parsed output model mismatch for FixedTestcaseAdjudicationOutput.")
        except Exception as exc:
            result = getattr(exc, "update", result)
            return {
                **graph_token_usage_update(result),
                "critic_review_iteration": iteration,
                "critic_request_succeeded": False,
                "pending_code_correction": False,
                "critic_review": {"code_correction_instruction": "", "error": f"{type(exc).__name__}: {exc}"},
                "flow_status": "critic_request_failed",
                "llm_request_metrics": append_llm_metric(
                    state, agent="fixed_testcase_adjudication", started_at=started_at, success=False, update=result, error=exc
                ),
                "loop_metrics": append_loop_metric(
                    state, phase="fixed_testcase_adjudication", started_at=started_at, success=False, outcome={"error": str(exc)}
                ),
                "flow_events": append_flow_event(state, "fixed_testcase_adjudication", "failed", error=str(exc)),
            }
        instruction = parsed.code_correction_instruction.strip()
        review = parsed.model_dump()
        metrics = dict(state.get("testcase_adjudication_metrics", {}))
        metrics.update(
            reviews=int(metrics.get("reviews", 0)) + 1,
            code_updates_requested=int(metrics.get("code_updates_requested", 0)) + int(bool(instruction)),
            testcase_updates_requested=int(metrics.get("testcase_updates_requested", 0)),
        )
        update = {
            **state,
            "critic_review": review,
            "critic_review_iteration": iteration,
            "code_correction_iteration": int(state.get("code_correction_iteration", 0)) + int(bool(instruction)),
            "code_revision_iteration": int(state.get("code_revision_iteration", 0)),
            "pending_code_correction": bool(instruction),
            "pending_testcase_correction": False,
            "testcase_adjudication_metrics": metrics,
            "needs_revision": bool(instruction),
            "final_code": str(state.get("generated_code", "")).strip(),
            "flow_status": (
                "critic_revision_pending" if instruction
                else "accepted" if has_passing_execution(state)
                else "critic_inconclusive"
            ),
        }
        recovery = state.get("code_correction_compile_recovery")
        if isinstance(recovery, dict):
            # Recover code without changing the fixed testcase suite.
            restored_code = str(recovery.get("previous_compiling_code", "")).strip()
            restored_report = recovery.get("previous_compilation_report", {})
            update.update(
                code_correction_compile_iteration=0,
                code_correction_compile_recovery=None,
                code_correction_memory_reset_pending=bool(instruction),
            )
            if restored_code:
                update.update(
                    generated_code=restored_code,
                    final_code=restored_code,
                    compilation_succeeded=True,
                    compilation_report=restored_report if isinstance(restored_report, dict) else {},
                )
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
            **graph_token_usage_update(result),
            "critic_request_succeeded": True,
            "share_message": {"critic_feedback": review},
            "flow_status": update["flow_status"],
            "llm_request_metrics": append_llm_metric(
                state, agent="fixed_testcase_adjudication", started_at=started_at, success=True, update=result
            ),
            "loop_metrics": append_loop_metric(state, phase="fixed_testcase_adjudication", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "fixed_testcase_adjudication", "succeeded", critic_review=review),
        }
