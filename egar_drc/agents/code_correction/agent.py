"""Dedicated corrective Ruby-deck agent for EGAR-DRC."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from egar_drc.agents.base import RoleAgent
from egar_drc.agents.code_generation.agent import (
    CodeGenerationAgent,
    load_klayout_docs,
)
from egar_drc.agents.flow_node_helpers import (
    append_flow_event,
    append_llm_metric,
    append_loop_metric,
    graph_token_usage_update,
    owned_update,
)
from egar_drc.agents.testcase_adjudication.agent import TestcaseAdjudicationAgent
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool


class CodeCorrectionAgent(RoleAgent):
    """Correct an existing Ruby deck from compilation, test, or critic evidence."""

    reset_state_key = "code_correction_memory_reset_pending"

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Initialize the agent from its local descriptor."""
        descriptor_path = Path(__file__).with_name("code_correction_agent.yaml")
        super().__init__(runtime, descriptor_path, tools=tools)

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract the current deck and deterministic feedback for correction.

        Args:
            state: Flow state carrying the candidate deck and its latest evidence.

        Returns:
            Descriptor prompt values for a correction request.
        """
        title = str(state.get("title", "")).strip()
        handoff = state.get("share_message", {})
        feedback: dict[str, Any] = {}
        if isinstance(handoff, dict):
            # Prioritize compiler feedback over semantic correction.
            compilation = handoff.get("compilation")
            if isinstance(compilation, dict) and not bool(compilation.get("success", compilation.get("ok", False))):
                feedback = self._compact_correction_feedback("compilation", compilation)
            else:
                for key in ("critic_feedback", "test_execution"):
                    candidate = handoff.get(key)
                    if isinstance(candidate, dict):
                        feedback = self._compact_correction_feedback(key, candidate)
                        break
        return {
            "problem_id": str(state.get("problem_id", "")).strip(),
            "title": title,
            "ruby_report_title": CodeGenerationAgent._ruby_string_literal(title),
            "rule_description": str(state.get("rule_description", "")).strip(),
            "rule_category_contract": CodeGenerationAgent._format_rule_category_contract(state.get("rules", [])),
            "categories": state.get("categories", []),
            "layer_contract": CodeGenerationAgent._format_layer_contract(state.get("layers", {})),
            "thresholds": state.get("thresholds", {}),
            "candidate_code": str(state.get("generated_code", state.get("final_code", ""))).strip(),
            "correction_reason": str(state.get("code_correction_reason", "")).strip(),
            "correction_feedback": json.dumps(feedback, indent=2, ensure_ascii=True, default=str),
            "klayout_docs": load_klayout_docs(),
        }

    @staticmethod
    def _compact_correction_feedback(kind: str, feedback: dict[str, Any]) -> dict[str, Any]:
        """Remove duplicate evaluator payloads before sending correction evidence.

        Args:
            kind: Handoff message category.
            feedback: Raw deterministic handoff payload.

        Returns:
            Compact compilation, testcase, or critic evidence.
        """
        if kind == "compilation":
            return {
                "phase": str(feedback.get("phase", "compilation")),
                "success": bool(feedback.get("success", feedback.get("ok", False))),
                "returncode": feedback.get("returncode"),
                "error": str(feedback.get("error", "")),
            }
        if kind == "test_execution":
            generated_testcases = feedback.get("generated_testcases", [])
            return {
                "test_execution": TestcaseAdjudicationAgent._compact_test_execution_report(
                    feedback.get("test_execution_report", {}),
                    generated_testcases,
                ),
                "testcase_failure_evidence": feedback.get("testcase_failure_evidence", []),
                "critic_review": feedback.get("critic_review", {}),
            }
        return feedback

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Provide updated KLayout failures to a continuing correction session.

        Args:
            state: Flow state following another failed correction attempt.

        Returns:
            Follow-up correction request when fresh deterministic feedback exists.
        """
        handoff = state.get("share_message", {})
        if not isinstance(handoff, dict):
            return None
        compilation = handoff.get("compilation")
        if isinstance(compilation, dict) and not bool(compilation.get("success", compilation.get("ok", False))):
            return (
                "The preceding correction still did not compile. Correct the complete Ruby deck again using this "
                "KLayout result. Return only the complete revised Ruby deck.\n\n"
                + json.dumps(
                    self._compact_correction_feedback("compilation", compilation),
                    indent=2,
                    ensure_ascii=True,
                    default=str,
                )
            )
        critic_feedback = handoff.get("critic_feedback")
        if isinstance(critic_feedback, dict):
            instruction = str(critic_feedback.get("code_correction_instruction", "")).strip()
            if instruction:
                return (
                    "Apply this new critic-requested correction to the current complete Ruby deck. "
                    "Return only the complete revised Ruby deck.\n\n"
                    + json.dumps(
                        {"code_correction_instruction": instruction},
                        indent=2,
                        ensure_ascii=True,
                        default=str,
                    )
                )
        return None

    async def invoke(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Correct the current DRC deck and retain the direct Ruby output.

        Args:
            state: Current correction state.

        Returns:
            Corrected Ruby source, or a failure update with captured usage when
            invocation fails or the returned deck is empty.
        """
        started_at = time.perf_counter()
        correction_state = {
            **state,
            "code_revision_iteration": int(state.get("code_revision_iteration", 0)) + 1,
            "code_correction_reason": "critic_review",
        }
        ai_result: dict[str, Any] = {}
        try:
            ai_result = await self._invoke_model(correction_state)
            final_code = CodeGenerationAgent.strip_code_fences(str(ai_result.get("final_output", "")))
            if not final_code:
                raise ValueError("Code-correction agent returned an empty Ruby deck.")
        except Exception as exc:
            ai_result = getattr(exc, "update", ai_result)
            report = {"success": False, "phase": "code_correction", "error": f"{type(exc).__name__}: {exc}"}
            return {
                **graph_token_usage_update(ai_result),
                "share_message": {"critic_feedback": report},
                "code_origin": "code_correction",
                "code_correction_memory_reset_pending": False,
                "flow_status": "code_correction_failed",
                "llm_request_metrics": append_llm_metric(
                    state,
                    agent="code_correction",
                    started_at=started_at,
                    success=False,
                    update=ai_result,
                    error=exc,
                ),
                "loop_metrics": append_loop_metric(
                    state,
                    phase="code_correction",
                    started_at=started_at,
                    success=False,
                    outcome={"error": str(exc)},
                ),
                "flow_events": append_flow_event(state, "code_correction", "failed", error=str(exc)),
            }
        update = {
            **correction_state,
            "generated_code": final_code,
            "final_code": final_code,
        }
        return {
            **owned_update(
                update,
                (
                    "generated_code",
                    "final_code",
                    "code_revision_iteration",
                ),
            ),
            **graph_token_usage_update(ai_result),
            "pending_code_correction": False,
            "code_correction_memory_reset_pending": False,
            "code_origin": "code_correction",
            "code_correction_reason": "critic_review",
            "flow_status": "corrected_code_ready_for_compilation",
            "llm_request_metrics": append_llm_metric(
                state,
                agent="code_correction",
                started_at=started_at,
                success=True,
                update=ai_result,
            ),
            "loop_metrics": append_loop_metric(state, phase="code_correction", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "code_correction", "succeeded"),
        }
