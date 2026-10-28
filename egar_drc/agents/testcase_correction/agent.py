"""Testcase-generator correction agent for Rule2DRC flows."""

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
    owned_update,
)
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool


class TestcaseCorrectionAgent(RoleAgent):
    """Correct a Python ``pya`` testcase generator from critic or execution feedback."""

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Initialize the testcase correction agent from its descriptor."""
        descriptor_path = Path(__file__).with_name("testcase_correction_agent.yaml")
        super().__init__(runtime, descriptor_path, tools=tools)

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract prompt variables for testcase-script correction.

        Args:
            state: Flow state containing the current script and critic feedback.

        Returns:
            Descriptor prompt values.
        """
        return {
            "problem_id": str(state.get("problem_id", "")).strip(),
            "title": str(state.get("title", "")).strip(),
            "rule_description": str(state.get("rule_description", "")).strip(),
            "rules": state.get("rules", []),
            "categories": state.get("categories", []),
            "layers": state.get("layers", {}),
            "thresholds": state.get("thresholds", {}),
            "spec_yaml": state.get("spec_yaml", ""),
            "layer_context": state.get("layer_context", ""),
            "current_testcase_generator": str(state.get("testcase_generator", "")).strip(),
            "testcase_update_instruction": str(state.get("testcase_update_instruction", "")).strip(),
            "testcase_feedback": json.dumps(
                state.get("testcase_feedback", {}),
                indent=2,
                ensure_ascii=True,
                default=str,
            ),
            "klayout_docs": load_klayout_docs(),
        }

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Format deterministic script-preparation feedback for a retry.

        Args:
            state: Flow state carrying the latest testcase preparation failure.

        Returns:
            Follow-up message for the continuing correction conversation.
        """
        handoff = state.get("share_message", {})
        if not isinstance(handoff, dict):
            return None
        feedback = handoff.get("testcase_generation")
        if isinstance(feedback, dict) and not bool(feedback.get("success", False)):
            # Prioritize preparation failures over semantic correction.
            return (
                "The corrected testcase generator still failed deterministic preparation. "
                "Return a complete corrected Python generator script only. Preserve the rule, "
                "manifest, and OUT_DIR contracts.\n\n" + json.dumps(feedback, indent=2, ensure_ascii=True, default=str)
            )
        critic_feedback = handoff.get("critic_feedback")
        if isinstance(critic_feedback, dict):
            instruction = str(critic_feedback.get("testcase_script_update_instruction", "")).strip()
            if instruction:
                return (
                    "Apply this new critic-requested update to the current complete testcase generator. "
                    "Return only the complete revised Python generator script.\n\n"
                    + json.dumps(
                        {"testcase_script_update_instruction": instruction},
                        indent=2,
                        ensure_ascii=True,
                        default=str,
                    )
                )
        return None

    @staticmethod
    def _strip_code_fences(script: str) -> str:
        """Remove accidental Markdown fences from generated Python source.

        Args:
            script: Model output that may contain fenced code.

        Returns:
            Python source without surrounding fences.
        """
        stripped = script.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    async def invoke(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Correct the testcase generator script.

        Args:
            state: Current flow state.

        Returns:
            Updated state containing the corrected testcase generator.
        """
        started_at = time.perf_counter()
        review = state.get("critic_review", {})
        instruction = (
            str(review.get("testcase_script_update_instruction", "")).strip() if isinstance(review, dict) else ""
        )
        correction_state = {
            **state,
            "testcase_update_instruction": instruction,
            "testcase_feedback": {
                "critic_review": state.get("critic_review", {}),
                "testcase_preparation_report": state.get("testcase_preparation_report", {}),
                "test_execution_report": state.get("test_execution_report", {}),
                "generated_testcases": state.get("generated_testcases", []),
            },
        }
        ai_result: dict[str, Any] = {}
        try:
            ai_result = await self._invoke_model(correction_state)
            testcase_generator = self._strip_code_fences(str(ai_result.get("final_output", "")))
            if not testcase_generator:
                raise ValueError("Testcase-correction agent returned an empty generator script.")
        except Exception as exc:
            ai_result = getattr(exc, "update", ai_result)
            report = {"success": False, "phase": "testcase_correction", "error": f"{type(exc).__name__}: {exc}"}
            return {
                **graph_token_usage_update(ai_result),
                "share_message": {"testcase_generation": report},
                "testcase_origin": "testcase_correction",
                "flow_status": "testcase_correction_failed",
                "llm_request_metrics": append_llm_metric(
                    state,
                    agent="testcase_correction",
                    started_at=started_at,
                    success=False,
                    update=ai_result,
                    error=exc,
                ),
                "loop_metrics": append_loop_metric(
                    state,
                    phase="testcase_correction",
                    started_at=started_at,
                    success=False,
                    outcome={"error": str(exc)},
                ),
                "flow_events": append_flow_event(state, "testcase_correction", "failed", error=str(exc)),
            }
        update = {
            **correction_state,
            "testcase_generator": testcase_generator,
            # Track revisions separately from preparation retries.
            "testcase_correction_iteration": int(state.get("testcase_correction_iteration", 0)) + 1,
        }
        return {
            **owned_update(update, ("testcase_generator", "testcase_correction_iteration")),
            **graph_token_usage_update(ai_result),
            "pending_testcase_correction": False,
            "testcase_origin": "testcase_correction",
            "flow_status": "corrected_testcase_script_ready_for_preparation",
            "llm_request_metrics": append_llm_metric(
                state,
                agent="testcase_correction",
                started_at=started_at,
                success=True,
                update=ai_result,
            ),
            "loop_metrics": append_loop_metric(state, phase="testcase_correction", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "testcase_correction", "succeeded"),
        }
