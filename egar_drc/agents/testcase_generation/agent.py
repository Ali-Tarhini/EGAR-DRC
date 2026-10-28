"""Generate complete Python testcase scripts for EGAR-DRC problems."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from egar_drc.agents.base import RoleAgent
from egar_drc.agents.flow_node_helpers import (
    append_flow_event,
    append_llm_metric,
    append_loop_metric,
    graph_token_usage_update,
    owned_update,
)
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool

_KLAYOUT_DOCS_PATH = Path(__file__).resolve().parents[2] / "reference" / "klayout_docs.txt"
_KLAYOUT_DOCS_CACHE: str | None = None


def _load_klayout_docs() -> str:
    """Cache the same reference text used in every role's initial prompt prefix.

    Returns:
        KLayout reference text, or an empty string if it cannot be read.
    """
    global _KLAYOUT_DOCS_CACHE
    if _KLAYOUT_DOCS_CACHE is None:
        try:
            _KLAYOUT_DOCS_CACHE = _KLAYOUT_DOCS_PATH.read_text(encoding="utf-8").rstrip()
        except OSError:
            _KLAYOUT_DOCS_CACHE = ""
    return _KLAYOUT_DOCS_CACHE


class TestcaseGenerationAgent(RoleAgent):
    """Generate a ``pya`` testcase script and its required edge-case coverage."""

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Initialize the agent from its local descriptor."""
        descriptor_path = Path(__file__).with_name("testcase_generation_agent.yaml")
        super().__init__(runtime, descriptor_path, tools=tools)

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract prompt variables from the EGAR-DRC flow state.

        Args:
            state: Flow state carrying the target rule and dataset context.

        Returns:
            Prompt-rendering input payload.
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
            "max_testcases": int(state.get("max_testcases", 8)),
            "klayout_docs": _load_klayout_docs(),
        }

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Format testcase-generator execution feedback for a retry.

        Args:
            state: Flow state carrying the latest generator failure.

        Returns:
            Focused repair request, or None when no retry feedback exists.
        """
        handoff = state.get("share_message", {})
        if not isinstance(handoff, dict):
            return None
        feedback = handoff.get("testcase_generation")
        if not isinstance(feedback, dict):
            return None
        return (
            "The testcase generator you returned failed deterministic preparation. "
            "Return a complete corrected Python generator script only. Preserve the "
            "original problem and manifest contract.\n\n"
            + json.dumps(feedback, indent=2, ensure_ascii=True, default=str)
        )

    @staticmethod
    def _strip_code_fences(script: str) -> str:
        """Remove accidental Markdown fences from generated Python source.

        Args:
            script: Model output that may be wrapped in a fenced code block.

        Returns:
            Executable Python source without surrounding fence markers.
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
        """Generate a script; deterministic preparation creates and checks GDS files.

        Args:
            state: Current flow state.

        Returns:
            Updated state containing the executable testcase script.
        """
        started_at = time.perf_counter()
        ai_result: dict[str, Any] = {}
        try:
            ai_result = await self._invoke_model(state)
            testcase_generator = self._strip_code_fences(str(ai_result.get("final_output", "")))
            if not testcase_generator:
                raise ValueError("Edge-case agent returned an empty testcase generator.")
        except Exception as exc:
            ai_result = getattr(exc, "update", ai_result)
            report = {"success": False, "phase": "testcase_generation", "error": f"{type(exc).__name__}: {exc}"}
            return {
                **graph_token_usage_update(ai_result),
                "testcase_generation_iteration": int(state.get("testcase_generation_iteration", 0)) + 1,
                "testcase_generation_succeeded": False,
                "testcase_preparation_report": report,
                "share_message": {"testcase_generation": report},
                "flow_status": "testcase_generation_failed",
                "llm_request_metrics": append_llm_metric(
                    state,
                    agent="testcase_generation",
                    started_at=started_at,
                    success=False,
                    update=ai_result,
                    error=exc,
                ),
                "loop_metrics": append_loop_metric(
                    state,
                    phase="testcase_generation",
                    started_at=started_at,
                    success=False,
                    outcome={"error": str(exc)},
                ),
                "flow_events": append_flow_event(state, "testcase_generation", "failed", error=str(exc)),
            }
        testcase_generation_iteration = int(state.get("testcase_generation_iteration", 0)) + 1
        update = {
            **state,
            "testcase_generator": testcase_generator,
            "testcase_generation_iteration": testcase_generation_iteration,
        }
        return {
            **owned_update(
                update,
                (
                    "testcase_generator",
                    "testcase_generation_iteration",
                ),
            ),
            **graph_token_usage_update(ai_result),
            "testcase_origin": "initial_testcases",
            "flow_status": "testcase_script_ready_for_preparation",
            "llm_request_metrics": append_llm_metric(
                state,
                agent="testcase_generation",
                started_at=started_at,
                success=True,
                update=ai_result,
            ),
            "loop_metrics": append_loop_metric(state, phase="testcase_generation", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "testcase_generation", "succeeded"),
        }
