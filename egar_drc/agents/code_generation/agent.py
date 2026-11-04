"""Code-generation agent for KLayout DRC decks."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from egar_drc.agents.base import RoleAgent
from egar_drc.agents.flow_node_helpers import (
    append_flow_event,
    append_llm_metric,
    append_loop_metric,
    graph_token_usage_update,
)
from egar_drc.models.runtime import ModelRuntime
from langchain_core.tools import BaseTool

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_klayout_docs() -> str:
    """Cache the shared prompt-prefix reference, or empty text if unavailable."""
    path = _PACKAGE_ROOT / "reference" / "klayout_docs.txt"
    try:
        return path.read_text(encoding="utf-8").rstrip()
    except OSError:
        return ""


class CodeGenerationAgent(RoleAgent):
    """Generate or revise a complete KLayout Ruby DRC deck."""

    # Start fresh history on regeneration.
    reset_state_key = "code_regeneration_pending"

    def __init__(self, runtime: ModelRuntime, *, tools: list[BaseTool] | None = None) -> None:
        """Initialize the agent with its publication descriptor."""
        super().__init__(runtime, Path(__file__).with_name("code_generation_agent.yaml"), tools=tools)

    @staticmethod
    def _ruby_string_literal(value: str) -> str:
        """Encode text as a Ruby-compatible double-quoted string."""
        return json.dumps(value, ensure_ascii=True)

    @staticmethod
    def _format_rule_category_contract(rules: Any) -> str:
        """Render rule text beside its required output category."""
        if not isinstance(rules, list):
            return "- No structured rules were supplied."
        lines: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                lines.append(f"- {rule}")
                continue
            text = str(rule.get("text", "")).strip()
            category = str(rule.get("category", "")).strip()
            if text and category:
                lines.append(f'- {text} -> output category "{category}"')
            elif text:
                lines.append(f"- {text}")
            elif category:
                lines.append(f'- output category "{category}"')
        return "\n".join(lines) if lines else "- No structured rules were supplied."

    @staticmethod
    def _format_layer_contract(layers: Any) -> str:
        """Render named GDS layers as layer/datatype pairs."""
        if not isinstance(layers, dict):
            return "- No structured layer map was supplied."
        lines = [
            (
                f"- {name} = ({pair[0]}, {pair[1]})"
                if isinstance(pair, (list, tuple)) and len(pair) == 2
                else f"- {name} = {pair}"
            )
            for name, pair in layers.items()
        ]
        return "\n".join(lines) if lines else "- No structured layer map was supplied."

    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Extract code-generation prompt variables from flow state."""
        title = str(state.get("title", "")).strip()
        return {
            "problem_id": str(state.get("problem_id", "")).strip(),
            "title": title,
            "ruby_report_title": self._ruby_string_literal(title),
            "rule_description": str(state.get("rule_description", "")).strip(),
            "rule_category_contract": self._format_rule_category_contract(state.get("rules", [])),
            "categories": state.get("categories", []),
            "layer_contract": self._format_layer_contract(state.get("layers", {})),
            "thresholds": state.get("thresholds", {}),
            "klayout_docs": load_klayout_docs(),
        }

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Format only the latest revision feedback."""
        handoff = state.get("share_message", {})
        if not isinstance(handoff, dict):
            return None
        feedback = handoff.get("compilation") or handoff.get("critic_feedback") or handoff.get("test_execution")
        if not isinstance(feedback, dict):
            return None
        return (
            "Revise the deck you returned in the previous turn. The latest compilation, "
            "test execution, or critic feedback is below. Return only the complete "
            "revised Ruby deck, with no Markdown fences or explanation.\n\n"
            + json.dumps(feedback, indent=2, ensure_ascii=True, default=str)
        )

    @staticmethod
    def strip_code_fences(code: str) -> str:
        """Remove optional Markdown fences from a generated deck."""
        stripped = code.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        return "\n".join(lines).strip()

    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Generate a deck and return a narrow LangGraph state update."""
        started_at = time.perf_counter()
        invocation_state = dict(state)
        result: dict[str, Any] = {}
        try:
            result = await self._invoke_model(invocation_state)
            code = self.strip_code_fences(str(result["final_output"]))
            if not code:
                raise ValueError("Code-generation agent returned an empty Ruby deck.")
        except Exception as exc:
            result = getattr(exc, "update", result)
            report = {"success": False, "phase": "code_generation", "error": f"{type(exc).__name__}: {exc}"}
            return {
                **graph_token_usage_update(result),
                "share_message": {"compilation": report},
                "code_regeneration_pending": False,
                "flow_status": "code_generation_failed",
                "llm_request_metrics": append_llm_metric(
                    state, agent="code_generation", started_at=started_at, success=False, update=result, error=exc
                ),
                "loop_metrics": append_loop_metric(
                    state, phase="code_generation", started_at=started_at, success=False, outcome={"error": str(exc)}
                ),
                "flow_events": append_flow_event(state, "code_generation", "failed", error=str(exc)),
            }
        iteration = int(state.get("code_generation_iteration", 0)) + 1
        return {
            "generated_code": code,
            "final_code": code,
            "code_generation_iteration": iteration,
            "code_regeneration_pending": False,
            "code_origin": "initial_code_generation",
            "flow_status": "code_ready_for_compilation",
            **graph_token_usage_update(result),
            "llm_request_metrics": append_llm_metric(
                state, agent="code_generation", started_at=started_at, success=True, update=result
            ),
            "loop_metrics": append_loop_metric(state, phase="code_generation", started_at=started_at, success=True),
            "flow_events": append_flow_event(state, "code_generation", "succeeded"),
        }

