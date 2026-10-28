"""Single generation with no compiler or feedback loop."""

from typing import Any, cast

from egar_drc.orchestration.code_generation_baseline import EgarDrcCodeGenerationBaselineFlow
from egar_drc.orchestration.egar_drc import EgarDrcState, _event
from langgraph.graph import END, START, StateGraph


class EgarDrcSingleSampleBaselineFlow(EgarDrcCodeGenerationBaselineFlow):
    """Generate one unvalidated deck without claiming acceptance."""

    assistant_id = "egar_drc_single_sample_baseline"

    def make_graph(self) -> Any:
        """Build the straight-through single-sample graph."""
        builder = StateGraph(EgarDrcState)
        builder.add_node("initialize", self.initialize)
        builder.add_node("generate_code", cast(Any, self._code_generation_agent.invoke))
        builder.add_node("finalize", self.finalize)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate_code")
        builder.add_edge("generate_code", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile()

    async def finalize(self, state: EgarDrcState) -> dict[str, Any]:
        """Record generation success without compilation or candidate selection."""
        generated = state.get("flow_status") == "code_ready_for_compilation"
        status = "generated" if generated else "code_generation_failed"
        return {
            "flow_status": status,
            "final_selection": {
                "status": status,
                "selected_candidate_id": "",
                "selected_reason": "single_generated_sample" if generated else "code_generation_failed",
                "best_code_candidate": {},
            },
            "flow_events": _event(state, "flow", "finished", final_status=status),
        }
