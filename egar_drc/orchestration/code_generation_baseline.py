"""Code-generation-only baseline with standalone compilation feedback."""

from typing import Any, Literal, cast
from uuid import uuid4

from egar_drc.agents.code_generation.agent import CodeGenerationAgent
from egar_drc.models.runtime import ModelRuntime
from egar_drc.orchestration.egar_drc import EgarDrcFlow, EgarDrcState, _event
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph


class EgarDrcCodeGenerationBaselineFlow(EgarDrcFlow):
    """Generate and compile a deck without testcase or critic roles."""

    assistant_id = "egar_drc_code_generation_baseline"

    def __init__(self, runtime: ModelRuntime) -> None:
        """Initialize only the generation role and selected graph."""
        self.runtime = runtime
        self._code_generation_agent = CodeGenerationAgent(runtime)
        self._graph = self.make_graph()

    async def invoke(self, state: dict[str, Any], config: RunnableConfig | None = None) -> dict[str, Any]:
        """Await compile and regeneration loops, unlike sync Runnable.invoke."""
        state = {**state, "run_id": state.get("run_id") or uuid4().hex}
        compile_budget = int(state.get("max_code_generation_compile_iterations", 15))
        regeneration_budget = int(state.get("max_code_generation_retry_iterations", 5))
        graph_config = dict(config or {})
        # Allow enough LangGraph steps for the configured retries.
        graph_config["recursion_limit"] = max(
            int(str(graph_config.get("recursion_limit", 0))),
            12 + compile_budget * (regeneration_budget + 1) * 3,
        )
        return cast(dict[str, Any], await self._graph.ainvoke(state, config=cast(RunnableConfig, graph_config)))

    def make_graph(self) -> Any:
        """Build the generation and compiler feedback loop."""
        builder = StateGraph(EgarDrcState)
        builder.add_node("initialize", self.initialize)
        builder.add_node("generate_code", self._code_generation_agent.invoke)
        builder.add_node("compile_code", self.compile_code)
        builder.add_node("finalize", self.finalize)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate_code")
        builder.add_edge("generate_code", "compile_code")
        builder.add_conditional_edges(
            "compile_code",
            self.route_after_compile,
            {
                "generate_code": "generate_code",
                "finalize": "finalize",
            },
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def route_after_compile(state: EgarDrcState) -> Literal["generate_code", "finalize"]:
        """Stop on compilation success or exhausted initial generation retries."""
        if bool(state.get("compilation_succeeded", False)):
            return "finalize"
        if int(state.get("code_regeneration_iteration", 0)) > int(state.get("max_code_generation_retry_iterations", 5)):
            return "finalize"
        return "generate_code"

    async def finalize(self, state: EgarDrcState) -> dict[str, Any]:
        """Mark compilation-only acceptance, without claiming testcase success."""
        compiled = bool(state.get("compilation_succeeded", False))
        status = "accepted" if compiled else "initial_code_generation_failed"
        selected_id = self._current_candidate_id(state)
        return {
            "flow_status": status,
            "final_selection": {
                "status": status,
                "selected_candidate_id": selected_id,
                "selected_reason": "compiled_baseline_candidate" if compiled else "no_compiling_candidate",
                "best_code_candidate": {},
            },
            "flow_events": _event(state, "flow", "finished", final_status=status, selected_candidate_id=selected_id),
        }
