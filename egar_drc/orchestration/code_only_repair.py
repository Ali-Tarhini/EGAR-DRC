"""Code-only repair: correct the DRC script against a fixed generated test suite."""

from typing import Any, Literal, cast

from egar_drc.agents import CodeCorrectionAgent, CodeGenerationAgent, TestcaseGenerationAgent
from egar_drc.agents.fixed_testcase_adjudication.agent import FixedTestcaseAdjudicationAgent
from egar_drc.models.runtime import ModelRuntime
from egar_drc.orchestration.egar_drc import EgarDrcFlow, EgarDrcState
from langgraph.graph import END, START, StateGraph


class EgarDrcCodeOnlyRepairFlow(EgarDrcFlow):
    """Evaluate and correct code while keeping prepared testcases fixed."""

    assistant_id = "egar_drc_code_only_repair"

    def __init__(self, runtime: ModelRuntime) -> None:
        """Initialize the four roles used by the code-only repair graph."""
        self.runtime = runtime
        self._testcase_generation_agent = TestcaseGenerationAgent(runtime)
        self._code_generation_agent = CodeGenerationAgent(runtime)
        self._fixed_testcase_adjudication_agent = FixedTestcaseAdjudicationAgent(runtime)
        self._code_correction_agent = CodeCorrectionAgent(runtime)
        self._graph = self.make_graph()

    def make_graph(self) -> Any:
        """Build code repair without any testcase correction or restore node."""
        builder = StateGraph(EgarDrcState)
        builder.add_node("initialize", self.initialize)
        builder.add_node("generate_code", cast(Any, self._code_generation_agent.invoke))
        builder.add_node("compile_code", self.compile_code)
        builder.add_node("generate_testcases", cast(Any, self._testcase_generation_agent.invoke))
        builder.add_node("prepare_testcases", self.prepare_testcases)
        builder.add_node("execute_testcases", self.execute_testcases)
        builder.add_node("adjudicate_testcases", cast(Any, self._fixed_testcase_adjudication_agent.invoke))
        builder.add_node("correct_code", cast(Any, self._code_correction_agent.invoke))
        builder.add_node("finalize", self.finalize)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "generate_code")
        builder.add_edge("generate_code", "compile_code")
        builder.add_conditional_edges(
            "compile_code",
            self.route_after_compile,
            {
                "generate_code": "generate_code",
                "correct_code": "correct_code",
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
                "execute_testcases": "execute_testcases",
                "adjudicate_testcases": "adjudicate_testcases",
            },
        )
        builder.add_edge("execute_testcases", "adjudicate_testcases")
        builder.add_conditional_edges(
            "adjudicate_testcases",
            self.route_after_critic,
            {
                "correct_code": "correct_code",
                "adjudicate_testcases": "adjudicate_testcases",
                "finalize": "finalize",
            },
        )
        builder.add_edge("correct_code", "compile_code")
        builder.add_edge("finalize", END)
        return builder.compile()

    @staticmethod
    def route_after_prepare(
        state: EgarDrcState,
    ) -> Literal["generate_testcases", "execute_testcases", "adjudicate_testcases"]:
        """Retry initial preparation, never repair a generated suite."""
        if bool(state.get("testcase_generation_succeeded", False)):
            return "execute_testcases"
        if int(state.get("testcase_generation_prepare_iteration", 0)) < int(
            state.get("max_testcase_generation_prepare_iterations", 5)
        ):
            return "generate_testcases"
        return "adjudicate_testcases"

    @staticmethod
    def route_after_compile(
        state: EgarDrcState,
    ) -> Literal[
        "generate_code",
        "correct_code",
        "generate_testcases",
        "adjudicate_testcases",
        "execute_testcases",
        "finalize",
    ]:
        """Compile repaired code without entering testcase mutation paths."""
        if bool(state.get("compilation_succeeded", False)):
            if str(state.get("code_origin", "")) == "code_correction":
                return (
                    "execute_testcases"
                    if bool(state.get("testcase_generation_succeeded", False))
                    else "adjudicate_testcases"
                )
            return "generate_testcases"
        if str(state.get("code_origin", "")) == "code_correction":
            # Return exhausted compiler retries to code-only adjudication.
            if int(state.get("code_correction_compile_iteration", 0)) < int(
                state.get("max_code_correction_compile_iterations", 15)
            ):
                return "correct_code"
            return (
                "adjudicate_testcases"
                if int(state.get("critic_review_iteration", 0)) < int(state.get("max_critic_review_iterations", 3))
                else "finalize"
            )
        if int(state.get("code_regeneration_iteration", 0)) > int(state.get("max_code_generation_retry_iterations", 5)):
            return "finalize"
        return "generate_code"

    @staticmethod
    def route_after_critic(state: EgarDrcState) -> Literal["correct_code", "adjudicate_testcases", "finalize"]:
        """Honor code-only instructions and upstream correction limits."""
        if not bool(state.get("critic_request_succeeded", True)):
            return (
                "adjudicate_testcases"
                if int(state.get("critic_review_iteration", 0)) < int(state.get("max_critic_review_iterations", 3))
                else "finalize"
            )
        review = state.get("critic_review", {})
        instruction = str(review.get("code_correction_instruction", "")).strip() if isinstance(review, dict) else ""
        if not instruction:
            return "finalize"
        if int(state.get("critic_review_iteration", 0)) >= int(state.get("max_critic_review_iterations", 3)):
            return "finalize"
        if int(state.get("code_correction_iteration", 0)) > int(state.get("max_code_correction_iterations", 5)):
            return "finalize"
        return "correct_code"
