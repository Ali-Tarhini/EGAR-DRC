"""Descriptor-driven role adapters over native LangChain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from pathlib import Path
from typing import Any, TypedDict

from egar_drc.agents.descriptor import AgentDescriptor, load_descriptor
from egar_drc.models.runtime import ModelRuntime, RequestUsage, RoleContext
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class RoleInput(TypedDict):
    """Message delta accepted by the native agent graph."""

    messages: list[AnyMessage | dict[str, Any]]


class RoleInvocationError(RuntimeError):
    """Carry captured request usage through invocation or validation failures."""

    def __init__(self, cause: Exception, update: dict[str, Any]) -> None:
        """Retain the original failure and completed request measurements."""
        super().__init__(str(cause))
        self.cause = cause
        self.update = update


class RoleAgent(ABC):
    """Adapt role prompts to create_agent without owning conversation data."""

    reset_state_key: str | None = None

    def __init__(
        self, runtime: ModelRuntime, descriptor_path: str | Path, *, tools: list[BaseTool] | None = None
    ) -> None:
        """Compile the role with shared model, saver, and request middleware."""
        self.runtime = runtime
        self.descriptor = load_descriptor(descriptor_path)
        # Provider-native constrained output is disabled due to observed vLLM latency.
        # With supporting providers, native schema output is an alternative to
        # local parsing and corrective prompting.
        self.agent = create_agent(
            runtime.model,
            tools=tools or [],
            system_prompt=self.descriptor.system_prompt,
            middleware=[runtime.request_limit],
            context_schema=RoleContext,
            checkpointer=runtime.saver,
            name=self.descriptor.agent_name,
        )

    @abstractmethod
    def extract_initial_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """Map flow state to descriptor variables."""

    def build_followup_message(self, state: dict[str, Any]) -> str | None:
        """Build a delta-only continuation message for a re-entered agent."""
        del state
        return None

    def output_model(self) -> type[BaseModel] | None:
        """Return the optional structured output model."""
        return None

    @staticmethod
    def _parse_structured_output(raw_output: str, output_model: type[BaseModel]) -> BaseModel:
        """Validate JSON directly or inside the original supported Markdown fences."""
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        candidates = [text, *re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_output, re.DOTALL | re.IGNORECASE)]
        for candidate in candidates:
            try:
                return output_model.model_validate_json(candidate.strip())
            except ValueError:
                continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}") from exc
        return output_model.model_validate({"resolved_items": loaded} if isinstance(loaded, list) else loaded)

    @abstractmethod
    async def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return domain state asynchronously, unlike LangChain's sync invoke."""

    async def _invoke_model(self, state: dict[str, Any]) -> dict[str, Any]:
        """Continue a role checkpoint or start a fresh generation."""
        run_id = str(state["run_id"])
        role = self.descriptor.agent_name
        generation = int(state.get("role_generations", {}).get(role, 0))
        reset = self.reset_state_key is not None and state.get(self.reset_state_key, False)
        generation += int(bool(reset))
        usage = RequestUsage()
        # Isolate checkpoint history by run, role, and generation.
        config: RunnableConfig = {"configurable": {"thread_id": f"{run_id}/{role}/{generation}"}}
        checkpoint = await self.agent.aget_state(config)
        # Append fresh evidence to existing conversations.
        prompt = (
            self.build_followup_message(state)
            if checkpoint.values.get("messages")
            else self.descriptor.render_user_prompt(self.extract_initial_input(state))
        )
        agent_input: RoleInput = {"messages": [HumanMessage(content=prompt)] if prompt else []}
        structured: BaseModel | None = None
        try:
            settings = dict(self.descriptor.additional_model_parameters)
            if self.descriptor.model is not None:
                settings["model"] = self.descriptor.model
            if self.descriptor.temperature is not None:
                settings["temperature"] = self.descriptor.temperature
            schema = self.output_model()
            result = await self.agent.ainvoke(
                agent_input, config=config, context={"usage": usage, "settings": settings}
            )
            if schema is not None:
                structured = self._parse_structured_output(result["messages"][-1].text, schema)
            else:
                structured = None
        except Exception as exc:
            # Preserve captured usage when validation fails.
            raise RoleInvocationError(exc, {**usage.update(), "role_generations": {role: generation}}) from exc
        return {
            **usage.update(),
            "role_generations": {role: generation},
            "final_output": result["messages"][-1].text,
            "parsed_output": structured,
        }
