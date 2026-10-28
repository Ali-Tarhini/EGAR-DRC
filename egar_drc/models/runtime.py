"""Shared native LangChain model resources for EGAR roles."""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict
from uuid import UUID

import httpx
from egar_drc.models.config import ModelConfig
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


class CompatibleChatOpenAI(ChatOpenAI):
    """Omit empty tool lists for provider compatibility."""

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Build the request payload without an empty tool list."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if payload.get("tools") == []:
            payload.pop("tools")
        return payload


class RequestUsage(AsyncCallbackHandler):
    """Capture model-call usage independently of role validation."""

    def __init__(self) -> None:
        """Initialize invocation-local measurements."""
        self.calls: list[dict[str, Any]] = []
        self.started: dict[UUID, float] = {}

    async def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Start timing one actual model request."""
        self.started[run_id] = time.perf_counter()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Capture standard LangChain message usage and provider metadata."""
        message = response.generations[0][0].message
        metadata = message.response_metadata
        raw = metadata.get("token_usage", {})
        prompt = int(raw.get("prompt_tokens", 0) or 0)
        completion = int(raw.get("completion_tokens", 0) or 0)
        self.calls.append(
            {
                "model": metadata.get("model_name", metadata.get("model", "")),
                "provider_request_id": message.id,
                "finish_reason": metadata.get("finish_reason"),
                "duration_seconds": time.perf_counter() - self.started.pop(run_id, time.perf_counter()),
                "token_usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": int(raw.get("total_tokens", 0) or 0),
                },
                "success": True,
            }
        )

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """Record a callback-visible failure without inventing token counts."""
        self.calls.append(
            {
                "duration_seconds": time.perf_counter() - self.started.pop(run_id, time.perf_counter()),
                "token_usage": {},
                "success": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )

    def update(self) -> dict[str, Any]:
        """Aggregate reported token usage and retain per-call evidence."""
        usage = {
            key: sum(int(call["token_usage"].get(key, 0)) for call in self.calls)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return {"last_token_usage": usage, "model_call_metric": {"model_calls": list(self.calls)}}


class RoleContext(TypedDict):
    """Invocation-local measurements and model settings for middleware."""

    usage: RequestUsage
    settings: dict[str, Any]


class RequestLimit(AgentMiddleware[AgentState, RoleContext]):
    """Limit concurrent model calls, not tool execution or entire role runs."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        """Retain the runtime-wide request gate."""
        self.semaphore = semaphore

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> ModelResponse:
        """Attach invocation-local callbacks and limit concurrent model calls."""
        context = request.runtime.context or {}
        model = request.model.model_copy(update={"callbacks": [context["usage"]]})
        request = request.override(
            model=model, model_settings={**request.model_settings, **context.get("settings", {})}
        )
        async with self.semaphore:
            return await handler(request)


class ModelRuntime:
    """Own a shared chat model, checkpoint saver, and request limit."""

    def __init__(
        self, model: BaseChatModel, *, max_concurrency: int = 16, saver: InMemorySaver | None = None,
    ) -> None:
        """Accept real LangChain models, including deterministic test models."""
        self.model = model
        self.saver = saver if saver is not None else InMemorySaver()
        self.request_limit = RequestLimit(asyncio.Semaphore(max_concurrency))
        self._http_async: httpx.AsyncClient | None = None
        self._http_sync: httpx.Client | None = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> ModelRuntime:
        """Build the provider adapter and its connection pools."""
        limits = httpx.Limits(
            max_connections=config.max_concurrency,
            max_keepalive_connections=config.max_concurrency,
        )
        http_async = httpx.AsyncClient(limits=limits)
        http_sync = httpx.Client(limits=limits)
        additional_parameters = dict(config.additional_parameters)
        reasoning_effort = additional_parameters.pop("reasoning_effort", config.reasoning_effort)
        reasoning_parameters = {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}
        model = CompatibleChatOpenAI(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_seconds,
            max_retries=config.retry_attempts - 1,
            model_kwargs=additional_parameters,
            **reasoning_parameters,
            http_async_client=http_async,
            http_client=http_sync,
            http_socket_options=(),
        )
        runtime = cls(model, max_concurrency=config.max_concurrency)
        runtime._http_async = http_async
        runtime._http_sync = http_sync
        return runtime

    async def release_run(self, run_id: str) -> None:
        """Delete role-thread lineages under this run's slash-delimited prefix."""
        prefix = f"{run_id}/"
        for thread_id in list(self.saver.storage):
            if thread_id.startswith(prefix):
                await self.saver.adelete_thread(thread_id)

    async def close(self) -> None:
        """Release owned connections and remaining in-memory checkpoints."""
        for thread_id in list(self.saver.storage):
            await self.saver.adelete_thread(thread_id)
        if self._http_async is not None:
            await self._http_async.aclose()
        if self._http_sync is not None:
            self._http_sync.close()
