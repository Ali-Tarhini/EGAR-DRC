"""Shared graph-node helpers for EGAR-DRC agents."""

import time
from typing import Any


def has_passing_execution(state: dict[str, Any]) -> bool:
    """Require successful preparation and a nonempty, fully passing execution."""
    report = state.get("test_execution_report", {})
    if not isinstance(report, dict):
        return False
    n_cases = int(report.get("n_cases", 0) or 0)
    return (
        bool(state.get("compilation_succeeded", False))
        and bool(state.get("testcase_generation_succeeded", False))
        and bool(report.get("success", False))
        and not report.get("precheck")
        and n_cases > 0
        and int(report.get("n_passing_cases", 0) or 0) == n_cases
    )


def append_flow_event(
    state: dict[str, Any],
    event_phase: str,
    event_status: str,
    **payload: Any,
) -> list[dict[str, Any]]:
    """Append one serializable flow event.

    Args:
        state: Current flow state.
        event_phase: Flow phase name.
        event_status: Event status.
        payload: Additional JSON-serializable event fields.

    Returns:
        Full updated event list.
    """
    events = [dict(record) for record in state.get("flow_events", [])]
    event = {
        **payload,
        "event_index": len(events) + 1,
        "phase": event_phase,
        "status": event_status,
    }
    events.append(event)
    return events


def append_loop_metric(
    state: dict[str, Any],
    *,
    phase: str,
    started_at: float,
    success: bool,
    outcome: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one deterministic or agent-node attempt metric.

    Args:
        state: Current flow state.
        phase: Metric phase name.
        started_at: ``time.perf_counter`` value captured before the attempt.
        success: Whether the attempt succeeded.
        outcome: Compact outcome metadata.

    Returns:
        Updated loop metrics mapping.
    """
    metrics = dict(state.get("loop_metrics", {}))
    phase_metrics = dict(metrics.get(phase, {}))
    durations = [float(value) for value in phase_metrics.get("durations_seconds", [])]
    duration = round(time.perf_counter() - started_at, 6)
    durations.append(duration)
    attempts = int(phase_metrics.get("attempts", 0)) + 1
    successes = int(phase_metrics.get("successes", 0)) + int(success)
    attempt_records = [dict(record) for record in phase_metrics.get("attempt_records", [])]
    attempt_records.append(
        {
            "attempt": attempts,
            "success": bool(success),
            "duration_seconds": duration,
            **dict(outcome or {}),
        }
    )
    phase_metrics.update(
        {
            "attempts": attempts,
            "successes": successes,
            "failures": attempts - successes,
            "durations_seconds": durations,
            "total_duration_seconds": round(sum(durations), 6),
            "attempt_records": attempt_records,
        }
    )
    metrics[phase] = phase_metrics
    return metrics


def append_llm_metric(
    state: dict[str, Any],
    *,
    agent: str,
    started_at: float,
    success: bool,
    update: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> list[dict[str, Any]]:
    """Append one flow-level LLM request metric.

    Args:
        state: Current flow state.
        agent: Agent metric name.
        started_at: ``time.perf_counter`` value captured before the request.
        success: Whether the request succeeded.
        update: Completed response or agent update, even if validation failed.
        error: Provider or parsing exception when the request failed.

    Returns:
        Updated LLM request metrics list.
    """
    records = [dict(record) for record in state.get("llm_request_metrics", [])]
    # Retain captured usage even when validation fails.
    token_usage = dict((update or {}).get("last_token_usage", {}))
    record: dict[str, Any] = {
        **dict((update or {}).get("model_call_metric", {})),
        "request_index": len(records) + 1,
        "agent": agent,
        "success": bool(success),
        "response_time_seconds": round(time.perf_counter() - started_at, 6),
        "token_usage": token_usage,
    }
    if error is not None:
        error_record: dict[str, Any] = {
            "type": type(error).__name__,
            "message": str(error),
            "module": type(error).__module__,
        }
        for attr_name in ("status_code", "code", "request_id"):
            attr_value = getattr(error, attr_name, None)
            if attr_value is not None:
                error_record[attr_name] = attr_value
        record["error"] = error_record
    records.append(record)
    return records


def owned_update(update: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Keep only graph fields owned by the current agent node.

    Args:
        update: Full agent update.
        keys: State keys owned by the agent node.

    Returns:
        Narrow graph-state delta.
    """
    return {key: update[key] for key in keys if key in update}


def graph_token_usage_update(update: dict[str, Any]) -> dict[str, Any]:
    """Return model usage and run-scoped generation deltas for graph state.

    The token reducer adds updates, so emit only this invocation's last usage,
    never a flow-wide total or an inherited history snapshot.

    Args:
        update: Agent-local state containing normalized token tracking fields.

    Returns:
        Graph update for the additive ``token_usage_total`` channel.
    """
    last_usage = update.get("last_token_usage", {})
    graph_update: dict[str, Any] = {
        "token_usage_total": {
            str(key): int(value)
            for key, value in last_usage.items()
            if isinstance(last_usage, dict)
            if isinstance(value, (int, float))
        }
    }
    graph_update["role_generations"] = dict(update.get("role_generations", {}))
    return graph_update
