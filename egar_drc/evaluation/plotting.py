"""Separate-process plotting worker for canonical evaluation results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from egar_drc.evaluation.metrics import load_records, outcome, record_summary, summarize

plt.switch_backend("Agg")

_COLORS = {"success": "#2A9D8F", "fail": "#D9AD32", "error": "#D65F49", "unscored": "#8D99AE"}


def _save(figure: Figure, output_dir: Path, name: str, dpi: int) -> None:
    """Save PNG and SVG output and release the figure."""
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def _number(value: Any) -> float | None:
    """Return an observed finite nonnegative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _observed_score(record: dict[str, Any]) -> dict[str, Any]:
    """Exclude unscored and errored records from benchmark distributions."""
    return record["score"] if outcome(record) in ("success", "fail") else {}


def _plot_overview(metrics: dict[str, Any], records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot the shared metrics without legacy run-level performance overlays."""
    benchmark = metrics["benchmark"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    labels = ["Successful", "Failed", "Errors", "Unscored"]
    axes[0, 0].bar(labels, [benchmark[key] for key in ("successful", "failed", "errors", "unscored")], color=list(_COLORS.values()))
    axes[0, 0].set_title("Benchmark outcomes")
    statuses = Counter(str(record.get("flow_status") or "unknown") for record in records)
    axes[0, 1].barh(list(statuses), list(statuses.values()), color="#457B9D")
    axes[0, 1].set_title("Flow status (not benchmark success)")
    rates = [_number(_observed_score(record).get("compile_rate")) for record in records]
    axes[1, 0].hist([rate for rate in rates if rate is not None], bins=20)
    axes[1, 0].set_title("Observed benchmark compile rates")
    axes[1, 1].axis("off")
    rate = benchmark["success_rate"]
    tokens = metrics["tokens"]["total_tokens"]
    lines = [
        f"Records: {metrics['coverage']['records']}",
        f"Scored: {metrics['coverage']['scored']}",
        "Success among scored: " + (f"{rate:.1%}" if rate is not None else "unavailable"),
        "Observed total tokens: " + (f"{tokens:,}" if tokens is not None else "unavailable"),
        f"Token coverage: {metrics['coverage']['token_records']['total_tokens']} records",
    ]
    axes[1, 1].text(0.05, 0.85, "\n".join(lines), va="top", linespacing=1.8)
    _save(figure, output_dir, "01_overview", dpi)


def _plot_rule_resources(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot observed durations and token totals without filling missing values."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for record in records:
        duration = _number(record.get("duration_seconds"))
        usage = record.get("token_usage")
        tokens = _number(usage.get("total_tokens")) if isinstance(usage, dict) else None
        if duration is not None:
            axes[0].scatter(str(record.get("id", "")), duration, color=_COLORS[outcome(record)])
        if duration is not None and tokens is not None:
            axes[1].scatter(duration, tokens, color=_COLORS[outcome(record)])
    axes[0].set_title("Observed per-rule duration (seconds)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[1].set_title("Observed token cost versus duration")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("Tokens")
    _save(figure, output_dir, "02_rule_duration_and_tokens", dpi)


def _plot_token_agents(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot the same role token accounting exposed by summarize."""
    metrics = summarize(records)
    roles = metrics["tokens"]["by_role"]
    if not roles:
        return
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axis = axes[0]
    for field, offset, color in (("prompt_tokens", -0.2, "#457B9D"), ("completion_tokens", 0.2, "#2A9D8F")):
        observed = [(index, values[field]) for index, values in enumerate(roles.values()) if values[field] is not None]
        axis.bar([index + offset for index, _ in observed], [value for _, value in observed], width=0.4, label=field, color=color)
    axis.set_xticks(range(len(roles)), labels=list(roles), rotation=25)
    axis.set_title("Observed tokens by role (missing usage omitted)")
    axis.legend()
    latencies: dict[str, list[float]] = defaultdict(list)
    for record in records:
        requests = record.get("llm_request_metrics")
        for request in requests if isinstance(requests, list) else []:
            if not isinstance(request, dict):
                continue
            latency = _number(request.get("response_time_seconds"))
            if latency is not None:
                latencies[str(request.get("agent") or request.get("role") or "unknown")].append(latency)
    if latencies:
        axes[1].boxplot(list(latencies.values()), tick_labels=list(latencies))
    axes[1].set_title("Observed request latency (seconds)")
    failures = metrics["requests"]["by_role"]
    known = {role: values["failed"] for role, values in failures.items() if values["failed"] is not None}
    axes[2].bar(list(known), list(known.values()), color=_COLORS["error"])
    axes[2].set_title("Explicitly failed requests by role")
    for panel in axes:
        panel.tick_params(axis="x", rotation=25)
    _save(figure, output_dir, "03_llm_agents", dpi)


def _plot_loops(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot observed phase attempts from the shared metrics builder."""
    loops = summarize(records)["loops"]
    known = {phase: values["attempts"] for phase, values in loops.items() if values["attempts"] is not None}
    if not known:
        return
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(list(known), list(known.values()), color="#457B9D")
    axis.tick_params(axis="x", rotation=30)
    axis.set_title("Observed phase attempts (partial coverage possible)")
    _save(figure, output_dir, "04_loop_behavior", dpi)


def _plot_categories(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot only explicitly scored category F1 observations."""
    categories: dict[str, list[float]] = defaultdict(list)
    for record in records:
        scores = _observed_score(record).get("per_category_f1")
        for category, value in (scores.items() if isinstance(scores, dict) else []):
            number = _number(value)
            if number is not None and number <= 1:
                categories[str(category)].append(number)
    if not categories:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.boxplot(list(categories.values()), tick_labels=list(categories))
    axis.set_ylim(0, 1)
    axis.set_title("Observed benchmark category F1")
    _save(figure, output_dir, "05_category_f1", dpi)


def _plot_failures(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot errors and explicit mismatch evidence, not missing-score zeros."""
    errors = Counter(str(record["error"]).split(":", 1)[0] for record in records if record.get("error"))
    mismatches = [len(score["mismatches"]) for record in records if isinstance((score := _observed_score(record)).get("mismatches"), (dict, list))]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(list(errors), list(errors.values()), color=_COLORS["error"])
    axes[0].set_title("Recorded runtime errors")
    axes[1].hist(mismatches, bins=20)
    axes[1].set_title("Observed benchmark mismatch counts")
    _save(figure, output_dir, "06_failures_and_mismatches", dpi)


def _plot_testcase_inventory(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot observed generated testcase inventory without inferring gold validity."""
    summaries = [record_summary(record) for record in records]
    known = [summary for summary in summaries if summary["generated_cases_total"] is not None]
    if not known:
        return
    figure, axis = plt.subplots(figsize=(10, max(4, len(known) * 0.3)))
    axis.barh([summary["id"] for summary in known], [summary["generated_cases_total"] for summary in known])
    axis.set_title("Observed generated testcase counts (not correctness)")
    _save(figure, output_dir, "09_testcase_inventory_by_rule", dpi)


def _plot_execution_evolution(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Plot observed first and last executions, omitting missing counts."""
    summaries = [record_summary(record) for record in records]
    known = [summary for summary in summaries if summary["initial_testcases_fails_percentage"] is not None and summary["last_execution_testcases_fails_percentage"] is not None]
    if not known:
        return
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.hist([summary["initial_testcases_fails_percentage"] for summary in known], bins=20, range=(0, 100), alpha=0.6, label="First execution")
    axis.hist([summary["last_execution_testcases_fails_percentage"] for summary in known], bins=20, range=(0, 100), alpha=0.6, label="Last execution")
    axis.set_title("Joint deck-and-suite evolution")
    axis.set_xlabel("Failing testcases (%)")
    axis.text(0.5, -0.22, "Suites may differ; last execution need not use the selected deck. Not a controlled comparison.", ha="center", transform=axis.transAxes)
    axis.legend()
    _save(figure, output_dir, "11_deck_behavior_on_generated_testcases", dpi)


def _plot_status_outcomes(records: list[dict[str, Any]], output_dir: Path, dpi: int) -> None:
    """Include UNSCORED explicitly in flow status versus benchmark counts."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        counts[str(record.get("flow_status") or "unknown")][outcome(record)] += 1
    if not counts:
        return
    figure, axis = plt.subplots(figsize=(9, max(4, len(counts) * 0.4)))
    outcomes = list(_COLORS)
    matrix = [[values[key] for key in outcomes] for values in counts.values()]
    axis.imshow(matrix, aspect="auto", cmap="YlGnBu")
    axis.set_xticks(range(len(outcomes)), labels=["PASS", "FAIL", "ERROR", "UNSCORED"])
    axis.set_yticks(range(len(counts)), labels=list(counts))
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center")
    axis.set_title("Flow status versus benchmark outcome")
    _save(figure, output_dir, "10_flow_status_vs_benchmark_outcome", dpi)


def render_run(run_dir: Path, output_dir: Path, dpi: int = 180) -> list[Path]:
    """Render canonical results in the optional plotting worker.

    The pipeline verifies freshness before invoking this worker. Reading only
    canonical records preserves stage-2 overrides rather than re-enriching from
    historical flow artifacts; direct invocation does not verify the manifest.

    Args:
        run_dir: Run root containing evaluation/results.json, or evaluation itself.
        output_dir: Destination for plots and rule_flow_summary.csv.
        dpi: PNG resolution.

    Returns:
        PNG paths produced by this invocation. SVG counterparts are also saved.
    """
    records = load_records(Path(run_dir))
    metrics = summarize(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    _plot_overview(metrics, records, output_dir, dpi)
    for plot in (_plot_rule_resources, _plot_token_agents, _plot_loops, _plot_categories, _plot_failures, _plot_testcase_inventory, _plot_execution_evolution, _plot_status_outcomes):
        plot(records, output_dir, dpi)
    summaries = [record_summary(record) for record in records]
    with (output_dir / "rule_flow_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]) if summaries else ["id"])
        writer.writeheader()
        writer.writerows(summaries)
    return sorted(output_dir.glob("*.png"))


def main() -> int:
    """Run the internal plotting subprocess, not a separate public command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    render_run(args.run_dir, args.output or args.run_dir / "plots", dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())