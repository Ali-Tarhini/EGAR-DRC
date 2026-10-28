# Architecture

EGAR-DRC synthesizes a KLayout DRC script for a natural-language design rule by
iterating on one candidate rather than sampling many. Every candidate script and
every generated test suite is checked by running KLayout, and the results decide
what happens next.

![EGAR-DRC flow](../egar_drc/EGAR-DRC.png)

## The Repair Loop

The flow combines five LLM agents with deterministic KLayout steps.

1. **Generate the script.** The code-generation agent writes a Ruby DRC script.
   KLayout compiles it; compiler errors go back to the agent until the script
   compiles. If it still fails after several attempts, generation restarts
   from a fresh conversation.
2. **Generate tests.** The testcase-generation agent writes a Python program that
   builds GDS layouts and records the result each layout should produce. The
   program is executed to materialize the test suite; failures are reported back
   to the agent.
3. **Execute.** The script runs on every generated layout, and each result is
   compared with its expected label.
4. **Adjudicate.** The critic reviews the rule, the script, the tests, and the
   execution results. A mismatch can mean the script is wrong, the test is
   wrong, or both, so the critic can request a script correction, a test
   correction, both, or none.
5. **Repair.** The code-correction and testcase-correction agents apply the
   critic's instructions. Corrected scripts are recompiled and corrected tests
   are regenerated, then the suite is executed and reviewed again.

The loop ends when the critic accepts a script that passes every generated test,
or when a budget is exhausted. In the second case, the flow returns the compiled
candidate that passed the most generated tests.

The benchmark's gold scripts and labels are never shown to the agents. They are
used only to score the final script after the flow ends.

## Flows

| `--flow` | What it does |
| --- | --- |
| `egar-drc` | The full method: the critic can repair the script, the tests, or both. |
| `code_only_repair` | The same loop, but the generated tests are fixed; the critic can only request script corrections. |
| `code_generation_baseline` | Script generation with compiler feedback, without tests or a critic. |
| `single_sample_baseline` | One generation call, without any feedback. |

`code_only_repair` isolates the effect of repairing the tests, and the two
baselines isolate the effect of execution feedback.

## Budgets

Each loop is bounded. The defaults live in
[flow_config.yaml](../egar_drc/runtime/flow_config.yaml):

| Setting | Default | Bounds |
| --- | --- | --- |
| `max_code_generation_compile_iterations` | 5 | Compile attempts per generation conversation. |
| `max_code_generation_retry_iterations` | 20 | Fresh generation conversations. |
| `max_testcase_generation_prepare_iterations` | 20 | Attempts to produce a valid initial test suite. |
| `max_critic_review_iterations` | 10 | Critic reviews. |
| `max_code_correction_iterations` | 20 | Script corrections. |
| `max_code_correction_compile_iterations` | 5 | Compile attempts per script correction. |
| `max_testcase_correction_iterations` | 20 | Test-suite corrections. |
| `max_testcase_correction_prepare_iterations` | 20 | Attempts to materialize a corrected suite. |
| `max_testcases` | 16 | Generated layouts per suite. |

## Implementation

The outer loop is a [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph`. Agents are nodes, KLayout steps are plain functions, and routing
between them is an explicit conditional edge. The graph state records every
candidate, test suite, execution result, and model request.

Each agent is a LangChain agent sharing one OpenAI-compatible chat model. An
agent keeps its own conversation for the duration of a problem, so a correction
builds on its earlier attempts; a regeneration starts a new conversation. All
agents share the same system prompt and reference-document prefix, which lets
servers such as vLLM reuse the prompt cache across agents. The critic replies in
JSON, which is validated locally; an invalid reply counts as a failed review.

Problems run concurrently. `--jobs` sets how many problems run at once, and
`max_concurrency` in the model configuration caps simultaneous model requests.

## Code Layout

| Path | Contents |
| --- | --- |
| [egar_drc/agents](../egar_drc/agents) | One package per agent, each with its implementation and prompt YAML. |
| [egar_drc/orchestration](../egar_drc/orchestration) | The four flows. |
| [egar_drc/models](../egar_drc/models) | Model configuration and the shared chat model. |
| [egar_drc/runtime](../egar_drc/runtime) | The batch runner, logging, and run metadata. |
| [egar_drc/evaluation](../egar_drc/evaluation) | Scoring, test-suite audits, metrics, and plots. |
| [egar_drc/prepare_dataset.py](../egar_drc/prepare_dataset.py) | Benchmark download and preparation. |
| [egar_drc/reference](../egar_drc/reference) | The KLayout DRC reference given to the agents. |

## Reproducibility

Every run writes `run_metadata.json` with the package and prompt hashes, the
problem specification hashes, the model and decoding settings, the flow budgets,
and the Python, dependency, and KLayout versions. Credentials are never
recorded. Use a separate output directory for each configuration.
