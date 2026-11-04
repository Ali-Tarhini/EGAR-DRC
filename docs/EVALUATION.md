# Evaluation

## Scoring

A problem counts as solved only if the final script reproduces the gold result
on every benchmark layout for that rule. There is no partial credit.

Scoring happens twice:

- **During generation**, each problem is scored as soon as its flow finishes.
  The result is recorded in `results.jsonl` and `summary.json`.
- **With `egar-drc-evaluate`**, after a run finishes, the saved scripts are
  re-scored and the run's metrics are computed.

## Running the Evaluation

```sh
egar-drc-evaluate results/egar --repository data/Rule2DRC
```

The command runs three steps by default; select them with `--steps`:

| Step | What it does |
| --- | --- |
| `benchmark` | Re-scores every final script against the gold layouts. |
| `testcases` | Checks the final generated test suite against the gold script, measuring how many generated labels were correct. |
| `metrics` | Summarizes success rate, tokens, requests, and loop counts. |
| `plots` | Draws charts from the metrics. Requires `python -m pip install '.[plots]'`. |

`--steps all` runs all four. Steps whose inputs are unchanged are reused on the
next invocation; `--force` recomputes them.

With the scripts, `sh scripts/02_evaluate/run_evaluation.sh` evaluates
`results/egar`. Pass another run directory as its first argument to evaluate a
different run.

## Outputs

Generation writes to the run directory:

| File | Contents |
| --- | --- |
| `results.jsonl` | One record per problem: status, score, tokens, and timing. |
| `summary.json` | Totals for the run. |
| `run.log` | Progress and any agent failures. |
| `run_metadata.json` | Code, prompt, model, and environment details. |
| `<problem>/` | The final script, the full flow record, and the generated test suite. |

Evaluation writes to `<run>/evaluation/`:

| File | Contents |
| --- | --- |
| `metrics.json` | The run's metrics, defined below. |
| `results.json` | Per-problem records combining generation and evaluation results. |
| `benchmark.json`, `testcases.json` | Per-problem results of each step. |
| `benchmark/`, `testcases/` | Detailed KLayout outputs per problem. |
| `plots/` | Charts, when requested. |
| `manifest.json` | Status and input fingerprints of each step. |

## Metrics

**`summary.json`**

| Field | Meaning |
| --- | --- |
| `success_rate` | Solved problems divided by completed problems. |
| `wall_time_seconds_total` | Elapsed time of the run, across all parallel jobs. |
| `problem_duration_seconds_mean` | Average time spent on one problem. |
| `token_usage` | Prompt, completion, and total tokens reported by the model server. |
| `tokens_per_success` | Total tokens divided by solved problems. |

When a run is resumed, the record counts cover every problem, while
`wall_time_seconds_total` covers only the latest invocation.

**`evaluation/metrics.json`**

| Field | Meaning |
| --- | --- |
| `benchmark.success_rate` | Solved problems divided by scored problems. |
| `benchmark.cohort_success_rate` | Solved problems divided by all problems, counting errors as unsolved. |
| `tokens` | Token totals, overall and per agent. |
| `requests` | Model requests, overall and per agent, including failed ones. |
| `loops` | Attempts per loop, such as compilations and critic reviews. |
| `testcases.accuracy` | Fraction of generated test labels that agree with the gold script. |

A value of `null` means the information was not available for that run.
