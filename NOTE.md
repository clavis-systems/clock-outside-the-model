# Temporal Blindness Is an Architecture Problem: A Clock Outside the Model Beats LLMs and Post-Training on TicToc

*Draft v1, 7 October 2026. Emanuele Rizzan, independent researcher. AI assistance: Claude (Anthropic)
implemented the experiments and drafted the text. Codex (OpenAI) independently re-implemented and recomputed the two
clock rules (single threshold and per-scenario TTL) from the raw data without reading our code. Its reports corrected
a scenario count and added scenario-level intervals. The local model runs and the threshold sensitivity were not
recomputed independently.*

## Abstract

LLM agents ignore the real time that passes between turns when they decide whether to call a tool again: this has
been called *temporal blindness*.

**The benchmark.** TicToc (Findings of ACL 2026) measures temporal blindness on 1,864 trajectories with human
preferences.
- With timestamps in context, no model exceeds 65% normalized alignment.
- Prompting helps little.
- DPO post-training raises 8B models to about 75%.

**The missing baseline.** We report a baseline absent from that study: a rule *outside* the model that calls the tool
again when its last result is older than 30 minutes.
- The rule was chosen on the training split only and frozen before evaluation.
- On the test split (26 scenarios not used for training, 1,069 cases) it reaches **96.3%**: 95% CI 93.7–98.1%,
  resampling whole scenarios. On the full data it reaches 94.5%.
- Any threshold between 5 and 60 minutes gives the same test result.

**Our reproduction.** Two local models evaluated with timestamps reach gemma4 61.0% and Qwen3-8B 56.9%, in line with the paper.

**Per-tool freshness.** When a 4B local model labels, once per scenario, how fast the tools' data age, per-class
thresholds raise the test score to **97.6%**.

**Why it works.** The human preferences in TicToc largely follow the elapsed-time scale, and models do not use that
scale.

**Proposal.** Freshness should be enforced by the agent host, not inferred by the model. The Model Context Protocol
already defines a `ttlMs` freshness hint, but only for list and resource results. We propose extending it to tool call
results.

## 1. Introduction

An agent that fetched a stock price four hours ago should fetch it again before answering. One that fetched it four
seconds ago should not.

Cheng et al. built TicToc to measure whether agents make this decision the way people do [1]. Their findings:
- Models perform near chance without timestamps, and below 65% with them.
- A reminder in the prompt has little effect.
- Rules given as examples help only reasoning models.
- DPO on part of the data brings 8B models to about 75% on the test split.

The study reads this as a gap in how models perceive time, to be closed by alignment.

We ask a simpler question first: how well does a rule do that never looks at the content, only at the clock? The
answer changes what the benchmark tells us. The decision TicToc asks for can be made almost perfectly by code, so the
failure is not that the task is hard. The failure is that agents leave to the model a computation that the host can do
exactly.

## 2. TicToc in brief

- **Trajectories.** Multi-turn conversations in 76 scenarios. Each ends with a user message that could be answered
  from an earlier tool result or by calling the tool again.
- **Time variants.** Each trajectory appears with three elapsed-time levels (`tv1`, `tv2`, `tv3`). Their scales depend
  on how time-sensitive the scenario is, from seconds to months [1].
- **Labels.** Human preferences, "call the tool" or "answer directly", with a preference score from 0 to 3.
- **Metric.** As in the authors' `get_metric.py`:
  - cases with an "any" preference, or with a score strictly between 0.5 and 2.5, are skipped;
  - the label is "direct" if the score is at most 0.5, and "tool" if it is at least 2.5;
  - the normalized alignment rate is 0.5 · (TP/(TP+FN) + TN/(TN+FP)), that is, balanced accuracy, where TP means
    calling the tool when people prefer it.
- **Splits.** The authors' train and test splits are disjoint.
  - They contain 50 and 26 scenario source files, 76 in all.
  - No scenario, trajectory or timestamp-free history appears in both.
  - Codex verified this independently.

## 3. A clock outside the model

**The rule:** call the tool again if more than 1,800 s have passed since its most recent result, or if there is no
result in context. Otherwise, answer from context.

**Protocol.**
- The threshold was chosen on the training split only.
  - Every threshold between about 400 s and 3,600 s gives the same training score, 93.6%.
  - 1,800 s is the centre of that range.
- The rule, the metric and a success criterion were committed before the test split was opened. The criterion: the
  lower bound of the 95% bootstrap interval on test must exceed 0.76, the best reported post-trained result.
- The test split was evaluated once.
- The data are the Hugging Face release [2], with SHA-256 of the test parquet `8e9e2927…` and of the train parquet
  `13d75921…`, cross-checked against the authors' JSON splits: same identifiers, and labels equal up to floating-point
  rounding.

## 4. Results

| System | Setting | Normalized alignment |
|---|---|---|
| 18 open and proprietary models [1] | with timestamps, full data | ≤ 65% (best: o3 ≈ 65%) |
| Llama-3.1-8B, Ministral-8B, Qwen3-8B after DPO [1] | test split | ≈ 73–76% |
| gemma4 e4b (Ollama, Q4), ours | with timestamps, test split (1,067 cases) | 61.0% |
| Qwen3-8B (Ollama, Q4), ours | with timestamps, test split (1,067 cases) | 56.9% |
| **Clock rule, 1,800 s** | test split (1,069 cases) | **96.3%** (93.7–98.1 by scenario; 95.2–97.3 by case) |
| Clock rule, 1,800 s, scores of exactly 0.5 and 2.5 excluded | test split (1,037 cases) | 96.8% |
| **Clock rule, TTL per scenario inferred by a 4B local model** | test split (1,069 cases) | **97.6%** (22 decisions gained, 2 lost vs the single threshold) |
| Clock rule, 1,800 s | full data (3,016 cases) | 94.5% |

Paper values marked ≈ are read from the paper's figures.

- **By elapsed-time level (test).**
  - `tv1`: 553 of 553 "direct" correct.
  - `tv3`: 360 of 362 "tool" correct.
  - `tv2`: 75.2% balanced; this is the hard part.
- **The signature of temporal blindness (gemma4, test).** gemma4 calls the tool again in 23.5% of the cases a few
  minutes after the last result (130 of 553). Hours to months later it does so in only 46.8% (169 of 361). Elapsed time
  barely moves its decision; the clock rule calls again in 0% and 100% of those cases.
- **Threshold sensitivity (test, descriptive).** Thresholds from 300 to 3,600 s give 96.3%. 7,200 s gives 91.3%;
  60 s or ≥ 21,600 s give 66–72%.
  - The plateau has a simple cause: no evaluable test case has a last-result age between 400 and 3,600 s.
  - The rule therefore separates two well-separated bands. It does not show fine sensitivity to expiry.
- **Scoring conventions.**
  - We follow the authors' `get_metric.py`, which counts scores of exactly 0.5 and 2.5.
  - Excluding them gives 96.8% (1,037 cases).
  - The scenario-level interval is wider than the case-level one, because time variants of a trajectory are
    correlated.

## 5. Why a clock is enough here, and where it is not

TicToc's preferences track the elapsed-time scale. On the training split:
- all 984 short-gap cases are "direct";
- 609 of 618 long-gap cases are "tool".

The scenarios were built so that "short" and "long" are separated by orders of magnitude within each sensitivity
class [1]. A fixed threshold between those scales separates almost every case. Models given the same timestamps do
not make this comparison.

The residual errors sit in the medium level, where the information type matters. On training, knowing each
scenario's sensitivity class and using one threshold per class would raise the score from 93.6% to 96.6%. This is an
upper bound: the class was derived from the benchmark's design.

That is the case for freshness declared per tool: the developer of a stock-price tool knows its data age in seconds,
while a statute lookup ages in months.

**Inferred freshness per tool.** We tested this with an automatic stand-in for a server-declared TTL.
- **Labels.** A small local model (gemma4 e4b) read each scenario's system prompt and tool descriptions once. It never
  saw preferences, scores, identifiers or timestamps. It classified how fast the data age: seconds, hours, days or
  months.
- **Thresholds.** One per class, fitted on the training split: 30 min for seconds and hours, 2 h for days, 1 week for
  months. Labels and thresholds were committed before testing.
- **Result on test.** 97.6% against 96.3% for the single threshold. Of the decisions that changed, 22 became correct
  and 2 became wrong; unnecessary tool calls fell from 40 to 18.

A model is useful here for a micro-judgment it makes once per tool, not for the arithmetic it fails at.

**How far can per-tool freshness go? (descriptive)**

| What the system knows | Test score |
|---|---|
| Nothing outside the model (gemma4, Qwen3-8B with timestamps) | 57–61% |
| One default TTL for every tool | 96.3% |
| A TTL class per scenario, inferred by a 4B model | 97.6% |
| The exact time scale of each scenario: ceiling, majority label per scenario and time level, in-sample | 99.5% |

The gap between 97.6% and 99.5% is the value of knowing each tool's own freshness exactly. The tool's author has that
knowledge, and the proposal in Section 6 asks the server to declare it.

**A negative result.** We also tried four yes/no questions to the small model: can others take what the tool shows?
does the data never change? is the user asking about the assistant's own completed action? is the user about to pay
now?
- Used as hard overrides of the clock, they lowered the training score to 93.2% (cross-validation 92.5%).
- Used as multipliers of the class TTL, they did not beat it in cross-validation: 96.7% against 96.9%.
- The test split was not opened for them.
- **Caveat.** Codex later found a flaw in our first implementation. In turns with several tool calls, results were
  matched to calls by order rather than by id; this happens in 36 of 6,033 tool-using responses in the training data.
  The negative result is therefore not a clean refutation, and a corrected version was not run.

## 6. Proposal: freshness for tool call results

The Model Context Protocol already has the right primitive.
- SEP-2549 added `ttlMs` and `cacheScope`, with HTTP `Cache-Control: max-age` semantics [3, 4].
- They cover only `server/discover`, `tools/list`, `prompts/list`, `resources/list`,
  `resources/templates/list` and `resources/read`.
- `tools/call` results, the data that go stale in TicToc, carry no freshness hint.

We propose:
1. Servers MAY include `ttlMs` on `CallToolResult`, with the same freshness calculation
   (`now < t_received + ttlMs`).
2. Before the model answers from a tool result in context, hosts SHOULD check its freshness. When the result is stale,
   they SHOULD either re-invoke the tool or mark the result as stale to the model. The model should not have to infer
   the age of its data.
3. Without a hint, hosts MAY apply a default threshold. The 1,800 s rule above is evidence that even a single
   default captures most human preferences in TicToc.

A draft SEP is in [5].

## 7. Limitations

- **A property of this benchmark.** The rule exploits how TicToc separates time scales. Real tools have volatility
  that varies within a tool, and a fixed default would be wrong for some of them. This is the motivation for
  server-declared TTLs.
- **Preference is not correctness.** Human preferences are a proxy for when re-fetching is worth it. Inter-annotator
  agreement is high (Krippendorff's α = 0.857 [1]).
- **Uncertain cases are excluded.** The official metric drops cases with uncertain preferences: 893 of 1,962 test
  cases (45.5%). The score says nothing about them, for the rule or for the models.
- **The rule decides when to refresh, not whether the tool is needed.** In TicToc every trajectory has a relevant
  earlier tool call. Deciding whether a new request needs a tool at all remains the model's job.
- **Our model runs differ from the authors'.** They use quantized local models through Ollama, temperature 0, no
  reasoning mode. Each message carries its timestamp as in the authors' templates, but the timestamp of the response
  turn itself is omitted; it is usually seconds after the last user message.

## 8. Reproducibility

- `rule.py` is the clock rule and the authors' metric. `ttl_per_scenario.py` is the per-scenario TTL, with the
  model's labels in `results/ttl_labels.json`. `local_models.py` runs the local models; their raw decisions are in
  `results/`.
- In our working repository, each rule, its threshold and its success criterion were committed before the test split
  was opened: commit `d14455b` for the single threshold and `8130448` for the per-scenario TTL. The commit history is
  available on request.
- Commands, hashes and outputs are in the repository https://github.com/clavis-systems/clock-outside-the-model.

**Acknowledgments.** We thank the TicToc authors for releasing data and code under Apache 2.0.

## References

[1] Y. Cheng, A. Soltani Moakhar, C. Fan, P. Hosseini, K. Faghih, Z. Sodagar, W. Wang, S. Feizi. *Your LLM Agents
are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time Perception.* Findings of ACL 2026.
arXiv:2510.23853v3 (15 April 2026).
[2] TicToc dataset. https://huggingface.co/datasets/yizecheng/TicToc ; code: https://github.com/chengez/TicToc
[3] Model Context Protocol specification, 2026-07-28, Caching.
https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching
[4] SEP-2549: TTL for List Results. https://modelcontextprotocol.io/seps/2549-TTL-for-list-results
[5] Draft SEP: Freshness Hints for Tool Call Results (this work).
[6] RFC 9111, HTTP Caching, 2022.
