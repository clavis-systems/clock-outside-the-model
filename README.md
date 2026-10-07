# A clock outside the model

**Temporal blindness in LLM agents is an architecture problem.** On TicToc (Findings of ACL 2026), a one-line rule run
by the agent host beats every language model and the post-trained models reported by the benchmark's authors.

| System | Setting | Normalized alignment |
|---|---|---|
| 18 open and proprietary models (paper) | with timestamps, full data | ≤ 65% |
| 8B models after DPO post-training (paper) | test split | ≈ 73–76% |
| gemma4 e4b, local (this repo) | with timestamps, test split | 61.0% |
| Qwen3-8B, local (this repo) | with timestamps, test split | 56.9% |
| **Clock rule: call again if the last result is older than 30 min** | test split, 1,069 cases | **96.3%** (95% CI 93.7–98.1, by scenario) |
| **Clock rule with a TTL per scenario, inferred once by a 4B local model** | test split, 1,069 cases | **97.6%** |

The threshold was chosen on the training split only and frozen before the test split was opened. An independent
re-implementation reproduced the numbers.

## Why it matters

TicToc asks: after some time has passed since the last tool call, should the agent call the tool again or answer from
what it has?
- The authors found that models handle this badly even with timestamps in context, and that prompting does not fix it.
- The age of a tool result is a number the host already knows. So the host should decide, or at least tell the model.

We propose extending the Model Context Protocol's freshness hint, `ttlMs`, from lists and resources to tool call
results, with host-side enforcement ([SEP_DRAFT.md](SEP_DRAFT.md)).

## Reproduce

1. Download the data (Apache 2.0) from https://huggingface.co/datasets/yizecheng/TicToc into `data/`:
   `test.parquet` and `train.parquet`.
2. Run the rule. It needs Python ≥ 3.10, plus `pyarrow` for parquet:

   ```bash
   python rule.py data/test.parquet
   python rule.py data/test.parquet --exclude-boundaries
   python rule.py data/train.parquet --threshold 1800
   ```

3. Optionally run a local model with [Ollama](https://ollama.com):

   ```bash
   python local_models.py gemma4:e4b-it-qat data/test.parquet
   ```

4. Run the tests of the reference host-side freshness check:

   ```bash
   python -m unittest test_freshness.py
   ```

## Files

- [`rule.py`](rule.py): the rule and the authors' metric, re-implemented, with intervals by case and by scenario.
- [`local_models.py`](local_models.py): local models in the "with timestamps" setting.
- [`freshness.py`](freshness.py): reference host-side freshness check. Stale read-only tools are re-invoked; for other
  tools the model is warned, and they are never re-run automatically.
- [`NOTE.md`](NOTE.md): the full note, with method, results and limitations.
- [`SEP_DRAFT.md`](SEP_DRAFT.md): the draft protocol extension.

## Limits

- The result reflects how TicToc separates time scales. No evaluable test case falls between 7 and 60 minutes.
- The medium time level is the hard part: the rule scores 75% there.
- Uncertain preferences (45.5% of the test split) are excluded by the official metric.

See [`NOTE.md`](NOTE.md) for details.

## Credits

- **TicToc:** Y. Cheng et al., *Your LLM Agents are Temporally Blind*, Findings of ACL 2026, arXiv:2510.23853.
- **This work:** Emanuele Rizzan, independent researcher (GitHub @clavis-systems). Code and text written with Claude (Anthropic). Results independently
  recomputed by Codex (OpenAI).

License: Apache-2.0 (see [LICENSE](LICENSE)). Copyright 2026 Emanuele Rizzan.
