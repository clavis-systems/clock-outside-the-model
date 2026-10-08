# SEP-XXXX: Freshness Hints for Tool Call Results

| Field | Value |
| - | - |
| **SEP** | XXXX (to be assigned) |
| **Title** | Freshness Hints for Tool Call Results |
| **Status** | Draft |
| **Type** | Standards Track |
| **Created** | 2026-10-07 |
| **Author(s)** | Emanuele Rizzan, independent researcher (GitHub @clavis-systems) |
| **Sponsor** | [NEEDED: a core maintainer] |
| **PR** | [to be opened] |

## Abstract

This SEP extends the caching model of SEP-2549 to the results of `tools/call`. A server MAY attach a `ttlMs` freshness
hint to a `CallToolResult`. Hosts SHOULD check the freshness of tool results already in the model's context before
the model answers from them, and SHOULD re-invoke a stale read-only tool or tell the model that the result is stale.

The motivation is empirical. On TicToc, a published benchmark of agents' tool-call decisions over elapsed time:
- every one of 18 language models with timestamps in context stays at or below 65% alignment with human preferences;
- an external freshness rule reaches 96.3% on the held-out split.

## Motivation

SEP-2549 gave MCP a freshness hint (`ttlMs`, HTTP `Cache-Control: max-age` semantics) for lists and resources. Its
cacheable results are `server/discover`, `tools/list`, `prompts/list`, `resources/list`, `resources/templates/list`
and `resources/read`. Tool call results carry no such hint.

Tool results are what agents most often reuse: prices, availability, order status, weather. Whether a result in
context can still be used is today left to the model, which sees at most the timestamps that some hosts inject.

**Models make this decision badly.**
- TicToc (Findings of ACL 2026, arXiv:2510.23853) asks whether an agent should call a tool again or answer from an
  earlier result, after elapsed times from seconds to months. Labels come from human preferences.
- The authors report that, with timestamps in context, no model exceeds 65% normalized alignment (balanced accuracy).
- Prompt-based reminders have little effect; DPO post-training brings 8B models to about 75%.

**A host-side rule makes it well.**
- The rule: re-invoke if the last result is older than 30 minutes. It is chosen on the training split and evaluated
  once on the test split.
- It reaches 96.3% (95% CI 93.7–98.1%, resampling scenarios) on 1,069 test cases from 26 scenarios not used in
  training. The rule calculations were independently re-implemented and checked (not included in the repository).
- These are benchmark results, not evidence that a TTL guarantees correct real-world answers: TicToc separates time
  scales widely and excludes uncertain preferences from scoring.

The age of a result is a number the host already knows exactly. The model should not have to infer it.

**A fixed default still loses information.** In TicToc the remaining errors concentrate where data volatility
differs by tool:
- a stock quote is stale in minutes;
- a statute or an airport code in months.

The server knows this volatility, which is why the hint belongs in the result.

On the same test split, per-scenario TTL classes (inferred by a small local model from the tool descriptions, as a
stand-in for server-declared hints) raise the score from 96.3% to 97.6%. Unnecessary re-invocations fall from 40 to
18.

## Specification

### Schema change

`CallToolResult` MAY include the `ttlMs` field defined by `CacheableResult` (SEP-2549):

```typescript
// Addition to CallToolResult in the 2026-07-28 schema. All existing fields (resultType, content,
// structuredContent, isError) are unchanged.
export interface CallToolResult extends Result {
  /**
   * OPTIONAL freshness hint: a non-negative integer number of milliseconds, only on results with
   * resultType "complete". Same freshness calculation as SEP-2549: fresh while now < t_received + ttlMs.
   * Absent means unknown freshness and the host applies its own policy. This deliberately differs from the
   * caching utility, where an absent ttlMs defaults to 0 for list and resource results.
   */
  ttlMs?: number;
}
```

`cacheScope` is not added. Tool results already belong to the requesting context; sharing them across users is out of
scope.

### Host behavior

Hosts record `t_received` for every tool result they place in a model's context.

Before a model turn that may rely on an earlier result of the same tool, the host evaluates its freshness:

1. **Fresh** (`now < t_received + ttlMs`): no action.
2. **Stale, read-only tool**, classified read-only by the host itself (a trusted server's `readOnlyHint`, or host
   policy; `readOnlyHint` from an untrusted server is not enough): the host SHOULD re-invoke the tool with the same
   arguments before the model answers. Alternatively it SHOULD inform the model that the earlier result is stale.
3. **Stale, any other tool:** the host MUST NOT re-invoke automatically, because re-running a side-effecting call
   can duplicate its effects. It SHOULD inform the model, for example with a system note naming the tool, the age of
   the result and the hint.
4. **No hint** (`ttlMs` absent): the host MAY apply a default policy. Hosts SHOULD at least expose the age of each
   tool result to the model as data, rather than leaving the model to infer it.

Hosts SHOULD NOT treat `ttlMs` as a polling interval, consistent with SEP-2549.

## Rationale

- **Why tool results.** They are the data agents reuse across turns, and the data whose staleness changes answers.
  Lists and resources already have hints.
- **Why on the host.** Age can be tracked and expiry checked by host code. On TicToc, an elapsed-time rule matches the
  scored human preferences more closely than the evaluated models; this does not isolate an arithmetic failure.
- **Why optional and server-declared.** Volatility is a property of the data, known to the server author. Making the
  field optional keeps existing servers valid.
- **Why only trusted read-only tools for automatic re-invocation.** Re-running a side-effecting call can repeat its
  effects. Tool annotations from untrusted servers are hints, not guarantees, so the classification must come from
  the host. Never replaying writes is a conservative policy of this proposal, not a consequence of the benchmark.

## Backward Compatibility

- The field is optional. Existing servers and clients are unaffected.
- Clients that ignore the field keep today's behavior.
- No capability negotiation is required.

## Security Implications

- **Too long a TTL** can make a host keep stale data. Hosts MAY cap TTLs, and MAY re-invoke when other signals
  suggest a change, as SEP-2549 allows.
- **Automatic re-invocation is limited to tools the host itself classifies as read-only.** A server that lies about
  `readOnlyHint` could otherwise turn a refresh into a side effect; hosts must not rely on untrusted annotations, and
  should apply their usual authorization and rate limits to refreshes.
- **Tool-call frequency** may rise with short TTLs. Hosts may rate-limit.

## Reference Implementation

- A host-side freshness check in Python https://github.com/clavis-systems/clock-outside-the-model (`freshness.py`).
  It is an illustrative prototype, not an MCP SDK integration: it plans actions and does not verify trust or
  permissions. It treats age >= TTL as stale, while the TicToc rule uses age > threshold.
- The TicToc evaluation: `rule.py` (single threshold) and `ttl_per_scenario.py` (per-scenario classes), with the
  authors' metric re-implemented.

## References

- SEP-2549: TTL for List Results. https://modelcontextprotocol.io/seps/2549-TTL-for-list-results
- MCP specification 2026-07-28, Caching.
  https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching
- Y. Cheng et al., *Your LLM Agents are Temporally Blind: The Misalignment Between Tool Use Decisions and Human Time
  Perception*, Findings of ACL 2026. arXiv:2510.23853.
- RFC 9111, HTTP Caching.
