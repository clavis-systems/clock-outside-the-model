"""Host-side freshness check for tool results: an illustrative prototype for the draft SEP "Freshness Hints for Tool
Call Results", not an MCP SDK integration and not production-ready (no clock source, invalidation or rate limits).

The host, not the model, knows how old each tool result in the context is. Before a model turn, the host asks the
guard which results are stale:
  - fresh results need no action;
  - stale results of read-only tools should be re-invoked with the same arguments;
  - stale results of other tools must not be re-invoked automatically (side effects): the model is told instead.
A result without a server hint (`ttl_ms is None`) uses the host's default policy, here a single default TTL.
On the TicToc benchmark a 30-minute default reaches 96.3% alignment with human preferences on the test split, against
at most 65% for language models deciding from timestamps in context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

DEFAULT_TTL_MS = 30 * 60 * 1000


@dataclass(frozen=True)
class ToolResult:
    tool: str
    arguments: dict
    received_ms: int                 # host clock when the result arrived
    ttl_ms: int | None = None        # server hint (CallToolResult.ttlMs); None = unknown
    read_only: bool = False          # True only if the host trusts this classification (see check)
    server: str = ""                 # which server produced the result
    auth_context: str = ""           # authorization context (results must not leak across contexts)


@dataclass(frozen=True)
class Action:
    kind: str                        # "fresh" | "reinvoke" | "warn"
    result: ToolResult
    age_ms: int
    ttl_ms: int


@dataclass
class FreshnessGuard:
    default_ttl_ms: int = DEFAULT_TTL_MS
    max_ttl_ms: int | None = None    # optional cap against overly long hints
    results: list[ToolResult] = field(default_factory=list)

    def record(self, result: ToolResult) -> None:
        self.results.append(result)

    def ttl_of(self, result: ToolResult) -> int:
        ttl = self.default_ttl_ms if result.ttl_ms is None else max(0, result.ttl_ms)
        return ttl if self.max_ttl_ms is None else min(ttl, self.max_ttl_ms)

    def latest(self) -> list[ToolResult]:
        """The most recent result for each (tool, arguments) pair: older ones are superseded. Arguments may be nested;
        they are compared through their canonical JSON form."""
        newest: dict[tuple, ToolResult] = {}
        for r in self.results:
            key = (r.server, r.auth_context, r.tool, json.dumps(r.arguments, sort_keys=True, default=str))
            if key not in newest or r.received_ms >= newest[key].received_ms:
                newest[key] = r
        return list(newest.values())

    def check(self, now_ms: int) -> list[Action]:
        """A result is stale when its age is >= its TTL (the TicToc rule in rule.py uses age > threshold)."""
        actions = []
        for r in self.latest():
            ttl, age = self.ttl_of(r), now_ms - r.received_ms
            if age < ttl:
                kind = "fresh"
            else:
                # only a real boolean True counts: "false" as a string must never trigger a re-run. Hosts should set
                # read_only only from annotations of servers they trust (MCP: annotations of untrusted servers are hints).
                kind = "reinvoke" if r.read_only is True else "warn"
            actions.append(Action(kind, r, age, ttl))
        return actions

    @staticmethod
    def note_for_model(actions: list[Action]) -> str:
        """A short system note for stale results that the host did not re-invoke."""
        lines = [f"The result of {a.result.tool}({a.result.arguments}) is {a.age_ms // 1000} s old and may be stale "
                 f"(freshness hint {a.ttl_ms // 1000} s)." for a in actions if a.kind == "warn"]
        return "\n".join(lines)
