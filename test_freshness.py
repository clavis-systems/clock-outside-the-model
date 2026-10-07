"""Tests for the host-side freshness guard. Run: python -m unittest discover -s organo_tempo -p "test_*.py" -v"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freshness import DEFAULT_TTL_MS, FreshnessGuard, ToolResult  # noqa: E402

MIN = 60 * 1000


class Guard(unittest.TestCase):
    def test_fresh_and_stale_with_server_hint(self):
        g = FreshnessGuard()
        g.record(ToolResult("get_stock_price", {"ticker": "TSLA"}, received_ms=0, ttl_ms=5 * MIN, read_only=True))
        self.assertEqual(g.check(4 * MIN)[0].kind, "fresh")
        self.assertEqual(g.check(5 * MIN)[0].kind, "reinvoke")

    def test_default_policy_without_hint(self):
        g = FreshnessGuard()
        g.record(ToolResult("lookup_statute", {"id": "art. 7"}, received_ms=0, read_only=True))
        self.assertEqual(g.check(DEFAULT_TTL_MS - 1)[0].kind, "fresh")
        self.assertEqual(g.check(DEFAULT_TTL_MS)[0].kind, "reinvoke")

    def test_side_effect_tools_are_never_reinvoked(self):
        g = FreshnessGuard()
        g.record(ToolResult("reserve_spot", {"spot": "DLA101"}, received_ms=0, ttl_ms=MIN, read_only=False))
        actions = g.check(10 * MIN)
        self.assertEqual(actions[0].kind, "warn")
        self.assertIn("reserve_spot", FreshnessGuard.note_for_model(actions))

    def test_newest_result_supersedes_older_and_caps_apply(self):
        g = FreshnessGuard(max_ttl_ms=10 * MIN)
        g.record(ToolResult("weather", {"city": "Roma"}, received_ms=0, ttl_ms=60 * MIN, read_only=True))
        g.record(ToolResult("weather", {"city": "Roma"}, received_ms=20 * MIN, ttl_ms=60 * MIN, read_only=True))
        actions = g.check(25 * MIN)
        self.assertEqual(len(actions), 1)                       # only the newest result counts
        self.assertEqual((actions[0].kind, actions[0].ttl_ms), ("fresh", 10 * MIN))   # capped at 10 min
        self.assertEqual(g.check(31 * MIN)[0].kind, "reinvoke")


    def test_nested_arguments_and_untrusted_flags(self):
        g = FreshnessGuard()
        g.record(ToolResult("search", {"filters": {"city": "Roma", "tags": ["a", "b"]}}, received_ms=0,
                            read_only=True))
        g.record(ToolResult("search", {"filters": {"tags": ["a", "b"], "city": "Roma"}}, received_ms=10,
                            read_only=True))
        self.assertEqual(len(g.check(DEFAULT_TTL_MS + 20)), 1)          # same nested arguments, one entry
        g2 = FreshnessGuard()
        g2.record(ToolResult("cancel_order", {"id": 7}, received_ms=0, read_only="false"))   # a string, not a bool
        self.assertEqual(g2.check(DEFAULT_TTL_MS)[0].kind, "warn")        # never re-run on a non-boolean flag


if __name__ == "__main__":
    unittest.main(verbosity=2)
