"""Regression tests for per_tool.py, on synthetic data only (no dataset, no model, no network).

    python -m unittest test_per_tool
"""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import per_tool


class FitThreshold(unittest.TestCase):
    def test_true_ties_keep_the_lower_median(self):
        # 128 s and 512 s both give NAR 7/12; the lower median of the optimal thresholds is 128 s.
        pairs = [(1, "tool"), (256, "direct"), (256, "tool"), (4, "direct"), (4, "direct"), (1024, "tool"),
                 (4, "tool"), (1024, "direct"), (1024, "tool"), (64, "direct"), (64, "direct"), (4, "tool")]
        self.assertEqual(per_tool.fit_threshold(pairs), 128.0)

    def test_constant_labels_and_empty_input(self):
        self.assertEqual(per_tool.fit_threshold([(5, "tool"), (50, "tool")]), -math.inf)
        self.assertEqual(per_tool.fit_threshold([(5, "direct")]), math.inf)
        self.assertEqual(per_tool.fit_threshold([]), per_tool.FALLBACK_S)

    def test_separable_labels_are_split_between_them(self):
        t = per_tool.fit_threshold([(10, "direct"), (20, "direct"), (1000, "tool")])
        self.assertTrue(20 < t < 1000)


class PersonZero(unittest.TestCase):
    def test_zero_means_always_call_again_even_at_age_zero(self):
        path = Path(tempfile.mkdtemp()) / "person.json"
        path.write_text(json.dumps({"answers": {"a": {"ttl_seconds": 0}}}), encoding="utf-8")
        labels = Path(tempfile.mkdtemp()) / "ttl_labels.json"
        labels.write_text(json.dumps({"a": {"class": "secondi"}}), encoding="utf-8")
        cases = [{"id": "a_1_tv1", "scenario": "a", "label": "tool",
                  "history": [{"role": "tool", "time": "2026-01-01T00:00:00Z"},
                              {"role": "user", "time": "2026-01-01T00:00:00Z"}]}]
        with mock.patch.object(per_tool, "PERSON", path), mock.patch.object(per_tool, "RESULTS", labels.parent):
            out = per_tool.systems(cases, {"a_1_tv1": 0.0}, {"a": 1800.0}, per_tool.person_table({"a"}))
        self.assertEqual(out["TTL declared by one person, applied by the host"]["a_1_tv1"], "tool")


class CaseIds(unittest.TestCase):
    def test_only_a_terminal_tv_suffix_is_removed(self):
        self.assertEqual(per_tool.parts("stock_trading_1_tv2"), ("stock_trading_1", "tv2"))
        for bad in ("una_finale", "una_diversa", "x_tv", "x_tv2_extra"):
            with self.assertRaises(ValueError):
                per_tool.parts(bad)


class Bootstrap(unittest.TestCase):
    def test_undefined_replicates_are_counted_not_replaced(self):
        cases = [{"id": f"s_{i}_tv1", "scenario": "s", "label": "tool"} for i in range(3)]
        ages = {c["id"]: 100.0 for c in cases}
        decisions = {c["id"]: "tool" for c in cases}
        result = per_tool.bootstrap(cases, ages, decisions, None, "binned")
        self.assertEqual(result, {"ci95": None, "valid": 0, "undefined": per_tool.REPS})


class Declarations(unittest.TestCase):
    def write(self, entries) -> Path:
        path = Path(tempfile.mkdtemp()) / "declared.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_parse_rule(self):
        self.assertEqual(per_tool.parse_declaration('{"ttl_seconds": 60}'), 60.0)
        self.assertEqual(per_tool.parse_declaration('{"ttl_seconds": "never"}'), "never")
        for raw in ('{"ttl_seconds": 0}', '{"ttl_seconds": -1}', '{"ttl_seconds": true}', '{"ttl_seconds": "60"}',
                    '{"ttl_seconds": NaN}', '{"ttl": 60}', "not json"):
            self.assertIsNone(per_tool.parse_declaration(raw), raw)

    def test_invalid_answers_fall_back_and_are_counted(self):
        path = self.write({"a": {"split": "train", "raw": '{"ttl_seconds": 0}', "ttl_seconds": None},
                           "b": {"split": "test", "raw": '{"ttl_seconds": 60}', "ttl_seconds": 60.0}})
        with mock.patch.object(per_tool, "DECLARED", path):
            table, invalid = per_tool.declared_table({"a", "b"})
        self.assertEqual((table, invalid), ({"a": per_tool.FALLBACK_S, "b": 60.0}, 1))

    def test_out_of_contract_values_are_errors(self):
        for value, raw in ((-1, '{"ttl_seconds": -1}'), (0, '{"ttl_seconds": 0}'), (True, '{"ttl_seconds": true}'),
                           ("0", '{"ttl_seconds": "0"}'), (60.0, '{"ttl_seconds": 30}')):
            path = self.write({"a": {"split": "train", "raw": raw, "ttl_seconds": value}})
            with mock.patch.object(per_tool, "DECLARED", path), self.assertRaises(ValueError):
                per_tool.declared_table()

    def test_scenarios_must_match(self):
        path = self.write({"a": {"split": "train", "raw": '{"ttl_seconds": 60}', "ttl_seconds": 60.0}})
        with mock.patch.object(per_tool, "DECLARED", path), self.assertRaises(ValueError):
            per_tool.declared_table({"a", "b"})


class Logs(unittest.TestCase):
    def test_duplicates_unknown_ids_and_wrong_labels_are_errors(self):
        test_cases = [{"id": "s_1_tv2", "label": "tool"}, {"id": "s_2_tv2", "label": "direct"}]
        for lines in ([{"id": "s_1_tv2", "decisione": "tool"}, {"id": "s_1_tv2", "decisione": "direct"}],
                      [{"id": "s_9_tv2", "decisione": "tool"}],
                      [{"id": "s_1_tv2", "verita": "direct", "decisione": "tool"}]):
            path = Path(tempfile.mkdtemp()) / "log.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
            with self.assertRaises(ValueError):
                per_tool.read_log(path, test_cases)

    def test_invalid_and_missing_decisions_are_counted(self):
        test_cases = [{"id": "s_1_tv2", "label": "tool"}, {"id": "s_2_tv2", "label": "direct"},
                      {"id": "s_3_tv2", "label": "direct"}]
        path = Path(tempfile.mkdtemp()) / "log.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in [{"id": "s_1_tv2", "decisione": "tool"},
                                                          {"id": "s_2_tv2", "decisione": "errore"}]), encoding="utf-8")
        decisions, counts = per_tool.read_log(path, test_cases)
        self.assertEqual(decisions, {"s_1_tv2": "tool"})
        self.assertEqual(counts, {"valid": 1, "invalid": 1, "missing": 1})


class Regenerate(unittest.TestCase):
    def test_refuses_before_asking_the_model(self):
        existing = Path(tempfile.mkdtemp()) / "regenerated.json"
        existing.write_text("{}", encoding="utf-8")
        splits = {"train": [{"scenario": "a", "id": "a_1_tv1"}], "test": [{"scenario": "b", "id": "b_1_tv1"}]}
        with mock.patch.object(per_tool, "REGENERATED", existing), \
                mock.patch.object(per_tool, "ask", side_effect=AssertionError("model called")) as ask:
            with self.assertRaises(SystemExit):
                per_tool.regenerate(splits)
        ask.assert_not_called()
        self.assertEqual(existing.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
