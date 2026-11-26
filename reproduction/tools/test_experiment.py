"""Regression tests for state handoffs, arm construction and the provider adapter."""

import io
import json
from unittest import TestCase, main
from unittest.mock import patch
from experiment import Scaffold, attempt_value, live_response


class HandoffTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scaffold = Scaffold()

    def test_stage_two_depends_on_new_carried_state(self):
        row = next(
            r
            for r in self.scaffold.schedule
            if r["layer"] == "controlled_formal_1200" and r["condition"] == "N"
        )
        stage = row["stages"][-1]
        cfg = self.scaffold.profiles[stage["profile"]]
        left = self.scaffold.messages(
            row, stage["name"], cfg, {"carried_state": "first candidate"}
        )
        right = self.scaffold.messages(
            row, stage["name"], cfg, {"carried_state": "second candidate"}
        )
        self.assertNotEqual(left, right)
        self.assertIn("second candidate", right[1]["content"])
        self.assertNotIn("first candidate", right[1]["content"])

    def test_no_body_retains_failed_attempt(self):
        empty = next(
            a
            for trials in self.scaffold.calls.values()
            for a in trials
            if a["response_status"] == "no_return"
        )
        self.assertIsNone(empty["returned"])
        self.assertEqual(attempt_value(empty), {})
        self.assertNotIn("recorded_body_sha256", empty)

    def test_four_states_are_distinct(self):
        for arms in self.scaffold.tasks["deletion"].values():
            self.assertEqual(len(set(arms.values())), 4)

    def test_repair_changes_a_single_supplied_state(self):
        key = next(iter(self.scaffold.repair_states))
        before = self.scaffold.tasks["repair"][key]["baseline_state"]
        after = self.scaffold.repair_states[key]
        self.assertNotEqual(before, after)
        self.assertEqual(set(before), set(after))

    def test_provider_message_boundaries(self):
        text = ' \n{"example": true}\n '
        payload = {
            "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "model": "example",
        }

        class LocalResponse:
            def open(self, req, timeout):
                self.payload = json.loads(req.data)
                return io.BytesIO(json.dumps(payload).encode())

        fake = LocalResponse()
        with patch("experiment.request.build_opener", return_value=fake):
            returned = live_response(
                "https://example.invalid/test", "fixture", {"model": "example"}
            )
        self.assertEqual(fake.payload, {"model": "example"})
        self.assertEqual(returned["returned"]["choices"][0]["text"], text)


if __name__ == "__main__":
    main()
