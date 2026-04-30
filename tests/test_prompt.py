from __future__ import annotations

import unittest

from steering.prompt import build_steering_system_prompt
from steering.state import SteerItem, SteeringState


class PromptTests(unittest.TestCase):
    def test_empty_state_has_no_system_prompt(self) -> None:
        self.assertEqual(build_steering_system_prompt(SteeringState.empty()), "")

    def test_prompt_includes_labels_and_layers(self) -> None:
        state = SteeringState(items=(SteerItem(204, 30, (6,), "Python code"),))
        prompt = build_steering_system_prompt(state)
        self.assertIn("Python code", prompt)
        self.assertIn("feature_id=204", prompt)
        self.assertIn("layers=6", prompt)


if __name__ == "__main__":
    unittest.main()
