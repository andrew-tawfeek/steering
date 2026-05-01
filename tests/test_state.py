from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from steering.state import (
    SteerItem,
    SteeringError,
    clear_state,
    load_state,
    parse_layers,
    update_state,
)


class StateTests(unittest.TestCase):
    def test_parse_layers_accepts_commas_and_spaces(self) -> None:
        self.assertEqual(parse_layers("6, 8 10,8"), (6, 8, 10))

    def test_parse_layers_rejects_empty(self) -> None:
        with self.assertRaises(SteeringError):
            parse_layers(" , ")

    def test_steer_item_rejects_non_finite_strength(self) -> None:
        with self.assertRaisesRegex(SteeringError, "strength must be finite"):
            SteerItem(feature_id=204, strength=float("nan"), layers=(6,))

    def test_update_replace_append_and_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            first = SteerItem(feature_id=204, strength=30, layers=(6,), label="code")
            second = SteerItem(feature_id=7, strength=-5, layers=(8,))

            state = update_state(first, append=False, path=path)
            self.assertEqual(len(state.items), 1)
            self.assertEqual(state.items[0].feature_id, 204)

            state = update_state(second, append=True, path=path)
            self.assertEqual([item.feature_id for item in state.items], [204, 7])

            persisted = load_state(path)
            self.assertEqual(len(persisted.items), 2)
            self.assertEqual(json.loads(path.read_text())["version"], 1)

            cleared = clear_state(path)
            self.assertTrue(cleared.is_empty)
            self.assertTrue(load_state(path).is_empty)


if __name__ == "__main__":
    unittest.main()
