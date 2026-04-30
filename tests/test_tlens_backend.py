from __future__ import annotations

import unittest

from steering.state import SteerItem
from steering.tlens_backend import sae_ids_for_item


class TransformerLensBackendTests(unittest.TestCase):
    def test_sae_ids_for_layers(self) -> None:
        item = SteerItem(feature_id=204, strength=30, layers=(6, 8))

        self.assertEqual(
            sae_ids_for_item(item, "blocks.{layer}.hook_resid_pre"),
            ["blocks.6.hook_resid_pre", "blocks.8.hook_resid_pre"],
        )

    def test_explicit_sae_id_wins(self) -> None:
        item = SteerItem(
            feature_id=204,
            strength=30,
            layers=(6,),
            sae_id="custom.sae",
        )

        self.assertEqual(sae_ids_for_item(item, "blocks.{layer}.hook_resid_pre"), ["custom.sae"])


if __name__ == "__main__":
    unittest.main()
