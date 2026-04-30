from __future__ import annotations

import unittest

from steering.state import SteerItem
from steering.tlens_backend import DEFAULT_DEVICE, resolve_device, sae_ids_for_item


class TransformerLensBackendTests(unittest.TestCase):
    def test_default_device_is_cpu_for_stable_mac_demos(self) -> None:
        self.assertEqual(DEFAULT_DEVICE, "cpu")

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

    def test_auto_device_still_prefers_available_accelerator(self) -> None:
        class FakeMps:
            @staticmethod
            def is_available() -> bool:
                return True

        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class FakeBackends:
            mps = FakeMps()

        class FakeTorch:
            backends = FakeBackends()
            cuda = FakeCuda()

        self.assertEqual(resolve_device(FakeTorch(), "auto"), "mps")


if __name__ == "__main__":
    unittest.main()
