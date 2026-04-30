from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from steering.state import SteeringError, SteerItem
from steering.tlens_backend import (
    DEFAULT_DEVICE,
    BackendConfig,
    TransformerLensSteeringBackend,
    parse_float_env,
    resolve_device,
    sae_ids_for_item,
)


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

    def test_generation_lock_timeout_reports_busy_backend(self) -> None:
        backend = object.__new__(TransformerLensSteeringBackend)
        backend.config = BackendConfig(generation_lock_timeout=0.01)
        backend._lock = threading.Lock()
        backend._lock.acquire()
        backend.torch = object()

        class FakeModel:
            @staticmethod
            def to_tokens(prompt: str) -> list[int]:
                return [1]

        backend.model = FakeModel()
        try:
            with self.assertRaisesRegex(SteeringError, "another generation is still running"):
                next(backend.generate("hello", max_new_tokens=1, temperature=0, seed=None))
        finally:
            backend._lock.release()

    def test_generation_releases_lock_before_yielding_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = object.__new__(TransformerLensSteeringBackend)
            backend.config = BackendConfig(state_path=Path(tmp) / "state.json")
            backend._lock = threading.Lock()

            class FakeInferenceMode:
                def __enter__(self) -> None:
                    return None

                def __exit__(self, *args) -> None:
                    return None

            class FakeToken:
                def __init__(self, token_id: int) -> None:
                    self.token_id = token_id

                def item(self) -> int:
                    return self.token_id

                def reshape(self, *shape) -> list[int]:
                    return [self.token_id]

            class FakeTorch:
                @staticmethod
                def inference_mode() -> FakeInferenceMode:
                    return FakeInferenceMode()

                @staticmethod
                def cat(parts, dim: int):
                    return [*parts[0], *parts[1]]

            class FakeLogits:
                def __getitem__(self, key):
                    return self

            class FakeModel:
                @staticmethod
                def to_tokens(prompt: str) -> list[int]:
                    return [1]

                @staticmethod
                def run_with_hooks(tokens, *, return_type: str, fwd_hooks: list):
                    return FakeLogits()

                @staticmethod
                def to_string(tokens: list[int]) -> str:
                    return "x"

            backend.torch = FakeTorch()
            backend.model = FakeModel()
            backend._hooks_for_state = lambda state: []
            backend._sample_next_token = lambda logits, temperature: FakeToken(2)

            self.assertEqual(next(backend.generate("hello", max_new_tokens=1, temperature=0, seed=None)), "x")
            self.assertFalse(backend._lock.locked())

    def test_parse_float_env_rejects_non_positive_values(self) -> None:
        with mock.patch.dict("os.environ", {"STEERING_GENERATION_LOCK_TIMEOUT": "0"}):
            with self.assertRaisesRegex(RuntimeError, "greater than 0"):
                parse_float_env("STEERING_GENERATION_LOCK_TIMEOUT", 30.0)


if __name__ == "__main__":
    unittest.main()
