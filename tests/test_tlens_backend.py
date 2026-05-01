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
    decoder_weight_for_sae,
    hook_name_for_sae,
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

    def test_hook_name_prefers_metadata_then_config_then_fallback(self) -> None:
        class Metadata:
            hook_name = "metadata.hook"

        class Config:
            metadata = Metadata()
            hook_name = "config.hook"

        class Sae:
            cfg = Config()

        self.assertEqual(hook_name_for_sae(Sae(), "fallback.hook"), "metadata.hook")

        class ConfigOnly:
            metadata = object()
            hook_name = "config.hook"

        class ConfigOnlySae:
            cfg = ConfigOnly()

        self.assertEqual(hook_name_for_sae(ConfigOnlySae(), "fallback.hook"), "config.hook")
        self.assertEqual(hook_name_for_sae(object(), "fallback.hook"), "fallback.hook")

    def test_decoder_weight_supports_w_dec_and_decoder_module(self) -> None:
        class DirectSae:
            W_dec = "direct"

        self.assertEqual(decoder_weight_for_sae(DirectSae()), "direct")

        class Decoder:
            weight = "module-weight"

        class ModuleSae:
            W_dec = None
            decoder = Decoder()

        self.assertEqual(decoder_weight_for_sae(ModuleSae()), "module-weight")

    def test_decoder_weight_reports_missing_weights(self) -> None:
        with self.assertRaisesRegex(SteeringError, "could not find SAE decoder weights"):
            decoder_weight_for_sae(object())

    def test_hooks_for_state_rejects_model_mismatch_before_loading_sae(self) -> None:
        backend = object.__new__(TransformerLensSteeringBackend)
        backend.config = BackendConfig(model_name="gpt2-small")
        backend._load_sae = mock.Mock()

        with self.assertRaisesRegex(SteeringError, "does not match backend model"):
            backend._hooks_for_state(
                state=mock.Mock(
                    is_empty=False,
                    items=(SteerItem(204, 10, (6,), model_id="other-model"),),
                )
            )

        backend._load_sae.assert_not_called()

    def test_hooks_for_state_rejects_out_of_range_feature_id(self) -> None:
        backend = object.__new__(TransformerLensSteeringBackend)
        backend.config = BackendConfig(model_name="gpt2-small")

        class FakeDecoder:
            shape = (2,)

        class FakeSae:
            W_dec = FakeDecoder()

        backend._load_sae = mock.Mock(return_value=FakeSae())

        with self.assertRaisesRegex(SteeringError, "feature_id 3 is out of range"):
            backend._hooks_for_state(
                state=mock.Mock(
                    is_empty=False,
                    items=(SteerItem(3, 10, (6,), model_id="gpt2-small"),),
                )
            )


if __name__ == "__main__":
    unittest.main()
