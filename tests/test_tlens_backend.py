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
    format_chat_prompt,
    hook_name_for_sae,
    layer_from_sae_id,
    looks_like_chat_model_name,
    parse_float_env,
    resolve_generation_mode,
    resolve_device,
    sae_ids_for_inspection,
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

    def test_sae_ids_for_inspection_combines_explicit_and_layers(self) -> None:
        self.assertEqual(
            sae_ids_for_inspection(
                layers=[6, 6, 8],
                sae_id="custom.sae",
                template="blocks.{layer}.hook_resid_pre",
            ),
            ["custom.sae", "blocks.6.hook_resid_pre", "blocks.8.hook_resid_pre"],
        )

    def test_layer_from_sae_id_understands_common_source_names(self) -> None:
        self.assertEqual(layer_from_sae_id("blocks.6.hook_resid_pre"), 6)
        self.assertEqual(layer_from_sae_id("6-res-jb"), 6)
        self.assertIsNone(layer_from_sae_id("custom.sae"))

    def test_generation_mode_auto_uses_chat_for_chat_models(self) -> None:
        class Tokenizer:
            chat_template = "template"

            @staticmethod
            def apply_chat_template(*args, **kwargs):
                return "<chat>"

        class ChatModel:
            tokenizer = Tokenizer()

        class PlainModel:
            tokenizer = object()

        self.assertEqual(resolve_generation_mode("auto", model=ChatModel(), model_name="plain"), "chat")
        self.assertEqual(resolve_generation_mode("auto", model=PlainModel(), model_name="tiny-instruct"), "chat")
        self.assertEqual(resolve_generation_mode("auto", model=PlainModel(), model_name="gpt2-small"), "completion")
        self.assertTrue(looks_like_chat_model_name("mistral-7b-instruct"))

    def test_format_chat_prompt_uses_tokenizer_template(self) -> None:
        class Tokenizer:
            chat_template = "template"

            @staticmethod
            def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool):
                self.assertFalse(tokenize)
                self.assertTrue(add_generation_prompt)
                return "|".join(f"{message['role']}:{message['content']}" for message in messages) + "|assistant:"

        class Model:
            tokenizer = Tokenizer()

        self.assertEqual(
            format_chat_prompt(Model(), "hello", system_prompt="be brief"),
            "system:be brief|user:hello|assistant:",
        )

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

    def test_generate_chat_mode_formats_prompt_with_chat_template(self) -> None:
        backend = object.__new__(TransformerLensSteeringBackend)
        backend.config = BackendConfig(model_name="chat-model")
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

        class FakeTokenizer:
            chat_template = "template"
            eos_token_id = 99

            @staticmethod
            def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool):
                return "|".join(f"{message['role']}:{message['content']}" for message in messages) + "|assistant:"

        class FakeModel:
            tokenizer = FakeTokenizer()

            def __init__(self) -> None:
                self.prompt = None
                self.prepend_bos = None

            def to_tokens(self, prompt: str, *, prepend_bos=None) -> list[int]:
                self.prompt = prompt
                self.prepend_bos = prepend_bos
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

        text = next(
            backend.generate(
                "hello",
                max_new_tokens=1,
                temperature=0,
                seed=None,
                mode="chat",
                system_prompt="be brief",
                stop_on_eos=False,
            )
        )

        self.assertEqual(text, "x")
        self.assertEqual(backend.model.prompt, "system:be brief|user:hello|assistant:")
        self.assertFalse(backend.model.prepend_bos)

    def test_inspect_tokens_returns_top_features_for_output_tokens(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        backend = object.__new__(TransformerLensSteeringBackend)
        backend.config = BackendConfig(model_name="gpt2-small")
        backend._lock = threading.Lock()
        backend.torch = torch

        class FakeInferenceMode:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *args) -> None:
                return None

        class FakeTorch:
            @staticmethod
            def inference_mode() -> FakeInferenceMode:
                return FakeInferenceMode()

            @staticmethod
            def topk(value, *, k: int, dim: int):
                return torch.topk(value, k=k, dim=dim)

        class Config:
            hook_name = "blocks.6.hook_resid_pre"
            metadata = object()

        class FakeSae:
            cfg = Config()
            device = "cpu"

            @staticmethod
            def encode(activations):
                self.assertEqual(tuple(activations.shape), (1, 2, 2))
                return torch.tensor([[[0.0, 2.5, 1.0], [4.0, 0.0, 3.0]]])

        class FakeModel:
            def __init__(self) -> None:
                self.cache_request = None

            @staticmethod
            def to_tokens(text: str):
                return torch.tensor([[101, 102]])

            @staticmethod
            def to_str_tokens(tokens):
                return [" first", " second"]

            def run_with_cache(self, tokens, **kwargs):
                self.cache_request = kwargs
                return None, {"blocks.6.hook_resid_pre": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])}

        backend.torch = FakeTorch()
        backend.model = FakeModel()
        backend._load_sae = mock.Mock(return_value=FakeSae())

        result = backend.inspect_tokens(" output", layers=[6], top_k=2)

        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["tokens"][0]["text"], " first")
        self.assertEqual(result["tokens"][0]["features"][0]["feature_id"], 1)
        self.assertEqual(result["tokens"][0]["features"][0]["activation"], 2.5)
        self.assertEqual(result["tokens"][1]["features"][0]["feature_id"], 0)
        self.assertEqual(backend.model.cache_request["names_filter"], ["blocks.6.hook_resid_pre"])
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
