from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
import threading
from typing import Any, Callable, Iterator

from .state import SteerItem, SteeringError, SteeringState, load_state


DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"
DEFAULT_SAE_ID_TEMPLATE = "blocks.{layer}.hook_resid_pre"
DEFAULT_DEVICE = "auto"


@dataclass(frozen=True)
class BackendConfig:
    model_name: str = DEFAULT_MODEL_NAME
    sae_release: str = DEFAULT_SAE_RELEASE
    sae_id_template: str = DEFAULT_SAE_ID_TEMPLATE
    device: str = DEFAULT_DEVICE
    state_path: Path | None = None

    @classmethod
    def from_env(cls, state_path: Path | None = None) -> "BackendConfig":
        return cls(
            model_name=os.environ.get("STEERING_MODEL_NAME", DEFAULT_MODEL_NAME),
            sae_release=os.environ.get("STEERING_SAE_RELEASE", DEFAULT_SAE_RELEASE),
            sae_id_template=os.environ.get("STEERING_SAE_ID_TEMPLATE", DEFAULT_SAE_ID_TEMPLATE),
            device=os.environ.get("STEERING_DEVICE", DEFAULT_DEVICE),
            state_path=state_path,
        )


class TransformerLensSteeringBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._saes: dict[str, Any] = {}

        try:
            import torch
            from sae_lens import SAE
            from transformer_lens import HookedTransformer
        except ImportError as exc:
            raise RuntimeError(
                "TransformerLens backend requires torch, transformer-lens, and sae-lens. "
                "Install them with: pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.SAE = SAE
        device = resolve_device(torch, config.device)
        self.config = BackendConfig(
            model_name=config.model_name,
            sae_release=config.sae_release,
            sae_id_template=config.sae_id_template,
            device=device,
            state_path=config.state_path,
        )
        self.model = HookedTransformer.from_pretrained(
            config.model_name,
            device=device,
        )
        self.model.eval()

    def health(self) -> dict[str, Any]:
        return {
            "backend": "transformer-lens",
            "model_name": self.config.model_name,
            "sae_release": self.config.sae_release,
            "sae_id_template": self.config.sae_id_template,
            "device": self.config.device,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> Iterator[str]:
        if max_new_tokens < 1:
            raise SteeringError("max_new_tokens must be >= 1")
        if temperature < 0:
            raise SteeringError("temperature must be >= 0")

        if seed is not None:
            random.seed(seed)
            self.torch.manual_seed(seed)

        with self._lock, self.torch.inference_mode():
            tokens = self.model.to_tokens(prompt)
            generated: list[int] = []
            for _ in range(max_new_tokens):
                state = load_state(self.config.state_path)
                hooks = self._hooks_for_state(state)
                logits = self.model.run_with_hooks(
                    tokens,
                    return_type="logits",
                    fwd_hooks=hooks,
                )
                next_token = self._sample_next_token(logits[0, -1], temperature)
                token_id = int(next_token.item())
                generated.append(token_id)
                tokens = self.torch.cat([tokens, next_token.reshape(1, 1)], dim=1)
                yield self.model.to_string([token_id])

    def _hooks_for_state(self, state: SteeringState) -> list[tuple[str, Callable]]:
        if state.is_empty:
            return []

        deltas: dict[str, Any] = {}
        for item in state.items:
            if item.model_id and item.model_id != self.config.model_name:
                raise SteeringError(
                    f"state item model_id={item.model_id!r} does not match "
                    f"backend model {self.config.model_name!r}"
                )
            for sae_id in sae_ids_for_item(item, self.config.sae_id_template):
                sae = self._load_sae(sae_id)
                hook_name = hook_name_for_sae(sae, sae_id)
                decoder = decoder_weight_for_sae(sae)
                if item.feature_id >= decoder.shape[0]:
                    raise SteeringError(
                        f"feature_id {item.feature_id} is out of range for {sae_id} "
                        f"(d_sae={decoder.shape[0]})"
                    )
                vector = decoder[item.feature_id] * item.strength
                deltas[hook_name] = vector if hook_name not in deltas else deltas[hook_name] + vector

        hooks: list[tuple[str, Callable]] = []
        for hook_name, delta in deltas.items():
            hooks.append((hook_name, make_additive_hook(delta)))
        return hooks

    def _load_sae(self, sae_id: str) -> Any:
        if sae_id not in self._saes:
            sae = self.SAE.from_pretrained(
                release=self.config.sae_release,
                sae_id=sae_id,
                device=self.config.device,
            )
            if isinstance(sae, tuple):
                sae = sae[0]
            sae.eval()
            self._saes[sae_id] = sae
        return self._saes[sae_id]

    def _sample_next_token(self, logits: Any, temperature: float) -> Any:
        if temperature == 0:
            return logits.argmax(dim=-1)

        probs = self.torch.softmax(logits / temperature, dim=-1)
        return self.torch.multinomial(probs, num_samples=1).squeeze(0)


def sae_ids_for_item(item: SteerItem, template: str) -> list[str]:
    if item.sae_id:
        return [item.sae_id]
    return [template.format(layer=layer) for layer in item.layers]


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def hook_name_for_sae(sae: Any, fallback: str) -> str:
    cfg = getattr(sae, "cfg", None)
    metadata = getattr(cfg, "metadata", None)
    for owner in (metadata, cfg):
        hook_name = getattr(owner, "hook_name", None)
        if hook_name:
            return str(hook_name)
    return fallback


def decoder_weight_for_sae(sae: Any) -> Any:
    decoder = getattr(sae, "W_dec", None)
    if decoder is not None:
        return decoder

    decoder_module = getattr(sae, "decoder", None)
    weight = getattr(decoder_module, "weight", None)
    if weight is not None:
        return weight

    raise SteeringError("could not find SAE decoder weights")


def make_additive_hook(delta: Any) -> Callable:
    def hook(activation: Any, hook: Any | None = None) -> Any:
        return activation + delta.to(device=activation.device, dtype=activation.dtype)

    return hook
