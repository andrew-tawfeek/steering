from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
import re
import threading
from typing import Any, Callable, Iterator

from .state import SteerItem, SteeringError, SteeringState, load_state


DEFAULT_MODEL_NAME = "gpt2-small"
DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"
DEFAULT_SAE_ID_TEMPLATE = "blocks.{layer}.hook_resid_pre"
DEFAULT_DEVICE = "cpu"
DEFAULT_GENERATION_LOCK_TIMEOUT = 30.0
GENERATION_MODES = {"auto", "completion", "chat"}


@dataclass(frozen=True)
class BackendConfig:
    model_name: str = DEFAULT_MODEL_NAME
    sae_release: str = DEFAULT_SAE_RELEASE
    sae_id_template: str = DEFAULT_SAE_ID_TEMPLATE
    device: str = DEFAULT_DEVICE
    generation_lock_timeout: float = DEFAULT_GENERATION_LOCK_TIMEOUT
    state_path: Path | None = None

    @classmethod
    def from_env(cls, state_path: Path | None = None) -> "BackendConfig":
        return cls(
            model_name=os.environ.get("STEERING_MODEL_NAME", DEFAULT_MODEL_NAME),
            sae_release=os.environ.get("STEERING_SAE_RELEASE", DEFAULT_SAE_RELEASE),
            sae_id_template=os.environ.get("STEERING_SAE_ID_TEMPLATE", DEFAULT_SAE_ID_TEMPLATE),
            device=os.environ.get("STEERING_DEVICE", DEFAULT_DEVICE),
            generation_lock_timeout=parse_float_env(
                "STEERING_GENERATION_LOCK_TIMEOUT",
                DEFAULT_GENERATION_LOCK_TIMEOUT,
            ),
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
            generation_lock_timeout=config.generation_lock_timeout,
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
            "busy": self._lock.locked(),
            "chat_template_available": chat_template_available(self.model),
            "chat_model_guess": looks_like_chat_model_name(self.config.model_name),
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
        state_override: SteeringState | None = None,
        mode: str = "completion",
        system_prompt: str | None = None,
        stop_on_eos: bool | None = None,
    ) -> Iterator[str]:
        if max_new_tokens < 1:
            raise SteeringError("max_new_tokens must be >= 1")
        if temperature < 0:
            raise SteeringError("temperature must be >= 0")
        resolved_mode = resolve_generation_mode(mode, model=self.model, model_name=self.config.model_name)
        if stop_on_eos is None:
            stop_on_eos = resolved_mode == "chat"

        if seed is not None:
            random.seed(seed)
            self.torch.manual_seed(seed)

        prompt_text = self._generation_prompt_text(prompt, resolved_mode, system_prompt)
        tokens = self._tokens_for_prompt_text(prompt_text, resolved_mode)
        stop_ids = stop_token_ids(self.model) if stop_on_eos else set()
        for _ in range(max_new_tokens):
            state = state_override if state_override is not None else load_state(self.config.state_path)
            if not self._lock.acquire(timeout=self.config.generation_lock_timeout):
                raise SteeringError("another generation is still running; wait or restart the backend")
            try:
                with self.torch.inference_mode():
                    hooks = self._hooks_for_state(state)
                    logits = self.model.run_with_hooks(
                        tokens,
                        return_type="logits",
                        fwd_hooks=hooks,
                    )
                    next_token = self._sample_next_token(logits[0, -1], temperature)
                    token_id = int(next_token.item())
                    tokens = self.torch.cat([tokens, next_token.reshape(1, 1)], dim=1)
            finally:
                self._lock.release()

            if token_id in stop_ids:
                break
            yield self.model.to_string([token_id])

    def inspect_tokens(
        self,
        text: str,
        *,
        layers: list[int] | tuple[int, ...] = (),
        sae_id: str | None = None,
        top_k: int = 5,
        prompt: str | None = None,
        include_prompt: bool = False,
        mode: str = "completion",
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if not text:
            raise SteeringError("text is required")
        if top_k < 1:
            raise SteeringError("top_k must be >= 1")

        sae_ids = sae_ids_for_inspection(
            layers=layers,
            sae_id=sae_id,
            template=self.config.sae_id_template,
        )
        if not sae_ids:
            raise SteeringError("provide at least one layer or SAE id to inspect")

        resolved_mode = resolve_generation_mode(mode, model=self.model, model_name=self.config.model_name)
        prompt_text = self._generation_prompt_text(prompt or "", resolved_mode, system_prompt)
        context = f"{prompt_text}{text}"
        if not context:
            raise SteeringError("text is required")

        if not self._lock.acquire(timeout=self.config.generation_lock_timeout):
            raise SteeringError("another generation is still running; wait or restart the backend")
        try:
            with self.torch.inference_mode():
                sources = []
                hook_names = []
                for source_sae_id in sae_ids:
                    sae = self._load_sae(source_sae_id)
                    hook_name = hook_name_for_sae(sae, source_sae_id)
                    sources.append(
                        {
                            "sae_id": source_sae_id,
                            "hook_name": hook_name,
                            "layer": layer_from_sae_id(source_sae_id),
                            "sae": sae,
                        }
                    )
                    hook_names.append(hook_name)

                tokens = self._tokens_for_prompt_text(context, resolved_mode)
                prompt_token_count = 0
                if prompt_text:
                    prompt_token_count = int(self._tokens_for_prompt_text(prompt_text, resolved_mode).shape[-1])

                _, cache = self.model.run_with_cache(
                    tokens,
                    return_type=None,
                    names_filter=sorted(set(hook_names)),
                )

                token_ids = tensor_to_list(tokens[0])
                token_texts = token_strings(self.model, tokens[0], token_ids)
                start_position = 0 if include_prompt else prompt_token_count
                inspected_positions = range(start_position, len(token_ids))
                result_tokens = [
                    {
                        "position": position,
                        "token_id": int(token_ids[position]),
                        "text": token_texts[position],
                        "is_prompt": position < prompt_token_count,
                        "features": [],
                    }
                    for position in inspected_positions
                ]
                by_position = {token["position"]: token for token in result_tokens}

                for source in sources:
                    hook_name = source["hook_name"]
                    if hook_name not in cache:
                        raise SteeringError(f"hook {hook_name!r} was not captured")
                    activations = cache[hook_name]
                    if len(getattr(activations, "shape", ())) > 3:
                        activations = activations.flatten(-2, -1)
                    sae_input = move_to_sae_device(activations, source["sae"])
                    feature_acts = source["sae"].encode(sae_input)
                    if getattr(feature_acts, "is_sparse", False):
                        feature_acts = feature_acts.to_dense()
                    feature_rows = feature_acts[0]
                    k = min(top_k, int(feature_rows.shape[-1]))
                    values, indices = self.torch.topk(feature_rows, k=k, dim=-1)
                    value_rows = tensor_to_list(values)
                    index_rows = tensor_to_list(indices)
                    for position in inspected_positions:
                        token = by_position[position]
                        for activation, feature_id in zip(value_rows[position], index_rows[position]):
                            activation_value = float(activation)
                            if activation_value <= 0:
                                continue
                            token["features"].append(
                                {
                                    "feature_id": int(feature_id),
                                    "activation": activation_value,
                                    "sae_id": source["sae_id"],
                                    "hook_name": hook_name,
                                    "layer": source["layer"],
                                }
                            )

                for token in result_tokens:
                    token["features"].sort(key=lambda feature: feature["activation"], reverse=True)
                    token["features"] = token["features"][:top_k]

                return {
                    "model_name": self.config.model_name,
                    "sae_release": self.config.sae_release,
                    "text": text,
                    "prompt": prompt or "",
                    "mode": resolved_mode,
                    "system_prompt": system_prompt or "",
                    "prompt_token_count": prompt_token_count,
                    "token_count": len(result_tokens),
                    "sources": [
                        {
                            "sae_id": source["sae_id"],
                            "hook_name": source["hook_name"],
                            "layer": source["layer"],
                        }
                        for source in sources
                    ],
                    "tokens": result_tokens,
                }
        finally:
            self._lock.release()

    def _generation_prompt_text(self, prompt: str, mode: str, system_prompt: str | None) -> str:
        if mode == "chat":
            return format_chat_prompt(self.model, prompt, system_prompt=system_prompt)
        return prompt

    def _tokens_for_prompt_text(self, prompt_text: str, mode: str) -> Any:
        return model_to_tokens(self.model, prompt_text, prepend_bos=False if mode == "chat" else None)

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


def sae_ids_for_inspection(
    *,
    layers: list[int] | tuple[int, ...],
    sae_id: str | None,
    template: str,
) -> list[str]:
    ids: list[str] = []
    if sae_id and sae_id.strip():
        ids.append(sae_id.strip())
    ids.extend(template.format(layer=layer) for layer in dict.fromkeys(layers))
    return list(dict.fromkeys(ids))


def layer_from_sae_id(sae_id: str) -> int | None:
    for pattern in (r"blocks\.(\d+)\.", r"^(\d+)-"):
        match = re.search(pattern, sae_id)
        if match:
            return int(match.group(1))
    return None


def resolve_generation_mode(mode: str | None, *, model: Any, model_name: str) -> str:
    normalized = (mode or "auto").strip().lower()
    if normalized not in GENERATION_MODES:
        raise SteeringError("mode must be auto, completion, or chat")
    if normalized != "auto":
        return normalized
    if chat_template_available(model) or looks_like_chat_model_name(model_name):
        return "chat"
    return "completion"


def chat_template_available(model: Any) -> bool:
    tokenizer = getattr(model, "tokenizer", None)
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def looks_like_chat_model_name(model_name: str) -> bool:
    normalized = model_name.casefold()
    return any(
        marker in normalized
        for marker in (
            "chat",
            "instruct",
            "-it",
            "_it",
            "/it",
            "-sft",
            "_sft",
            "/sft",
            "rlhf",
            "dpo",
        )
    )


def format_chat_prompt(model: Any, prompt: str, *, system_prompt: str | None = None) -> str:
    clean_system = (system_prompt or "").strip()
    messages = []
    if clean_system:
        messages.append({"role": "system", "content": clean_system})
    messages.append({"role": "user", "content": prompt})

    tokenizer = getattr(model, "tokenizer", None)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template) and getattr(tokenizer, "chat_template", None):
        try:
            return str(apply_template(messages, tokenize=False, add_generation_prompt=True))
        except Exception:
            if clean_system:
                try:
                    return str(
                        apply_template(
                            [{"role": "user", "content": f"{clean_system}\n\n{prompt}"}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                except Exception:
                    pass

    parts = []
    if clean_system:
        parts.append(f"System: {clean_system}")
    parts.append(f"User: {prompt}")
    parts.append("Assistant:")
    return "\n\n".join(parts) + " "


def model_to_tokens(model: Any, text: str, *, prepend_bos: bool | None = None) -> Any:
    if prepend_bos is None:
        return model.to_tokens(text)
    try:
        return model.to_tokens(text, prepend_bos=prepend_bos)
    except TypeError:
        return model.to_tokens(text)


def stop_token_ids(model: Any) -> set[int]:
    tokenizer = getattr(model, "tokenizer", None)
    ids: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        ids.add(int(eos_token_id))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for token in ("<|eot_id|>", "<|end_of_turn|>", "<end_of_turn>", "<|im_end|>"):
            try:
                token_id = convert(token)
            except Exception:
                continue
            if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
                ids.add(token_id)
    return ids


def resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than 0")
    return value


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


def move_to_sae_device(activations: Any, sae: Any) -> Any:
    device = getattr(sae, "device", None)
    if device is not None and hasattr(activations, "to"):
        return activations.to(device)
    return activations


def tensor_to_list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def token_strings(model: Any, token_row: Any, token_ids: list[int]) -> list[str]:
    if hasattr(model, "to_str_tokens"):
        try:
            return [str(token) for token in model.to_str_tokens(token_row)]
        except Exception:
            pass
    return [str(model.to_string([int(token_id)])) for token_id in token_ids]


def make_additive_hook(delta: Any) -> Callable:
    def hook(activation: Any, hook: Any | None = None) -> Any:
        return activation + delta.to(device=activation.device, dtype=activation.dtype)

    return hook
