from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, parse, request

from .state import SteerItem, SteeringState


DEFAULT_NEURONPEDIA_URL = "https://www.neuronpedia.org"
DEFAULT_STEERING_MODEL = "gpt2-small"
DEFAULT_SAE_ID_TEMPLATE = "{layer}-res-jb"
SUPPORTED_STEERING_MODELS = ("gpt2-small", "gemma-2b", "gemma-2b-it")


class NeuronpediaError(RuntimeError):
    """Raised when the Neuronpedia API cannot satisfy a request."""


@dataclass(frozen=True)
class NeuronpediaClient:
    base_url: str = DEFAULT_NEURONPEDIA_URL
    timeout: float = 120.0

    @classmethod
    def from_env(cls, base_url: str | None = None) -> "NeuronpediaClient":
        raw_url = base_url or os.environ.get("NEURONPEDIA_BASE_URL") or DEFAULT_NEURONPEDIA_URL
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url}"
        return cls(base_url=raw_url.rstrip("/"))

    def feature(self, model_id: str, sae_id: str, feature_id: int) -> dict[str, Any]:
        path = "/api/feature/{}/{}/{}".format(
            parse.quote(model_id, safe=""),
            parse.quote(sae_id, safe=""),
            feature_id,
        )
        return self._json_request("GET", path)

    def steer(
        self,
        *,
        prompt: str,
        model_id: str,
        features: list[dict[str, Any]],
        temperature: float,
        n_tokens: int,
        freq_penalty: float,
        seed: int | None,
        strength_multiplier: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "modelId": model_id,
            "features": features,
            "temperature": temperature,
            "n_tokens": n_tokens,
            "freq_penalty": freq_penalty,
            "strength_multiplier": strength_multiplier,
        }
        if seed is not None:
            payload["seed"] = seed
        return self._json_request("POST", "/api/steer", payload)

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise NeuronpediaError(f"Neuronpedia returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise NeuronpediaError(f"could not reach Neuronpedia at {self.base_url}") from exc

        if not body:
            return {}
        data = json.loads(body)
        if not isinstance(data, dict):
            raise NeuronpediaError(f"unexpected Neuronpedia response: {data!r}")
        if "error" in data:
            raise NeuronpediaError(str(data["error"]))
        return data


def state_to_neuronpedia_features(
    state: SteeringState,
    *,
    default_model_id: str,
    sae_id_template: str,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for item in state.items:
        features.extend(
            item_to_neuronpedia_features(
                item,
                default_model_id=default_model_id,
                sae_id_template=sae_id_template,
            )
        )
    return features


def item_to_neuronpedia_features(
    item: SteerItem,
    *,
    default_model_id: str,
    sae_id_template: str,
) -> list[dict[str, Any]]:
    model_id = item.model_id or default_model_id
    if item.sae_id:
        return [
            {
                "modelId": model_id,
                "layer": item.sae_id,
                "index": item.feature_id,
                "strength": item.strength,
            }
        ]

    return [
        {
            "modelId": model_id,
            "layer": sae_id_template.format(layer=layer),
            "index": item.feature_id,
            "strength": item.strength,
        }
        for layer in item.layers
    ]


def summarize_feature(data: dict[str, Any]) -> str:
    model_id = data.get("modelId", "<unknown>")
    sae_id = data.get("layer", "<unknown>")
    feature_id = data.get("index", "<unknown>")
    lines = [f"{model_id}@{sae_id}:{feature_id}"]

    explanations = data.get("explanations")
    if isinstance(explanations, list) and explanations:
        description = explanations[0].get("description")
        if description:
            lines.append(f"explanation: {description}")

    default_strength = data.get("vectorDefaultSteerStrength")
    if default_strength is not None:
        lines.append(f"default steer strength: {default_strength}")

    max_activation = data.get("maxActApprox")
    if max_activation is not None:
        lines.append(f"max activation approx: {max_activation}")

    pos = data.get("pos_str")
    if isinstance(pos, list) and pos:
        lines.append("positive logits: " + ", ".join(str(item) for item in pos[:8]))

    neg = data.get("neg_str")
    if isinstance(neg, list) and neg:
        lines.append("negative logits: " + ", ".join(str(item) for item in neg[:8]))

    activations = data.get("activations")
    if isinstance(activations, list) and activations:
        tokens = activations[0].get("tokens")
        if isinstance(tokens, list):
            snippet = "".join(str(token).replace("Ċ", "\n") for token in tokens)
            lines.append("top activation: " + " ".join(snippet.split()))

    lines.append(f"url: https://www.neuronpedia.org/{model_id}/{sae_id}/{feature_id}")
    return "\n".join(lines)
