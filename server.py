from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import threading
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from steering.feature_cache import (
    FeatureCache,
    FeatureCacheError,
    FeatureLabel,
    NeuronpediaDatasetClient,
    build_source_cache,
    default_feature_cache_path,
)
from steering.state import (
    SteerItem,
    SteeringState,
    SteeringError,
    clear_state,
    default_state_path,
    load_state,
    update_state,
)
from steering.tlens_backend import BackendConfig, TransformerLensSteeringBackend, looks_like_chat_model_name


app = FastAPI(title="Steering CLI TransformerLens Backend")
backend: TransformerLensSteeringBackend | None = None
backend_lock = threading.Lock()
state_path = Path(default_state_path())
feature_cache_path = Path(default_feature_cache_path())


KNOWN_MODEL_PARAM_COUNTS: dict[str, int] = {
    "ai-forever/mGPT": 1_300_000_000,
    "bigcode/santacoder": 1_100_000_000,
    "distilgpt2": 82_000_000,
    "facebook/hubert-base-ls960": 95_000_000,
    "facebook/wav2vec2-base": 95_000_000,
    "facebook/wav2vec2-large": 317_000_000,
    "google-bert/bert-base-cased": 110_000_000,
    "google-bert/bert-base-uncased": 110_000_000,
    "google-bert/bert-large-cased": 340_000_000,
    "google-bert/bert-large-uncased": 340_000_000,
    "google-t5/t5-base": 220_000_000,
    "google-t5/t5-large": 770_000_000,
    "google-t5/t5-small": 60_000_000,
    "gpt2": 124_000_000,
    "gpt2-large": 774_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-xl": 1_558_000_000,
    "microsoft/phi-1": 1_300_000_000,
    "microsoft/phi-1_5": 1_300_000_000,
    "microsoft/phi-2": 2_700_000_000,
    "microsoft/phi-4": 14_000_000_000,
    "microsoft/Phi-3-mini-4k-instruct": 3_800_000_000,
    "mistralai/Mistral-Nemo-Base-2407": 12_000_000_000,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46_700_000_000,
    "mistralai/Mixtral-8x7B-v0.1": 46_700_000_000,
}


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=60, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    seed: int | None = None
    stream: bool = True
    steers_enabled: bool = True
    mode: str = "auto"
    system_prompt: str | None = None
    stop_on_eos: bool | None = None

    @staticmethod
    def clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class InspectTokensRequest(BaseModel):
    text: str = Field(min_length=1)
    prompt: str | None = None
    layers: list[int] = Field(default_factory=list)
    sae_id: str | None = None
    cache_model_id: str | None = None
    cache_source_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    include_prompt: bool = False
    mode: str = "auto"
    system_prompt: str | None = None

    @staticmethod
    def clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class LoadModelRequest(BaseModel):
    model_name: str = Field(min_length=1)
    sae_release: str | None = Field(default=None, min_length=1)
    sae_id_template: str | None = Field(default=None, min_length=1)
    clear_steers: bool = False

    @staticmethod
    def clean_text(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return value


class SteerItemRequest(BaseModel):
    feature_id: int = Field(ge=0)
    strength: float = Field(allow_inf_nan=False)
    layers: list[int] = Field(default_factory=list)
    label: str | None = None
    model_id: str | None = None
    sae_id: str | None = None
    append: bool = False

    @staticmethod
    def clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def to_item(self) -> SteerItem:
        return SteerItem(
            feature_id=self.feature_id,
            strength=self.strength,
            layers=tuple(dict.fromkeys(self.layers)),
            label=self.clean_optional_text(self.label),
            model_id=self.clean_optional_text(self.model_id),
            sae_id=self.clean_optional_text(self.sae_id),
        )


class DownloadSourceRequest(BaseModel):
    model_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    max_files: int | None = Field(default=None, ge=1)


@app.on_event("startup")
def startup() -> None:
    global backend
    backend = TransformerLensSteeringBackend(BackendConfig.from_env(state_path=state_path))


@app.get("/", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    return HTMLResponse(WEB_UI_HTML)


@app.get("/health")
def health() -> dict:
    return get_backend().health()


@app.get("/api/state")
def get_state() -> dict:
    return load_state(state_path).to_dict()


@app.post("/api/state/items")
def set_state_item(request: SteerItemRequest) -> dict:
    try:
        state = update_state(request.to_item(), append=request.append, path=state_path)
    except SteeringError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return state.to_dict()


@app.delete("/api/state")
def delete_state() -> dict:
    return clear_state(state_path).to_dict()


@app.get("/api/neuronpedia/models")
def neuronpedia_models() -> dict:
    try:
        models = NeuronpediaDatasetClient().list_models()
    except FeatureCacheError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"models": models, "count": len(models)}


@app.get("/api/neuronpedia/sources")
def neuronpedia_sources(
    model_id: str = Query(min_length=1),
    contains: str | None = None,
) -> dict:
    try:
        sources = NeuronpediaDatasetClient().list_sources(model_id.strip())
    except FeatureCacheError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if contains:
        needle = contains.casefold()
        sources = [source for source in sources if needle in source.casefold()]
    return {"model_id": model_id.strip(), "sources": sources, "count": len(sources)}


@app.get("/api/cache/status")
def cache_status() -> dict:
    rows = FeatureCache(feature_cache_path).status()
    return {"sources": [cached_source_to_dict(row) for row in rows], "count": len(rows)}


@app.post("/api/cache/source")
def cache_source(request: DownloadSourceRequest) -> dict:
    model_id = request.model_id.strip()
    source_id = request.source_id.strip()
    if not model_id or not source_id:
        raise HTTPException(status_code=400, detail="model_id and source_id are required")
    try:
        cached = build_source_cache(
            model_id=model_id,
            source_id=source_id,
            cache_path=feature_cache_path,
            max_files=request.max_files,
        )
    except FeatureCacheError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return cached_source_to_dict(cached)


@app.get("/api/cache/search")
def cache_search(
    query: str = Query(min_length=1),
    model_id: str | None = None,
    source_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict:
    try:
        labels = FeatureCache(feature_cache_path).search(
            query,
            model_id=model_id.strip() if model_id else None,
            source_id=source_id.strip() if source_id else None,
            limit=limit,
        )
    except FeatureCacheError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"labels": [feature_label_to_dict(label) for label in labels], "count": len(labels)}


@app.post("/api/model")
def load_model(request: LoadModelRequest) -> dict:
    global backend
    current = get_backend()
    model_name = LoadModelRequest.clean_text(request.model_name)
    if model_name is None:
        raise HTTPException(status_code=400, detail="model_name is required")
    sae_release = LoadModelRequest.clean_text(request.sae_release) or f"{model_name}-res-jb"
    sae_id_template = LoadModelRequest.clean_text(request.sae_id_template) or current.config.sae_id_template
    config = BackendConfig(
        model_name=model_name,
        sae_release=sae_release,
        sae_id_template=sae_id_template,
        device=current.config.device,
        generation_lock_timeout=current.config.generation_lock_timeout,
        state_path=state_path,
    )
    try:
        replacement = TransformerLensSteeringBackend(config)
        if request.clear_steers:
            clear_state(state_path)
    except (Exception, SteeringError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with backend_lock:
        backend = replacement
    return replacement.health()


@app.get("/api/model/options")
def model_options() -> dict:
    return {
        "models": list(cached_model_options()),
        "count": len(cached_model_options()),
        "size_basis": "Estimated checkpoint size from model parameter counts; actual Hugging Face downloads may differ.",
    }


@app.post("/api/inspect/tokens")
def inspect_tokens(request: InspectTokensRequest) -> dict:
    if any(layer < 0 for layer in request.layers):
        raise HTTPException(status_code=400, detail="layers must be non-negative")
    try:
        data = get_backend().inspect_tokens(
            request.text,
            prompt=InspectTokensRequest.clean_optional_text(request.prompt),
            layers=request.layers,
            sae_id=InspectTokensRequest.clean_optional_text(request.sae_id),
            top_k=request.top_k,
            include_prompt=request.include_prompt,
            mode=request.mode,
            system_prompt=InspectTokensRequest.clean_optional_text(request.system_prompt),
        )
        return enrich_inspection_with_cached_labels(
            data,
            cache_model_id=InspectTokensRequest.clean_optional_text(request.cache_model_id),
            cache_source_id=InspectTokensRequest.clean_optional_text(request.cache_source_id),
        )
    except SteeringError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        state_override = None if request.steers_enabled else SteeringState.empty()
        tokens = get_backend().generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
            state_override=state_override,
            mode=request.mode,
            system_prompt=GenerateRequest.clean_optional_text(request.system_prompt),
            stop_on_eos=request.stop_on_eos,
        )
        if request.stream:
            return StreamingResponse(stream_tokens(tokens), media_type="text/plain")
        return {"text": "".join(tokens)}
    except SteeringError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def get_backend() -> TransformerLensSteeringBackend:
    if backend is None:
        raise HTTPException(status_code=503, detail="backend is still starting")
    return backend


def stream_tokens(tokens: Iterator[str]) -> Iterator[bytes]:
    for token in tokens:
        yield token.encode("utf-8")


def cached_source_to_dict(row) -> dict:
    return {
        "model_id": row.model_id,
        "source_id": row.source_id,
        "label_count": row.label_count,
        "feature_count": row.feature_count,
        "fetched_at": row.fetched_at,
    }


def feature_label_to_dict(label) -> dict:
    return {
        "model_id": label.model_id,
        "source_id": label.source_id,
        "feature_id": label.feature_id,
        "description": label.description,
        "type_name": label.type_name,
        "explanation_model_name": label.explanation_model_name,
    }


def enrich_inspection_with_cached_labels(
    data: dict,
    *,
    cache_model_id: str | None = None,
    cache_source_id: str | None = None,
) -> dict:
    model_ids = inspection_label_model_candidates(data.get("model_name"), cache_model_id)
    if not model_ids:
        return data
    cache = FeatureCache(feature_cache_path)
    cached_sources = cached_source_ids_by_model(cache, model_ids)
    label_cache: dict[tuple[str, int, tuple[str, ...]], list[FeatureLabel]] = {}
    for token in data.get("tokens", []):
        for feature in token.get("features", []):
            feature_id = feature.get("feature_id")
            if feature_id is None:
                continue
            labels: list[FeatureLabel] = []
            source_candidates = inspection_label_source_candidates(feature, cache_source_id=cache_source_id)
            for model_id in model_ids:
                layer_sources = cached_sources_for_feature_layer(
                    cached_sources.get(model_id, []),
                    feature.get("layer"),
                    exclude=source_candidates,
                )
                for source_ids in (source_candidates, layer_sources):
                    if not source_ids:
                        continue
                    key = (model_id, int(feature_id), tuple(source_ids))
                    if key not in label_cache:
                        label_cache[key] = cache.get_feature_labels(
                            model_id=model_id,
                            feature_id=int(feature_id),
                            source_ids=source_ids,
                        )
                        label_cache[key] = rank_cached_feature_labels(label_cache[key], source_ids)
                    labels = label_cache[key]
                    if labels:
                        break
                if labels:
                    break
            feature["label_lookup"] = {
                "status": "cached" if labels else "not_found",
                "model_ids": model_ids,
                "source_ids": source_candidates,
            }
            if labels:
                feature["source_id"] = labels[0].source_id
                feature["label_model_id"] = labels[0].model_id
                feature["description"] = labels[0].description
                feature["labels"] = [feature_label_to_dict(label) for label in labels[:3]]
    return data


def inspection_label_model_candidates(model_name: object, cache_model_id: str | None = None) -> list[str]:
    candidates: list[str] = []
    for value in (cache_model_id, model_name):
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for value in tuple(candidates):
        if value == "gpt2":
            candidates.append("gpt2-small")
        if value == "gpt2-small":
            candidates.append("gpt2")
        if "/" in value:
            candidates.append(value.rsplit("/", 1)[-1])
    return list(dict.fromkeys(candidates))


def inspection_label_source_candidates(feature: dict, *, cache_source_id: str | None = None) -> list[str]:
    candidates: list[str] = []
    layer = feature.get("layer")
    if cache_source_id and source_id_matches_layer(cache_source_id, layer):
        candidates.append(cache_source_id)
    source_id = feature.get("source_id")
    if isinstance(source_id, str) and source_id.strip() and source_id_matches_layer(source_id, layer):
        candidates.append(source_id.strip())
    if isinstance(layer, int):
        candidates.append(f"{layer}-res-jb")
    for key in ("sae_id", "hook_name"):
        source_id = feature.get(key)
        if isinstance(source_id, str) and source_id.strip() and source_id_matches_layer(source_id, layer):
            candidates.append(source_id.strip())
    return list(dict.fromkeys(candidates))


def cached_source_ids_by_model(cache: FeatureCache, model_ids: list[str]) -> dict[str, list[str]]:
    wanted = set(model_ids)
    sources: dict[str, list[str]] = {model_id: [] for model_id in model_ids}
    for row in cache.status():
        if row.model_id in wanted:
            sources.setdefault(row.model_id, []).append(row.source_id)
    return sources


def cached_sources_for_feature_layer(source_ids: list[str], layer: object, *, exclude: list[str]) -> list[str]:
    if not isinstance(layer, int):
        return []
    excluded = set(exclude)
    return [
        source_id
        for source_id in source_ids
        if source_id not in excluded and source_id_layer(source_id) == layer
    ]


def source_id_matches_layer(source_id: str, layer: object) -> bool:
    source_layer = source_id_layer(source_id)
    return not isinstance(layer, int) or source_layer is None or source_layer == layer


def source_id_layer(source_id: str) -> int | None:
    for pattern in (r"blocks\.(\d+)\.", r"^(\d+)-"):
        match = re.search(pattern, source_id)
        if match:
            return int(match.group(1))
    return None


def rank_cached_feature_labels(labels: list[FeatureLabel], preferred_sources: list[str]) -> list[FeatureLabel]:
    source_rank = {source_id: index for index, source_id in enumerate(preferred_sources)}
    return sorted(
        labels,
        key=lambda label: (
            source_rank.get(label.source_id, len(source_rank)),
            label.explanation_model_name or "",
            label.type_name or "",
            label.description,
        ),
    )


@lru_cache(maxsize=1)
def cached_model_options() -> tuple[dict, ...]:
    try:
        from transformer_lens.supported_models import MODEL_ALIASES, OFFICIAL_MODEL_NAMES
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="TransformerLens is not installed") from exc

    rows = []
    for official_name in OFFICIAL_MODEL_NAMES:
        aliases = list(MODEL_ALIASES.get(official_name, []))
        load_name = preferred_model_load_name(official_name, aliases)
        params = estimate_parameter_count(official_name)
        download_bytes = estimate_checkpoint_bytes(official_name, params)
        is_chat_model = any(looks_like_chat_model_name(name) for name in (official_name, load_name, *aliases))
        rows.append(
            {
                "model_name": official_name,
                "load_name": load_name,
                "aliases": aliases,
                "parameter_count": params,
                "parameter_count_label": format_count(params, suffix=" params"),
                "estimated_download_bytes": download_bytes,
                "download_size_label": format_bytes(download_bytes),
                "size_source": "known" if official_name in KNOWN_MODEL_PARAM_COUNTS else "estimated",
                "sae_release_guess": f"{load_name.split('/')[-1]}-res-jb",
                "is_chat_model": is_chat_model,
            }
        )
    return tuple(sorted(rows, key=model_option_sort_key))


def model_option_sort_key(row: dict) -> tuple[int, int, str]:
    size = row["estimated_download_bytes"]
    return (0 if size is not None else 1, size or 0, row["model_name"].casefold())


def preferred_model_load_name(official_name: str, aliases: list[str]) -> str:
    if official_name == "gpt2" and "gpt2-small" in aliases:
        return "gpt2-small"
    return official_name


def estimate_parameter_count(model_name: str) -> int | None:
    if model_name in KNOWN_MODEL_PARAM_COUNTS:
        return KNOWN_MODEL_PARAM_COUNTS[model_name]

    normalized = model_name.lower().replace("_", ".")
    if "gpt2-small" in normalized:
        return KNOWN_MODEL_PARAM_COUNTS["gpt2"]
    if "gpt2-medium" in normalized:
        return KNOWN_MODEL_PARAM_COUNTS["gpt2-medium"]

    candidates: list[int] = []
    for match in re.finditer(r"(?<![a-z0-9])(\d+(?:\.\d+)?)([mb])(?=\b|[^a-z])", normalized):
        value = float(match.group(1))
        multiplier = {"m": 1_000_000, "b": 1_000_000_000}[match.group(2)]
        candidates.append(int(value * multiplier))
    for match in re.finditer(r"(?<![a-z0-9])(\d+)b(\d+)(?=\b|[^a-z])", normalized):
        candidates.append(int(float(f"{match.group(1)}.{match.group(2)}") * 1_000_000_000))
    if not candidates:
        return None
    return max(candidates)


def estimate_checkpoint_bytes(model_name: str, parameter_count: int | None) -> int | None:
    if parameter_count is None:
        return None
    lower_name = model_name.lower()
    legacy_fp32 = (
        parameter_count < 1_000_000_000
        or lower_name.startswith("gpt2")
        or lower_name in {"distilgpt2", "ai-forever/mgpt"}
    )
    bytes_per_param = 4 if legacy_fp32 else 2
    return int(parameter_count * bytes_per_param)


def format_count(value: int | None, *, suffix: str = "") -> str:
    if value is None:
        return "Unknown"
    for threshold, unit in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if value >= threshold:
            amount = value / threshold
            label = f"{amount:.1f}".rstrip("0").rstrip(".")
            return f"{label}{unit}{suffix}"
    return f"{value}{suffix}"


def format_bytes(value: int | None) -> str:
    if value is None:
        return "Unknown"
    for threshold, unit in (
        (1024**4, "TB"),
        (1024**3, "GB"),
        (1024**2, "MB"),
        (1024, "KB"),
    ):
        if value >= threshold:
            amount = value / threshold
            label = f"{amount:.1f}".rstrip("0").rstrip(".")
            return f"~{label} {unit}"
    return f"~{value} B"


WEB_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Steering</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f5f4;
      --panel: #ffffff;
      --panel-soft: #f7faf8;
      --text: #17201f;
      --muted: #66736f;
      --line: #d7dedb;
      --line-strong: #bcc8c3;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --accent-soft: #e4f4f1;
      --warning: #9a5b13;
      --danger: #b42318;
      --danger-strong: #8f1f17;
      --ok: #067647;
      --ink: #101828;
      --shadow: 0 10px 28px rgba(16, 24, 40, .06);
      --radius: 8px;
    }

    * { box-sizing: border-box; }

    body {
      min-width: 320px;
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px;
      line-height: 1.4;
    }

    button, input, select, textarea { font: inherit; }

    .app-bar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto minmax(360px, auto);
      gap: 18px;
      align-items: center;
      padding: 10px 18px;
      background: rgba(255, 255, 255, .96);
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 8px rgba(16, 24, 40, .05);
      backdrop-filter: blur(12px);
    }

    .brand {
      display: flex;
      gap: 12px;
      align-items: center;
      min-width: 0;
    }

    .mark {
      display: grid;
      place-items: center;
      width: 32px;
      height: 32px;
      border-radius: var(--radius);
      background: var(--ink);
      color: #fff;
      font-weight: 800;
      letter-spacing: 0;
    }

    h1, h2, h3, p { margin: 0; }

    header h1 {
      font-size: 17px;
      font-weight: 720;
      letter-spacing: 0;
    }

    #health {
      margin-top: 1px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .health-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(76px, auto));
      gap: 8px;
      justify-content: end;
    }

    .layout-controls {
      display: inline-flex;
      gap: 7px;
      justify-self: end;
      align-items: center;
    }

    .icon-button {
      display: inline-grid;
      place-items: center;
      width: 34px;
      min-width: 34px;
      padding: 0;
      color: #22312e;
    }

    .icon-button svg {
      width: 18px;
      height: 18px;
      stroke: currentColor;
      stroke-width: 2;
      stroke-linecap: round;
      stroke-linejoin: round;
      fill: none;
      pointer-events: none;
    }

    .icon-button[aria-pressed="true"] {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent-strong);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .11);
    }

    .metric {
      min-width: 76px;
      padding: 6px 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-soft);
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .04em;
      text-transform: uppercase;
    }

    .metric strong {
      display: block;
      max-width: 180px;
      margin-top: 1px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--text);
      font-size: 12px;
      font-weight: 700;
    }

    main,
    main.research-workspace {
      display: grid;
      grid-template-columns: minmax(700px, 1fr) minmax(420px, 520px);
      gap: 14px;
      max-width: 1680px;
      margin: 0 auto;
      padding: 14px 18px 28px;
    }

    .stack {
      display: grid;
      gap: 14px;
      align-content: start;
    }

    .side-stack {
      align-self: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    main.window-layout-canvas {
      position: relative;
      display: block;
      width: 100%;
      max-width: none;
      min-height: calc(100vh - 66px);
      margin: 0;
    }

    main.window-layout-canvas .stack {
      display: contents;
    }

    main.window-layout-canvas .window-panel {
      position: absolute;
      display: flex;
      min-width: 280px;
      min-height: 180px;
      flex-direction: column;
      overflow: hidden;
    }

    main.window-layout-canvas .window-panel .panel-body {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
    }

    body.window-layout-editing {
      user-select: none;
    }

    body.window-layout-editing main.window-layout-canvas .window-panel {
      outline: 2px solid rgba(15, 118, 110, .24);
      outline-offset: 2px;
    }

    body.window-layout-editing main.window-layout-canvas .window-panel .panel-header {
      cursor: move;
    }

    body.window-layout-editing main.window-layout-canvas .window-panel .panel-header button,
    body.window-layout-editing main.window-layout-canvas .window-panel .panel-header input,
    body.window-layout-editing main.window-layout-canvas .window-panel .panel-header select {
      cursor: pointer;
    }

    .resize-handle {
      position: absolute;
      z-index: 4;
      display: none;
      width: 22px;
      height: 22px;
      min-width: 0;
      min-height: 0;
      border: 0;
      background: transparent;
      padding: 0;
    }

    body.window-layout-editing main.window-layout-canvas .resize-handle {
      display: block;
    }

    .resize-handle:hover:not(:disabled) {
      background: transparent;
    }

    .resize-handle::after {
      content: "";
      position: absolute;
      width: 8px;
      height: 8px;
      border-color: var(--accent-strong);
      opacity: .8;
    }

    .resize-handle.nw { top: 0; left: 0; cursor: nwse-resize; }
    .resize-handle.ne { top: 0; right: 0; cursor: nesw-resize; }
    .resize-handle.sw { bottom: 0; left: 0; cursor: nesw-resize; }
    .resize-handle.se { right: 0; bottom: 0; cursor: nwse-resize; }
    .resize-handle.nw::after { top: 4px; left: 4px; border-top: 2px solid; border-left: 2px solid; }
    .resize-handle.ne::after { top: 4px; right: 4px; border-top: 2px solid; border-right: 2px solid; }
    .resize-handle.sw::after { bottom: 4px; left: 4px; border-bottom: 2px solid; border-left: 2px solid; }
    .resize-handle.se::after { right: 4px; bottom: 4px; border-right: 2px solid; border-bottom: 2px solid; }

    body.window-layout-dragging,
    body.window-layout-dragging * {
      cursor: grabbing !important;
    }

    .panel-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }

    .panel-header h2 {
      font-size: 14px;
      font-weight: 760;
      letter-spacing: 0;
    }

    .panel-header p {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }

    .panel-body {
      padding: 12px;
    }

    .generate-body {
      padding: 12px;
    }

    .generate-grid {
      display: grid;
      grid-template-columns: minmax(320px, .95fr) minmax(360px, 1.05fr);
      gap: 12px;
      align-items: start;
    }

    .compose-column,
    .inspect-column {
      display: grid;
      gap: 10px;
      min-width: 0;
    }

    .compact-grid {
      gap: 8px;
    }

    .compose-column .compact-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .compose-column .wide-field {
      grid-column: 1 / -1;
    }

    .inspect-column .compact-grid {
      grid-template-columns: repeat(2, minmax(150px, 1fr));
    }

    .inspect-controls {
      display: grid;
      gap: 8px;
      align-items: start;
    }

    .inspect-controls .check-help-row {
      margin-top: 0;
      padding-bottom: 0;
      white-space: nowrap;
    }

    .primary-actions {
      margin-top: 0;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .field-row {
      display: grid;
      gap: 5px;
    }

    label {
      color: #33423f;
      font-size: 12px;
      font-weight: 700;
    }

    .label-with-help {
      position: relative;
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      align-items: center;
      gap: 6px;
    }

    .check-help-row {
      position: relative;
      display: inline-flex;
      width: fit-content;
      max-width: 100%;
      align-items: center;
      gap: 6px;
      margin-top: 9px;
    }

    .help-dot {
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      min-height: 18px;
      padding: 0;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      line-height: 1;
      cursor: help;
    }

    .help-dot:hover,
    .help-dot:focus {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent-strong);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .1);
    }

    .help-text {
      position: absolute;
      left: 0;
      top: calc(100% + 7px);
      z-index: 25;
      display: none;
      width: min(340px, calc(100vw - 36px));
      padding: 9px 10px;
      border: 1px solid rgba(148, 163, 184, .32);
      border-radius: var(--radius);
      background: #0b1220;
      color: #e2e8f0;
      box-shadow: 0 16px 34px rgba(15, 23, 42, .22);
      font-size: 12px;
      font-weight: 520;
      line-height: 1.38;
    }

    .label-with-help:hover .help-text,
    .check-help-row:hover .help-text,
    .help-dot:focus + .help-text {
      display: block;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      min-height: 34px;
      padding: 7px 9px;
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease, background .16s ease;
    }

    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .13);
    }

    textarea {
      min-height: 238px;
      resize: vertical;
      line-height: 1.48;
    }

    .prompt-field textarea {
      min-height: 272px;
    }

    button {
      min-height: 34px;
      border: 1px solid var(--accent-strong);
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 7px 10px;
      font-weight: 720;
      cursor: pointer;
      white-space: nowrap;
    }

    button:hover:not(:disabled) { background: var(--accent-strong); }
    button:disabled { opacity: .5; cursor: wait; }
    button.secondary { color: #22312e; background: #fff; border-color: var(--line-strong); }
    button.secondary:hover:not(:disabled) { background: #f3f7f5; }
    button.danger { background: var(--danger); border-color: var(--danger-strong); }
    button.danger:hover:not(:disabled) { background: var(--danger-strong); }
    button.small { min-height: 30px; padding: 5px 9px; font-size: 12px; }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
      margin-top: 10px;
    }

    .inline-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
    }

    .status {
      min-height: 20px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }

    .status.error { color: var(--danger); }
    .status.ok { color: var(--ok); }
    .status.warn { color: var(--warning); }

    .output-shell {
      border: 1px solid #202939;
      border-radius: var(--radius);
      overflow: hidden;
      background: var(--ink);
    }

    .output-meta {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, .1);
      color: #cbd5e1;
      font-size: 12px;
    }

    pre {
      min-height: 180px;
      max-height: 42vh;
      margin: 0;
      padding: 13px;
      overflow: auto;
      background: var(--ink);
      color: #f8fafc;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.48;
    }

    .token-output {
      min-height: 180px;
      max-height: 42vh;
      margin: 0;
      padding: 13px;
      overflow: auto;
      background: var(--ink);
      color: #f8fafc;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 13px;
    }

    .output-shell pre,
    .output-shell .token-output {
      min-height: 380px;
      max-height: 56vh;
    }

    .token-chip {
      display: inline;
      border-bottom: 1px solid rgba(20, 184, 166, .45);
      border-radius: 3px;
      cursor: help;
      outline: 0;
    }

    .token-chip.prompt-token {
      border-bottom-color: rgba(203, 213, 225, .32);
      color: #cbd5e1;
    }

    .token-chip:hover,
    .token-chip:focus {
      background: rgba(20, 184, 166, .24);
      color: #fff;
    }

    .token-popover {
      position: fixed;
      z-index: 20;
      width: min(360px, calc(100vw - 24px));
      max-height: min(520px, calc(100vh - 24px));
      padding: 10px;
      overflow: auto;
      border: 1px solid rgba(148, 163, 184, .35);
      border-radius: var(--radius);
      background: #0b1220;
      color: #f8fafc;
      box-shadow: 0 18px 40px rgba(15, 23, 42, .28);
      font-size: 12px;
    }

    .token-popover strong {
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .token-popover .muted { color: #94a3b8; }

    .feature-list {
      display: grid;
      gap: 5px;
      margin-top: 8px;
    }

    .feature-hit {
      display: grid;
      grid-template-columns: minmax(78px, auto) 1fr;
      gap: 4px 8px;
      width: 100%;
      padding: 6px;
      border: 1px solid rgba(148, 163, 184, .22);
      border-radius: 6px;
      background: rgba(15, 23, 42, .78);
      color: inherit;
      font: inherit;
      text-align: left;
      cursor: pointer;
    }

    .feature-hit:hover,
    .feature-hit:focus {
      border-color: rgba(94, 234, 212, .62);
      background: rgba(20, 83, 77, .4);
    }

    .feature-hit b { color: #5eead4; }
    .feature-hit span { color: #cbd5e1; overflow-wrap: anywhere; }
    .feature-detail { grid-column: 1 / -1; }
    .feature-source {
      grid-column: 1 / -1;
      color: #94a3b8 !important;
      font-size: 11px;
    }

    .table-wrap {
      max-height: 240px;
      margin-top: 9px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: var(--radius);
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f7faf8;
      color: #475467;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .04em;
      text-transform: uppercase;
    }

    tbody tr:last-child td { border-bottom: 0; }
    tr.selectable { cursor: pointer; }
    tr.selectable:hover { background: #f1f8f6; }
    tr.selected { background: var(--accent-soft); }
    td.compact { width: 1%; white-space: nowrap; }
    td.wrap { overflow-wrap: anywhere; }

    .model-picker {
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }

    .model-table { max-height: 210px; }

    .model-name {
      display: grid;
      gap: 2px;
      overflow-wrap: anywhere;
    }

    .model-name strong { font-size: 13px; }

    .alias-list {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }

    .run-table { max-height: 280px; }
    .run-note { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }

    .mode-pill {
      display: inline-block;
      min-width: 64px;
      padding: 3px 7px;
      border: 1px solid var(--line-strong);
      border-radius: 999px;
      background: #fff;
      color: #2b3a37;
      text-align: center;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }

    .mode-pill.baseline {
      border-color: #b7c6d8;
      background: #eef5ff;
      color: #24476f;
    }

    .mode-pill.steered {
      border-color: #abd6ce;
      background: var(--accent-soft);
      color: var(--accent-strong);
    }

    .empty {
      padding: 14px;
      border: 1px dashed var(--line-strong);
      border-radius: var(--radius);
      color: var(--muted);
      background: #fbfcfc;
      text-align: center;
    }

    .steer-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }

    .steer-row {
      display: grid;
      grid-template-columns: minmax(92px, auto) 1fr;
      gap: 8px 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfc;
    }

    .steer-row strong { font-size: 13px; }
    .steer-row span { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }

    details {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfcfc;
    }

    summary {
      padding: 8px 10px;
      color: var(--muted);
      cursor: pointer;
      font-weight: 700;
      font-size: 12px;
    }

    details pre {
      min-height: 120px;
      max-height: 220px;
      border-top: 1px solid var(--line);
      border-radius: 0;
      background: #111827;
    }

    .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }

    .check-row input {
      width: auto;
      min-width: 16px;
      height: 16px;
      padding: 0;
      box-shadow: none;
    }

    .check-help-row .help-dot {
      flex: 0 0 auto;
    }

    .muted { color: var(--muted); }
    .hide { display: none !important; }

    @media (max-width: 1180px) {
      main,
      main.research-workspace { grid-template-columns: 1fr; }
      .app-bar { grid-template-columns: 1fr auto; }
      .health-grid { grid-column: 1 / -1; }
      .health-grid { justify-content: stretch; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric strong { max-width: none; }
    }

    @media (max-width: 760px) {
      .generate-grid { grid-template-columns: 1fr; }
      .compose-column .compact-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .output-shell pre,
      .output-shell .token-output { min-height: 300px; }
    }

    @media (max-width: 620px) {
      main,
      main.research-workspace { padding: 10px; }
      .app-bar { padding: 10px; }
      .form-grid { grid-template-columns: 1fr; }
      .compose-column .compact-grid,
      .inspect-column .compact-grid { grid-template-columns: 1fr; }
      .panel-header { display: grid; }
      .actions button, .inline-actions button { flex: 1 1 auto; }
      .prompt-field textarea { min-height: 190px; }
      .output-shell pre,
      .output-shell .token-output { min-height: 260px; }
      pre { max-height: 360px; }
      .token-output { max-height: 360px; }
    }
  </style>
</head>
<body>
  <header class="app-bar">
    <div class="brand">
      <div class="mark" aria-hidden="true">S</div>
      <div>
        <h1>Steering</h1>
        <div id="health">Checking backend...</div>
      </div>
    </div>
    <div class="layout-controls" aria-label="Window layout controls">
      <button class="secondary icon-button" id="moveWindows" type="button" aria-pressed="false" aria-label="Move and resize windows" title="Move and resize windows">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2v20"></path>
          <path d="m15 19-3 3-3-3"></path>
          <path d="m9 5 3-3 3 3"></path>
          <path d="M2 12h20"></path>
          <path d="m5 9-3 3 3 3"></path>
          <path d="m19 9 3 3-3 3"></path>
        </svg>
      </button>
      <button class="secondary icon-button" id="resetWindowLayout" type="button" aria-label="Reset window layout" title="Reset window layout">
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 12a9 9 0 1 0 3-6.7"></path>
          <path d="M3 4v6h6"></path>
        </svg>
      </button>
    </div>
    <div class="health-grid" aria-label="Backend status">
      <div class="metric"><span>Model</span><strong id="healthModel">-</strong></div>
      <div class="metric"><span>Device</span><strong id="healthDevice">-</strong></div>
      <div class="metric"><span>SAE</span><strong id="healthSae">-</strong></div>
      <div class="metric"><span>Status</span><strong id="healthBusy">-</strong></div>
    </div>
  </header>

  <main id="workspace" class="research-workspace">
    <div class="stack primary-stack">
      <section class="panel generate-panel window-panel" data-window-id="generate" aria-labelledby="generateTitle">
        <div class="panel-header">
          <div>
            <h2 id="generateTitle">Generate</h2>
            <p id="generateMeta">Ready</p>
          </div>
          <div class="inline-actions">
            <button class="secondary small" id="inspectOutput" type="button">Inspect output</button>
            <button class="secondary small" id="copyOutput" type="button">Copy output</button>
            <button class="secondary small" id="resetDefaults" type="button">Reset defaults</button>
          </div>
        </div>
        <div class="panel-body generate-body">
          <div class="generate-grid">
            <div class="compose-column">
              <div class="field-row prompt-field">
                <label for="prompt">Prompt</label>
                <textarea id="prompt">Today the weather report says</textarea>
              </div>
              <div class="form-grid compact-grid">
                <div class="field-row">
                  <label for="maxTokens">Max tokens</label>
                  <input id="maxTokens" type="number" min="1" max="512" value="40">
                </div>
                <div class="field-row">
                  <label for="temperature">Temperature</label>
                  <input id="temperature" type="number" min="0" step="0.1" value="0">
                </div>
                <div class="field-row">
                  <label for="generationMode">Response mode</label>
                  <select id="generationMode">
                    <option value="auto">Auto</option>
                    <option value="chat">Chat</option>
                    <option value="completion">Completion</option>
                  </select>
                </div>
                <div class="field-row wide-field">
                  <label for="systemPrompt">System prompt</label>
                  <input id="systemPrompt" value="You are a helpful research assistant.">
                </div>
              </div>
              <div class="actions primary-actions">
                <button id="generate" type="button">Generate</button>
                <button class="secondary" id="generateBaseline" type="button">Baseline</button>
                <button class="secondary" id="compareRuns" type="button">Compare</button>
                <button class="secondary" id="stopGenerate" type="button" disabled>Stop</button>
                <button class="secondary" id="refreshState" type="button">Refresh state</button>
              </div>
              <div class="status" id="generateStatus" role="status" aria-live="polite"></div>
            </div>
            <div class="inspect-column">
              <div class="inspect-controls">
                <div class="form-grid compact-grid">
                  <div class="field-row">
                    <label for="inspectLayers">Inspect layers</label>
                    <input id="inspectLayers" value="6">
                  </div>
                  <div class="field-row">
                    <label for="inspectTopK">Top features</label>
                    <input id="inspectTopK" type="number" min="1" max="20" value="5">
                  </div>
                </div>
                <label class="check-row" for="inspectPrompt">
                  <input id="inspectPrompt" type="checkbox">
                  Include prompt tokens
                </label>
              </div>
              <div class="output-shell">
                <div class="output-meta">
                  <span id="outputTokens">0 chars</span>
                  <span id="outputMode">streaming off</span>
                </div>
                <pre id="output" aria-live="polite"></pre>
                <div id="tokenOutput" class="token-output hide" aria-live="polite"></div>
              </div>
              <div id="tokenPopover" class="token-popover hide"></div>
              <div class="status" id="inspectStatus" role="status" aria-live="polite"></div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel research-panel window-panel" data-window-id="research" aria-labelledby="researchTitle">
        <div class="panel-header">
          <div>
            <h2 id="researchTitle">Research Runs</h2>
            <p id="researchMeta">No runs recorded</p>
          </div>
        </div>
        <div class="panel-body">
          <div class="field-row">
            <label for="runNote">Run note</label>
            <input id="runNote" placeholder="hypothesis, feature source, expected behavior">
          </div>
          <div class="actions">
            <button class="secondary" id="copyLatestRun" type="button">Copy latest JSON</button>
            <button class="secondary" id="exportRuns" type="button">Export JSON</button>
            <button class="danger" id="clearRuns" type="button">Clear log</button>
          </div>
          <div class="table-wrap run-table" aria-label="Research run log">
            <table>
              <thead><tr><th>Time</th><th>Mode</th><th>Model</th><th>Output</th></tr></thead>
              <tbody id="researchRuns"></tbody>
            </table>
          </div>
          <div class="status" id="researchStatus" role="status" aria-live="polite"></div>
        </div>
      </section>
    </div>

    <div class="stack side-stack">
      <section class="panel model-panel window-panel" data-window-id="model" aria-labelledby="modelTitle">
        <div class="panel-header">
          <div>
            <h2 id="modelTitle">Backend Model</h2>
            <p id="modelMeta">gpt2-small defaults</p>
          </div>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <div class="field-row">
              <label for="modelName">TransformerLens model</label>
              <input id="modelName" value="gpt2-small">
            </div>
            <div class="field-row">
              <label for="saeRelease">SAE Lens release</label>
              <input id="saeRelease" value="gpt2-small-res-jb">
            </div>
          </div>
          <div class="field-row">
            <label for="saeTemplate">Layer SAE id template</label>
            <input id="saeTemplate" value="blocks.{layer}.hook_resid_pre">
          </div>
          <div class="actions">
            <button id="loadModel" type="button">Load model</button>
            <button class="secondary" id="useSelectedModel" type="button">Use selected model</button>
            <button class="secondary" id="refreshHealth" type="button">Refresh health</button>
          </div>
          <div class="model-picker">
            <div class="field-row">
              <label for="modelFilter">Available TransformerLens models</label>
              <input id="modelFilter" placeholder="Filter by model name, alias, or size">
            </div>
            <div class="table-wrap model-table" aria-label="Available TransformerLens models">
              <table>
                <thead><tr><th>Model</th><th>Est. download</th><th>Params</th></tr></thead>
                <tbody id="modelOptions"></tbody>
              </table>
            </div>
            <div class="status" id="modelListStatus" role="status" aria-live="polite"></div>
          </div>
          <div class="status" id="modelStatus" role="status" aria-live="polite"></div>
        </div>
      </section>

      <section class="panel window-panel" data-window-id="steer" aria-labelledby="steerTitle">
        <div class="panel-header">
          <div>
            <h2 id="steerTitle">Active Steer</h2>
            <p id="stateSummary">No active steers</p>
          </div>
          <button class="secondary small" id="reloadState" type="button">Reload</button>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <div class="field-row">
              <label for="featureId">Feature id</label>
              <input id="featureId" type="number" min="0" value="204">
            </div>
            <div class="field-row">
              <label for="strength">Strength</label>
              <input id="strength" type="number" step="0.1" value="10">
            </div>
          </div>
          <div class="form-grid">
            <div class="field-row">
              <label for="layers">Layers</label>
              <input id="layers" value="6">
            </div>
            <div class="field-row">
              <label for="sourceId">SAE/source id</label>
              <input id="sourceId" value="6-res-jb">
            </div>
          </div>
          <div class="field-row">
            <label for="label">Label</label>
            <input id="label">
          </div>
          <div class="actions">
            <button id="setSteer" type="button">Set only</button>
            <button class="secondary" id="appendSteer" type="button">Append</button>
            <button class="danger" id="clearSteers" type="button">Clear</button>
          </div>
          <div id="steers" class="steer-list"></div>
          <details>
            <summary>State JSON</summary>
            <pre id="state"></pre>
          </details>
          <div class="status" id="stateStatus" role="status" aria-live="polite"></div>
        </div>
      </section>

      <section class="panel window-panel" data-window-id="sources" aria-labelledby="sourcesTitle">
        <div class="panel-header">
          <div>
            <h2 id="sourcesTitle">Neuronpedia Sources</h2>
            <p id="sourceMeta">Feature cache ready</p>
          </div>
        </div>
        <div class="panel-body">
          <div class="form-grid">
            <div class="field-row">
              <label for="cacheModel">Model</label>
              <input id="cacheModel" value="gpt2-small">
            </div>
            <div class="field-row">
              <label for="sourceFilter">Source filter</label>
              <input id="sourceFilter" value="res-jb">
            </div>
          </div>
          <div class="actions">
            <button id="listModels" type="button">List models</button>
            <button id="listSources" type="button">List sources</button>
            <button class="secondary" id="showCached" type="button">Show cached</button>
            <button class="secondary" id="cacheLayers" type="button">Cache Layers</button>
            <button id="downloadSource" type="button">Download selected source</button>
          </div>
          <div class="table-wrap" aria-label="Neuronpedia models and sources">
            <table>
              <thead><tr><th>Model/source</th><th>Cached</th></tr></thead>
              <tbody id="sources"></tbody>
            </table>
          </div>
          <div class="status" id="sourceStatus" role="status" aria-live="polite"></div>
        </div>
      </section>

      <section class="panel window-panel" data-window-id="search" aria-labelledby="searchTitle">
        <div class="panel-header">
          <div>
            <h2 id="searchTitle">Search Cached Labels</h2>
            <p id="searchMeta">Search cached labels across the selected model</p>
          </div>
        </div>
        <div class="panel-body">
          <div class="field-row">
            <label for="search">Search</label>
            <input id="search" value="time phrases">
          </div>
          <label class="check-row" for="searchSelectedOnly">
            <input id="searchSelectedOnly" type="checkbox">
            Search selected source only
          </label>
          <div class="actions">
            <button id="searchCache" type="button">Search</button>
            <button class="secondary" id="applyLabel" type="button" disabled>Apply selected label</button>
          </div>
          <div class="table-wrap" aria-label="Cached label search results">
            <table>
              <thead><tr><th>Feature</th><th>Source</th><th>Description</th></tr></thead>
              <tbody id="labels"></tbody>
            </table>
          </div>
          <div class="status" id="searchStatus" role="status" aria-live="polite"></div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const residualSourcePattern = /^(\d+)-res-jb$/;
    const researchStorageKey = 'steering.researchRuns.v1';
    const uiConfigStorageKey = 'steering.uiConfig.v1';
    const windowLayoutStorageKey = 'steering.windowLayout.v1';
    const windowLayoutVersion = 1;
    const windowResizeDirections = ['nw', 'ne', 'sw', 'se'];
    const minWindowWidth = 280;
    const minWindowHeight = 180;
    const defaultResearchConfig = {
      prompt: 'Today the weather report says',
      maxTokens: '40',
      temperature: '0',
      generationMode: 'auto',
      systemPrompt: 'You are a helpful research assistant.',
      inspectLayers: '6',
      inspectTopK: '5',
      inspectPrompt: false,
      modelName: 'gpt2-small',
      saeRelease: 'gpt2-small-res-jb',
      saeTemplate: 'blocks.{layer}.hook_resid_pre',
      runNote: '',
      featureId: '204',
      strength: '10',
      layers: '6',
      sourceId: '6-res-jb',
      label: '',
      cacheModel: 'gpt2-small',
      sourceFilter: 'res-jb',
      modelFilter: '',
      search: 'time phrases',
      searchSelectedOnly: false
    };
    const fieldHelp = {
      prompt: 'The text sent to the model. In completion mode this is a prefix; in chat mode it becomes the user message in the chat template.',
      maxTokens: 'Maximum generated tokens. Lower values make quick probes cheaper; higher values reveal longer behavioral effects.',
      temperature: 'Sampling randomness. Use 0 for deterministic comparisons, then raise it when studying robustness or diversity.',
      generationMode: 'Auto uses chat formatting for chat/instruct models and raw continuation for base models. Force Chat or Completion when you need a controlled comparison.',
      systemPrompt: 'Optional system message used only for chat-formatted requests. It is included in token inspection so features line up with the generated answer context.',
      inspectLayers: 'Residual-stream SAE layers to use for token hover analysis. Comma-separated layers inspect multiple SAE sources at once.',
      inspectTopK: 'How many highest-activation SAE features to show per token. Larger values surface more candidates but add visual noise.',
      inspectPrompt: 'Includes prompt and chat-template tokens in the hoverable token view, useful when comparing which context tokens activated a feature.',
      modelName: 'TransformerLens model to load locally. This controls both generation and the activation space being inspected.',
      saeRelease: 'SAE Lens release or checkpoint collection. It must match the model and source naming convention used by the loaded SAEs.',
      saeTemplate: 'Template used to turn a layer number into an SAE id, for example blocks.{layer}.hook_resid_pre for residual pre-activation SAEs.',
      modelFilter: 'Filters the local TransformerLens model list by name, alias, estimated download size, or chat/completion tag.',
      runNote: 'Free-form research annotation saved with each run, useful for hypotheses, expected direction, feature source, or experiment condition.',
      featureId: 'Sparse feature index inside the selected SAE. Hover inspection and cached label search can populate this directly.',
      strength: 'Multiplier applied to the SAE decoder vector. Positive and negative values push generation in opposite feature directions.',
      layers: 'Layer targets for the active steer. For residual JB sources this normally mirrors the number in the source id.',
      sourceId: 'Explicit SAE/source id for steering or labels, such as 6-res-jb or blocks.6.hook_resid_pre.',
      label: 'Human-readable note for the steer. Cached Neuronpedia descriptions or hovered token features can fill this in.',
      cacheModel: 'Neuronpedia model id used for listing sources and cached labels. For many gpt2-small workflows this matches the loaded model.',
      sourceFilter: 'Substring filter for Neuronpedia sources. Use res-jb for residual JumpReLU sources, or search hook names for other SAE families.',
      search: 'Searches locally cached feature descriptions. Cache a source first, then search by concept, behavior, or phrase.',
      searchSelectedOnly: 'Restricts cached-label search to the currently selected source so comparisons stay within one SAE dictionary.'
    };
    let selectedModel = null;
    let selectedSource = null;
    let selectedLabel = null;
    let selectedModelOption = null;
    let modelOptions = [];
    let currentState = {items: []};
    let currentHealth = {};
    let researchRuns = [];
    let selectedResearchRun = null;
    let cacheStatus = new Map();
    let visibleSources = [];
    let visibleSourceMode = 'sources';
    let generationController = null;
    let lastInspection = null;
    let tokenPopoverTimer = null;
    let windowLayoutActive = false;
    let windowMoveMode = false;
    let windowZCounter = 10;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: {'Content-Type': 'application/json', ...(options.headers || {})}
      });
      const text = await response.text();
      let data = {};
      if (text) {
        try { data = JSON.parse(text); } catch { data = {detail: text}; }
      }
      if (!response.ok) throw new Error(data.detail || response.statusText);
      return data;
    }

    function setStatus(id, message, cls = '') {
      const node = $(id);
      node.textContent = message || '';
      node.className = `status ${cls}`.trim();
    }

    function setBusy(button, busy) {
      if (button) button.disabled = busy;
    }

    function setBusyMany(ids, busy) {
      for (const id of ids) setBusy($(id), busy);
    }

    function fieldIdsForConfig() {
      return Object.keys(defaultResearchConfig).filter(id => $(id));
    }

    function applyUiConfig(config) {
      for (const id of fieldIdsForConfig()) {
        const node = $(id);
        const value = config[id];
        if (value === undefined) continue;
        if (node.type === 'checkbox') {
          node.checked = Boolean(value);
        } else {
          node.value = String(value);
        }
      }
      selectedModel = $('modelName').value.trim() || null;
      selectedSource = $('sourceId').value.trim() || null;
      highlightModelSelection();
      highlightSourceSelection();
      updateOutputMeta();
    }

    function labelTitle(label) {
      return label.textContent.replace(/\s+/g, ' ').trim();
    }

    function readUiConfig() {
      const config = {};
      for (const id of fieldIdsForConfig()) {
        const node = $(id);
        config[id] = node.type === 'checkbox' ? node.checked : node.value;
      }
      return config;
    }

    function saveUiConfig() {
      localStorage.setItem(uiConfigStorageKey, JSON.stringify(readUiConfig()));
    }

    function loadUiConfig() {
      let config = null;
      try {
        config = JSON.parse(localStorage.getItem(uiConfigStorageKey) || 'null');
      } catch {
        config = null;
      }
      applyUiConfig({...defaultResearchConfig, ...(config || {})});
      if (!config) saveUiConfig();
    }

    function resetUiConfig() {
      applyUiConfig(defaultResearchConfig);
      saveUiConfig();
      resetTokenInspection();
      setStatus('generateStatus', 'Research defaults restored.', 'ok');
      setStatus('modelStatus', 'Defaults prepared for gpt2-small residual SAE workflows.', 'ok');
    }

    function workspaceNode() {
      return $('workspace');
    }

    function windowPanels() {
      return Array.from(document.querySelectorAll('.window-panel[data-window-id]'));
    }

    function clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    }

    function numericStyle(node, property, fallback = 0) {
      const value = Number.parseFloat(node.style[property]);
      return Number.isFinite(value) ? value : fallback;
    }

    function readSavedWindowLayout() {
      try {
        const layout = JSON.parse(localStorage.getItem(windowLayoutStorageKey) || 'null');
        if (!layout || layout.version !== windowLayoutVersion || typeof layout.windows !== 'object') return null;
        return layout;
      } catch {
        return null;
      }
    }

    function windowPanelMetrics(panel) {
      const workspace = workspaceNode();
      const workspaceRect = workspace.getBoundingClientRect();
      const rect = panel.getBoundingClientRect();
      return {
        left: numericStyle(panel, 'left', rect.left - workspaceRect.left + workspace.scrollLeft),
        top: numericStyle(panel, 'top', rect.top - workspaceRect.top + workspace.scrollTop),
        width: numericStyle(panel, 'width', rect.width),
        height: numericStyle(panel, 'height', rect.height),
        z: Number.parseInt(panel.style.zIndex || '0', 10) || 1
      };
    }

    function captureCurrentWindowLayout() {
      const windows = {};
      for (const [index, panel] of windowPanels().entries()) {
        const metrics = windowPanelMetrics(panel);
        windows[panel.dataset.windowId] = {
          left: Math.max(0, Math.round(metrics.left)),
          top: Math.max(0, Math.round(metrics.top)),
          width: Math.max(minWindowWidth, Math.round(metrics.width)),
          height: Math.max(minWindowHeight, Math.round(metrics.height)),
          z: Number(metrics.z) || index + 1
        };
      }
      return {
        version: windowLayoutVersion,
        saved_at: new Date().toISOString(),
        viewport: {width: window.innerWidth, height: window.innerHeight},
        windows
      };
    }

    function applyWindowPanelMetrics(panel, metrics) {
      const workspace = workspaceNode();
      const workspaceWidth = Math.max(workspace.clientWidth, minWindowWidth + 36);
      const maxWidth = Math.max(minWindowWidth, workspaceWidth - 36);
      const width = clamp(Number(metrics.width) || minWindowWidth, minWindowWidth, maxWidth);
      const height = Math.max(minWindowHeight, Number(metrics.height) || minWindowHeight);
      const maxLeft = Math.max(0, workspaceWidth - width - 18);
      const left = clamp(Number(metrics.left) || 0, 0, maxLeft);
      const top = Math.max(0, Number(metrics.top) || 0);
      panel.style.left = `${Math.round(left)}px`;
      panel.style.top = `${Math.round(top)}px`;
      panel.style.width = `${Math.round(width)}px`;
      panel.style.height = `${Math.round(height)}px`;
      panel.style.zIndex = String(Number(metrics.z) || 1);
    }

    function updateWindowWorkspaceBounds() {
      if (!windowLayoutActive) return;
      const workspace = workspaceNode();
      const viewportSpace = Math.max(420, window.innerHeight - workspace.getBoundingClientRect().top);
      const bottom = windowPanels().reduce((max, panel) => {
        const metrics = windowPanelMetrics(panel);
        return Math.max(max, metrics.top + metrics.height);
      }, 0);
      workspace.style.minHeight = `${Math.ceil(Math.max(viewportSpace, bottom + 28))}px`;
    }

    function applyWindowLayout(layout, fallbackLayout = null) {
      const workspace = workspaceNode();
      workspace.classList.add('window-layout-canvas');
      document.body.classList.add('window-layout-active');
      windowLayoutActive = true;
      let maxZ = 1;
      for (const [index, panel] of windowPanels().entries()) {
        const id = panel.dataset.windowId;
        const metrics = layout.windows[id] || fallbackLayout?.windows?.[id] || captureCurrentWindowLayout().windows[id];
        applyWindowPanelMetrics(panel, {...metrics, z: metrics.z || index + 1});
        maxZ = Math.max(maxZ, Number(metrics.z) || index + 1);
      }
      windowZCounter = Math.max(windowZCounter, maxZ + 1);
      updateWindowWorkspaceBounds();
    }

    function ensureWindowLayout() {
      if (windowLayoutActive) return;
      const fallbackLayout = captureCurrentWindowLayout();
      applyWindowLayout(readSavedWindowLayout() || fallbackLayout, fallbackLayout);
    }

    function saveWindowLayout() {
      if (!windowLayoutActive) return;
      const layout = captureCurrentWindowLayout();
      localStorage.setItem(windowLayoutStorageKey, JSON.stringify(layout));
    }

    function clearWindowLayoutStyles() {
      const workspace = workspaceNode();
      workspace.classList.remove('window-layout-canvas');
      workspace.style.minHeight = '';
      document.body.classList.remove('window-layout-active', 'window-layout-editing', 'window-layout-dragging');
      for (const panel of windowPanels()) {
        panel.style.left = '';
        panel.style.top = '';
        panel.style.width = '';
        panel.style.height = '';
        panel.style.zIndex = '';
      }
      windowLayoutActive = false;
      windowMoveMode = false;
      updateWindowLayoutControls();
    }

    function restoreSavedWindowLayout() {
      const saved = readSavedWindowLayout();
      if (!saved) {
        updateWindowLayoutControls();
        return;
      }
      applyWindowLayout(saved, captureCurrentWindowLayout());
      setWindowMoveMode(false);
    }

    function updateWindowLayoutControls() {
      const button = $('moveWindows');
      button.setAttribute('aria-pressed', String(windowMoveMode));
      button.title = windowMoveMode ? 'Lock window layout' : 'Move and resize windows';
      button.setAttribute('aria-label', button.title);
    }

    function setWindowMoveMode(enabled) {
      if (enabled) ensureWindowLayout();
      windowMoveMode = enabled;
      document.body.classList.toggle('window-layout-editing', enabled);
      updateWindowLayoutControls();
      if (!enabled && !readSavedWindowLayout()) clearWindowLayoutStyles();
    }

    function resetWindowLayout() {
      localStorage.removeItem(windowLayoutStorageKey);
      clearWindowLayoutStyles();
      setStatus('generateStatus', 'Window layout reset.', 'ok');
    }

    function bringWindowToFront(panel) {
      windowZCounter += 1;
      panel.style.zIndex = String(windowZCounter);
    }

    function isInteractiveWindowTarget(target) {
      return target instanceof Element && Boolean(target.closest('button, input, select, textarea, a, summary, label, .resize-handle'));
    }

    function startWindowDrag(event) {
      if (!windowMoveMode || event.button !== 0 || isInteractiveWindowTarget(event.target)) return;
      const panel = event.currentTarget.closest('.window-panel');
      if (!panel) return;
      event.preventDefault();
      bringWindowToFront(panel);
      const start = windowPanelMetrics(panel);
      const startX = event.clientX;
      const startY = event.clientY;
      document.body.classList.add('window-layout-dragging');

      const move = (moveEvent) => {
        const left = start.left + moveEvent.clientX - startX;
        const top = start.top + moveEvent.clientY - startY;
        applyWindowPanelMetrics(panel, {...start, left, top, z: windowZCounter});
        updateWindowWorkspaceBounds();
      };
      const stop = () => {
        document.removeEventListener('pointermove', move);
        document.body.classList.remove('window-layout-dragging');
        saveWindowLayout();
      };

      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', stop, {once: true});
    }

    function resizedWindowMetrics(direction, start, dx, dy) {
      let {left, top, width, height} = start;
      if (direction.includes('e')) width = start.width + dx;
      if (direction.includes('s')) height = start.height + dy;
      if (direction.includes('w')) {
        width = start.width - dx;
        left = start.left + dx;
      }
      if (direction.includes('n')) {
        height = start.height - dy;
        top = start.top + dy;
      }
      if (width < minWindowWidth) {
        if (direction.includes('w')) left -= minWindowWidth - width;
        width = minWindowWidth;
      }
      if (height < minWindowHeight) {
        if (direction.includes('n')) top -= minWindowHeight - height;
        height = minWindowHeight;
      }
      if (left < 0) {
        if (direction.includes('w')) width += left;
        left = 0;
      }
      if (top < 0) {
        if (direction.includes('n')) height += top;
        top = 0;
      }
      return {left, top, width, height, z: start.z};
    }

    function startWindowResize(event) {
      if (!windowMoveMode || event.button !== 0) return;
      const panel = event.currentTarget.closest('.window-panel');
      if (!panel) return;
      event.preventDefault();
      event.stopPropagation();
      bringWindowToFront(panel);
      const direction = event.currentTarget.dataset.resize;
      const start = {...windowPanelMetrics(panel), z: windowZCounter};
      const startX = event.clientX;
      const startY = event.clientY;

      const move = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        const dy = moveEvent.clientY - startY;
        applyWindowPanelMetrics(panel, resizedWindowMetrics(direction, start, dx, dy));
        updateWindowWorkspaceBounds();
      };
      const stop = () => {
        document.removeEventListener('pointermove', move);
        saveWindowLayout();
      };

      document.addEventListener('pointermove', move);
      document.addEventListener('pointerup', stop, {once: true});
    }

    function clampWindowLayoutToViewport() {
      if (!windowLayoutActive) return;
      for (const panel of windowPanels()) {
        applyWindowPanelMetrics(panel, windowPanelMetrics(panel));
      }
      updateWindowWorkspaceBounds();
    }

    function setupWindowChrome() {
      for (const panel of windowPanels()) {
        const header = panel.querySelector('.panel-header');
        if (header && !header.dataset.windowDragReady) {
          header.addEventListener('pointerdown', startWindowDrag);
          header.dataset.windowDragReady = 'true';
        }
        for (const direction of windowResizeDirections) {
          if (panel.querySelector(`.resize-handle.${direction}`)) continue;
          const handle = document.createElement('button');
          handle.type = 'button';
          handle.className = `resize-handle ${direction}`;
          handle.dataset.resize = direction;
          handle.setAttribute('aria-label', `Resize ${panel.dataset.windowId} window`);
          handle.addEventListener('pointerdown', startWindowResize);
          panel.appendChild(handle);
        }
      }
    }

    function setupFieldHelp() {
      for (const [id, text] of Object.entries(fieldHelp)) {
        const label = document.querySelector(`label[for="${id}"]`);
        if (!label || label.dataset.helpReady) continue;
        const row = label.parentElement;
        const button = document.createElement('button');
        const tip = document.createElement('span');
        button.className = 'help-dot';
        button.type = 'button';
        button.textContent = '?';
        button.setAttribute('aria-label', `${labelTitle(label)}: ${text}`);
        button.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
        });
        tip.className = 'help-text';
        tip.setAttribute('role', 'tooltip');
        tip.textContent = text;
        if (label.classList.contains('check-row') && row) {
          const wrap = document.createElement('div');
          wrap.className = 'check-help-row';
          row.insertBefore(wrap, label);
          wrap.append(label, button, tip);
        } else if (row) {
          const wrap = document.createElement('div');
          wrap.className = 'label-with-help';
          row.insertBefore(wrap, label);
          wrap.append(label, button, tip);
        }
        label.dataset.helpReady = 'true';
      }
    }

    function compact(text, max = 96) {
      const value = String(text || '').replace(/\s+/g, ' ').trim();
      return value.length > max ? `${value.slice(0, max - 3).trimEnd()}...` : value;
    }

    function formatWhen(value) {
      if (!value) return '-';
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
    }

    function modelId() {
      return $('cacheModel').value.trim() || selectedModel || $('modelName').value.trim();
    }

    function looksLikeChatModelName(name) {
      const value = String(name || '').toLowerCase();
      return ['chat', 'instruct', '-it', '_it', '/it', '-sft', '_sft', '/sft', 'rlhf', 'dpo']
        .some(marker => value.includes(marker));
    }

    function autoModeLooksChatty() {
      return Boolean(
        currentHealth.chat_template_available
        || currentHealth.chat_model_guess
        || selectedModelOption?.is_chat_model
        || looksLikeChatModelName($('modelName').value)
      );
    }

    function responseModeLabel(mode) {
      if (mode === 'auto') return autoModeLooksChatty() ? 'auto chat' : 'auto completion';
      return mode;
    }

    function cacheModelFromLoadName(loadName) {
      if (!loadName) return '';
      const parts = loadName.split('/');
      return parts[parts.length - 1] || loadName;
    }

    function sourceKey(model, source) {
      return `${model}/${source}`;
    }

    function parseNumber(id, label) {
      const raw = $(id).value.trim();
      const value = Number(raw);
      if (raw === '' || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
      return value;
    }

    function parseLayers(raw) {
      const values = raw.split(/[,\s]+/).filter(Boolean).map((part) => {
        const value = Number(part);
        if (!Number.isInteger(value) || value < 0) throw new Error('Layers must be non-negative whole numbers.');
        return value;
      });
      return [...new Set(values)];
    }

    function compatibleResidualSources(sources) {
      return [...new Set(sources.filter((source) => residualSourcePattern.test(source)))]
        .sort((a, b) => Number(a.match(residualSourcePattern)[1]) - Number(b.match(residualSourcePattern)[1]));
    }

    function syncSourceToLayer(source) {
      const match = source.match(residualSourcePattern);
      if (match) {
        $('layers').value = match[1];
        $('inspectLayers').value = match[1];
      }
    }

    function selectSource(value, mode) {
      if (mode === 'models') {
        selectedModel = value;
        selectedSource = null;
        $('cacheModel').value = value;
        $('modelName').value = value;
        $('saeRelease').value = `${value}-res-jb`;
        setStatus('sourceStatus', `Selected model ${value}.`, 'ok');
      } else {
        let source = value;
        let model = modelId();
        if (mode === 'cached' && value.includes('/')) {
          [model, source] = value.split('/', 2);
          $('cacheModel').value = model;
          $('modelName').value = model;
        }
        selectedSource = source;
        $('sourceId').value = source;
        syncSourceToLayer(source);
        setStatus('sourceStatus', `Selected ${model}/${source}.`, 'ok');
      }
      highlightSourceSelection();
      saveUiConfig();
    }

    function highlightSourceSelection() {
      const rows = $('sources').querySelectorAll('tr[data-value]');
      rows.forEach((row) => {
        const mode = row.dataset.mode;
        const value = row.dataset.value;
        const selected = mode === 'models' ? value === selectedModel : value === selectedSource || value.endsWith(`/${selectedSource}`);
        row.classList.toggle('selected', Boolean(selected));
      });
    }

    function renderHealth(data) {
      currentHealth = data || {};
      const model = data.model_name || '-';
      const device = data.device || '-';
      const sae = data.sae_release || '-';
      const busy = data.busy ? 'Busy' : 'Ready';
      $('health').textContent = `${model} on ${device} | ${sae}`;
      $('healthModel').textContent = model;
      $('healthDevice').textContent = device;
      $('healthSae').textContent = sae;
      $('healthBusy').textContent = busy;
      $('modelName').value = data.model_name || $('modelName').value;
      $('saeRelease').value = data.sae_release || $('saeRelease').value;
      $('saeTemplate').value = data.sae_id_template || $('saeTemplate').value;
      $('modelMeta').textContent = data.sae_id_template || 'Layer SAE id template';
      highlightModelSelection();
    }

    async function refreshHealth() {
      try {
        renderHealth(await api('/health'));
        setStatus('modelStatus', '');
      } catch (err) {
        $('health').textContent = `Backend unavailable: ${err.message}`;
        $('healthModel').textContent = '-';
        $('healthDevice').textContent = '-';
        $('healthSae').textContent = '-';
        $('healthBusy').textContent = 'Offline';
      }
    }

    async function listModelOptions() {
      setStatus('modelListStatus', 'Loading available models...', '');
      try {
        const data = await api('/api/model/options');
        modelOptions = data.models || [];
        renderModelOptions();
        setStatus('modelListStatus', `${data.count} TransformerLens models listed with estimated download sizes.`, 'ok');
      } catch (err) {
        setStatus('modelListStatus', err.message, 'error');
      }
    }

    function modelOptionSearchText(option) {
      return [
        option.model_name,
        option.load_name,
        ...(option.aliases || []),
        option.download_size_label,
        option.parameter_count_label,
        option.is_chat_model ? 'chat instruct' : 'completion',
      ].join(' ').toLowerCase();
    }

    function renderModelOptions() {
      const needle = $('modelFilter').value.trim().toLowerCase();
      const rows = needle ? modelOptions.filter(option => modelOptionSearchText(option).includes(needle)) : modelOptions;
      const body = $('modelOptions');
      body.innerHTML = '';
      if (!rows.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="3" class="muted">No matching models</td>';
        body.appendChild(tr);
        return;
      }
      for (const option of rows) {
        const tr = document.createElement('tr');
        tr.className = 'selectable';
        tr.dataset.loadName = option.load_name;

        const modelCell = document.createElement('td');
        const nameWrap = document.createElement('div');
        const name = document.createElement('strong');
        const aliases = document.createElement('span');
        modelCell.className = 'wrap';
        nameWrap.className = 'model-name';
        aliases.className = 'alias-list';
        name.textContent = option.load_name;
        const aliasText = [option.model_name, ...(option.aliases || [])]
          .filter(value => value && value !== option.load_name)
          .slice(0, 4)
          .join(' / ');
        const modeTag = option.is_chat_model ? 'chat/instruct' : 'completion';
        aliases.textContent = `${aliasText || 'official name'} · ${modeTag}`;
        nameWrap.append(name, aliases);
        modelCell.appendChild(nameWrap);

        const sizeCell = document.createElement('td');
        const paramCell = document.createElement('td');
        sizeCell.className = 'compact';
        paramCell.className = 'compact muted';
        sizeCell.textContent = option.download_size_label || 'Unknown';
        paramCell.textContent = option.parameter_count_label || 'Unknown';
        tr.append(modelCell, sizeCell, paramCell);
        tr.addEventListener('click', () => selectModelOption(option));
        body.appendChild(tr);
      }
      highlightModelSelection();
    }

    function selectModelOption(option) {
      selectedModelOption = option;
      selectedModel = option.load_name;
      $('modelName').value = option.load_name;
      $('cacheModel').value = cacheModelFromLoadName(option.load_name);
      if (option.sae_release_guess) $('saeRelease').value = option.sae_release_guess;
      highlightModelSelection();
      const modeHint = option.is_chat_model ? ' Auto response mode will use chat formatting.' : '';
      setStatus('modelListStatus', `Selected ${option.load_name}; estimated download ${option.download_size_label}.${modeHint}`, 'ok');
      saveUiConfig();
    }

    function highlightModelSelection() {
      const active = selectedModelOption ? selectedModelOption.load_name : $('modelName').value.trim();
      $('modelOptions').querySelectorAll('tr[data-load-name]').forEach((row) => {
        row.classList.toggle('selected', row.dataset.loadName === active);
      });
    }

    function loadResearchRuns() {
      try {
        const saved = JSON.parse(localStorage.getItem(researchStorageKey) || '[]');
        researchRuns = Array.isArray(saved) ? saved : [];
      } catch {
        researchRuns = [];
      }
      renderResearchRuns();
    }

    function saveResearchRuns() {
      localStorage.setItem(researchStorageKey, JSON.stringify(researchRuns.slice(0, 200)));
    }

    function runModeClass(mode) {
      return mode === 'baseline' ? 'baseline' : 'steered';
    }

    function renderResearchRuns() {
      const body = $('researchRuns');
      body.innerHTML = '';
      $('researchMeta').textContent = researchRuns.length
        ? `${researchRuns.length} saved ${researchRuns.length === 1 ? 'run' : 'runs'}`
        : 'No runs recorded';
      if (!researchRuns.length) {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.colSpan = 4;
        td.className = 'muted';
        td.textContent = 'No research runs yet';
        tr.appendChild(td);
        body.appendChild(tr);
        return;
      }
      for (const run of researchRuns) {
        const tr = document.createElement('tr');
        tr.className = 'selectable';
        tr.dataset.runId = run.id;

        const timeCell = document.createElement('td');
        const modeCell = document.createElement('td');
        const modelCell = document.createElement('td');
        const outputCell = document.createElement('td');
        const pill = document.createElement('span');
        const output = document.createElement('div');
        const note = document.createElement('div');

        timeCell.className = 'compact';
        modeCell.className = 'compact';
        modelCell.className = 'compact';
        outputCell.className = 'wrap';
        pill.className = `mode-pill ${runModeClass(run.mode)}`;
        pill.textContent = run.mode || 'run';
        modelCell.textContent = run.model || '-';
        timeCell.textContent = new Date(run.created_at).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
        output.textContent = compact(run.output, 170);
        if (run.note) {
          note.className = 'run-note';
          note.textContent = run.note;
        }

        modeCell.appendChild(pill);
        outputCell.append(output, note);
        tr.append(timeCell, modeCell, modelCell, outputCell);
        tr.addEventListener('click', () => selectResearchRun(run));
        body.appendChild(tr);
      }
      highlightResearchRun();
    }

    function highlightResearchRun() {
      $('researchRuns').querySelectorAll('tr[data-run-id]').forEach((row) => {
        row.classList.toggle('selected', selectedResearchRun && row.dataset.runId === selectedResearchRun.id);
      });
    }

    function selectResearchRun(run) {
      selectedResearchRun = run;
      resetTokenInspection();
      $('prompt').value = run.prompt || $('prompt').value;
      $('generationMode').value = run.generation_mode || $('generationMode').value;
      $('systemPrompt').value = run.system_prompt || $('systemPrompt').value;
      $('output').textContent = run.output || '';
      $('generateMeta').textContent = `${run.mode || 'run'} | ${run.response_mode || run.generation_mode || '-'} | ${run.model || '-'}`;
      updateOutputMeta();
      highlightResearchRun();
      setStatus('researchStatus', `Loaded ${run.mode || 'run'} from ${formatWhen(run.created_at)}.`, 'ok');
    }

    function addResearchRun({mode, prompt, output, settings, steersEnabled}) {
      const run = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        created_at: new Date().toISOString(),
        mode,
        prompt,
        output,
        model: $('modelName').value.trim(),
        sae_release: $('saeRelease').value.trim(),
        sae_id_template: $('saeTemplate').value.trim(),
        max_new_tokens: settings.max_new_tokens,
        temperature: settings.temperature,
        generation_mode: settings.generation_mode,
        response_mode: settings.response_mode,
        system_prompt: settings.system_prompt,
        steers_enabled: steersEnabled,
        state: currentState,
        note: $('runNote').value.trim(),
      };
      researchRuns = [run, ...researchRuns].slice(0, 200);
      selectedResearchRun = run;
      saveResearchRuns();
      renderResearchRuns();
      return run;
    }

    async function copyLatestRun() {
      const run = selectedResearchRun || researchRuns[0];
      if (!run) return setStatus('researchStatus', 'No research run to copy.', 'warn');
      try {
        await navigator.clipboard.writeText(JSON.stringify(run, null, 2));
        setStatus('researchStatus', 'Research run JSON copied.', 'ok');
      } catch (err) {
        setStatus('researchStatus', err.message, 'error');
      }
    }

    function exportRuns() {
      if (!researchRuns.length) return setStatus('researchStatus', 'No research runs to export.', 'warn');
      const blob = new Blob([JSON.stringify(researchRuns, null, 2)], {type: 'application/json'});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = `steering-runs-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(link.href);
      setStatus('researchStatus', `Exported ${researchRuns.length} research runs.`, 'ok');
    }

    function clearResearchRuns() {
      if (!researchRuns.length) return;
      if (!window.confirm('Clear saved research runs from this browser?')) return;
      researchRuns = [];
      selectedResearchRun = null;
      saveResearchRuns();
      renderResearchRuns();
      setStatus('researchStatus', 'Research log cleared.', 'ok');
    }

    function renderState(state) {
      currentState = state;
      const items = Array.isArray(state.items) ? state.items : [];
      $('state').textContent = JSON.stringify(state, null, 2);
      $('stateSummary').textContent = items.length
        ? `${items.length} active ${items.length === 1 ? 'steer' : 'steers'} | updated ${formatWhen(state.updated_at)}`
        : 'No active steers';
      const list = $('steers');
      list.innerHTML = '';
      if (!items.length) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'No active steers';
        list.appendChild(empty);
        return;
      }
      for (const item of items) {
        const row = document.createElement('div');
        row.className = 'steer-row';
        const target = item.sae_id || (item.layers || []).join(', ') || '-';
        const label = item.label ? compact(item.label, 140) : '-';
        const rows = [
          [`Feature ${item.feature_id}`, label],
          [`Strength ${item.strength}`, item.model_id || modelId() || '-'],
          ['Target', target],
        ];
        for (const [name, value] of rows) {
          const strong = document.createElement('strong');
          const span = document.createElement('span');
          strong.textContent = name;
          span.textContent = value;
          row.append(strong, span);
        }
        list.appendChild(row);
      }
    }

    async function refreshState() {
      const state = await api('/api/state');
      renderState(state);
      return state;
    }

    async function refreshCacheStatus() {
      const data = await api('/api/cache/status');
      cacheStatus = new Map(data.sources.map(row => [sourceKey(row.model_id, row.source_id), row]));
      $('sourceMeta').textContent = data.count ? `${data.count} cached sources` : 'Feature cache ready';
      return data;
    }

    function cacheLabelFor(model, source) {
      const cached = cacheStatus.get(sourceKey(model, source));
      if (!cached) return '-';
      return `${cached.label_count} labels / ${cached.feature_count} features`;
    }

    function renderSources(rows, mode, sourceModel = modelId()) {
      visibleSources = rows;
      visibleSourceMode = mode;
      const body = $('sources');
      body.innerHTML = '';
      if (!rows.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="2" class="muted">No rows</td>';
        body.appendChild(tr);
        return;
      }
      for (const value of rows) {
        const tr = document.createElement('tr');
        tr.className = 'selectable';
        tr.dataset.value = value;
        tr.dataset.mode = mode;
        const nameCell = document.createElement('td');
        const cacheCell = document.createElement('td');
        nameCell.className = 'wrap';
        nameCell.textContent = value;
        let marker = '-';
        if (mode === 'sources') marker = cacheLabelFor(sourceModel, value);
        if (mode === 'cached') {
          const [cachedModel, cachedSource] = value.split('/', 2);
          marker = cacheLabelFor(cachedModel, cachedSource);
        }
        cacheCell.textContent = marker;
        if (marker === '-') cacheCell.className = 'muted';
        tr.append(nameCell, cacheCell);
        tr.addEventListener('click', () => selectSource(value, mode));
        body.appendChild(tr);
      }
      highlightSourceSelection();
    }

    async function listModels() {
      setBusy($('listModels'), true);
      try {
        const data = await api('/api/neuronpedia/models');
        renderSources(data.models, 'models');
        setStatus('sourceStatus', `Found ${data.count} Neuronpedia export models.`, data.count ? 'ok' : '');
      } catch (err) {
        setStatus('sourceStatus', err.message, 'error');
      } finally {
        setBusy($('listModels'), false);
      }
    }

    async function listSources() {
      setBusy($('listSources'), true);
      try {
        await refreshCacheStatus();
        const selected = modelId();
        const params = new URLSearchParams({model_id: selected});
        const filter = $('sourceFilter').value.trim();
        if (filter) params.set('contains', filter);
        const data = await api(`/api/neuronpedia/sources?${params}`);
        renderSources(data.sources, 'sources', data.model_id);
        setStatus('sourceStatus', `Found ${data.count} sources for ${data.model_id}.`, data.count ? 'ok' : '');
      } catch (err) {
        setStatus('sourceStatus', err.message, 'error');
      } finally {
        setBusy($('listSources'), false);
      }
    }

    async function showCached() {
      setBusy($('showCached'), true);
      try {
        const data = await refreshCacheStatus();
        const rows = data.sources.map(row => `${row.model_id}/${row.source_id}`);
        renderSources(rows, 'cached');
        setStatus('sourceStatus', data.count ? `Showing ${data.count} cached sources.` : 'No cached sources yet.', data.count ? 'ok' : '');
      } catch (err) {
        setStatus('sourceStatus', err.message, 'error');
      } finally {
        setBusy($('showCached'), false);
      }
    }

    async function downloadSource() {
      const source = selectedSource || $('sourceId').value.trim();
      if (!source) return setStatus('sourceStatus', 'Select or enter a source first.', 'error');
      setBusy($('downloadSource'), true);
      try {
        const row = await api('/api/cache/source', {
          method: 'POST',
          body: JSON.stringify({model_id: modelId(), source_id: source})
        });
        await refreshCacheStatus();
        if (visibleSourceMode) renderSources(visibleSources, visibleSourceMode, modelId());
        setStatus('sourceStatus', `Cached ${row.label_count} labels for ${row.model_id}/${row.source_id}.`, 'ok');
      } catch (err) {
        setStatus('sourceStatus', err.message, 'error');
      } finally {
        setBusy($('downloadSource'), false);
      }
    }

    async function cacheLayers() {
      const ids = ['cacheLayers', 'downloadSource', 'listSources', 'listModels', 'showCached'];
      setBusyMany(ids, true);
      try {
        await refreshCacheStatus();
        const selected = modelId();
        const data = await api(`/api/neuronpedia/sources?${new URLSearchParams({model_id: selected, contains: 'res-jb'})}`);
        const sources = compatibleResidualSources(data.sources);
        if (!sources.length) {
          setStatus('sourceStatus', `No residual JB sources found for ${selected}.`, 'error');
          return;
        }
        let downloaded = 0;
        for (let i = 0; i < sources.length; i += 1) {
          const source = sources[i];
          const cached = cacheStatus.get(sourceKey(selected, source));
          if (cached && cached.label_count > 0) continue;
          setStatus('sourceStatus', `Caching ${selected}/${source} (${i + 1}/${sources.length})...`, 'ok');
          await api('/api/cache/source', {
            method: 'POST',
            body: JSON.stringify({model_id: selected, source_id: source})
          });
          downloaded += 1;
          await refreshCacheStatus();
        }
        renderSources(sources, 'sources', selected);
        const total = sources.reduce((sum, source) => {
          const cached = cacheStatus.get(sourceKey(selected, source));
          return sum + (cached ? cached.label_count : 0);
        }, 0);
        setStatus('sourceStatus', `Ready: ${sources.length} residual JB layers, ${total} cached labels, ${downloaded} downloaded.`, 'ok');
      } catch (err) {
        setStatus('sourceStatus', err.message, 'error');
      } finally {
        setBusyMany(ids, false);
      }
    }

    function selectLabel(label, row) {
      selectedLabel = label;
      $('labels').querySelectorAll('tr').forEach((node) => node.classList.remove('selected'));
      if (row) row.classList.add('selected');
      $('featureId').value = label.feature_id;
      $('sourceId').value = label.source_id;
      $('cacheModel').value = label.model_id;
      $('modelName').value = label.model_id;
      $('label').value = label.description;
      syncSourceToLayer(label.source_id);
      $('applyLabel').disabled = false;
      setStatus('searchStatus', `Selected ${label.model_id}/${label.source_id}/${label.feature_id}.`, 'ok');
    }

    function renderLabels(labels) {
      const body = $('labels');
      body.innerHTML = '';
      selectedLabel = null;
      $('applyLabel').disabled = true;
      if (!labels.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="3" class="muted">No matching cached labels</td>';
        body.appendChild(tr);
        return;
      }
      for (const label of labels) {
        const tr = document.createElement('tr');
        tr.className = 'selectable';
        const featureCell = document.createElement('td');
        const sourceCell = document.createElement('td');
        const descriptionCell = document.createElement('td');
        featureCell.className = 'compact';
        sourceCell.className = 'compact';
        descriptionCell.className = 'wrap';
        featureCell.textContent = label.feature_id;
        sourceCell.textContent = `${label.model_id}/${label.source_id}`;
        descriptionCell.textContent = compact(label.description, 180);
        tr.append(featureCell, sourceCell, descriptionCell);
        tr.addEventListener('click', () => selectLabel(label, tr));
        body.appendChild(tr);
      }
    }

    async function searchCache() {
      const query = $('search').value.trim();
      if (!query) return setStatus('searchStatus', 'Search query is required.', 'error');
      setBusy($('searchCache'), true);
      try {
        const params = new URLSearchParams({query, model_id: modelId(), limit: '50'});
        const source = $('sourceId').value.trim();
        if ($('searchSelectedOnly').checked && source) params.set('source_id', source);
        const data = await api(`/api/cache/search?${params}`);
        renderLabels(data.labels);
        setStatus('searchStatus', `Found ${data.count} cached labels.`, data.count ? 'ok' : '');
      } catch (err) {
        setStatus('searchStatus', err.message, 'error');
      } finally {
        setBusy($('searchCache'), false);
      }
    }

    function applySelectedLabel() {
      if (!selectedLabel) return setStatus('searchStatus', 'Select a cached label first.', 'error');
      selectLabel(selectedLabel);
    }

    function steerPayload(append) {
      const source = $('sourceId').value.trim();
      const residual = source.match(residualSourcePattern);
      const layers = residual ? [Number(residual[1])] : parseLayers($('layers').value.trim());
      if (!layers.length && !source) throw new Error('Provide a layer or SAE/source id.');
      return {
        feature_id: parseNumber('featureId', 'Feature id'),
        strength: parseNumber('strength', 'Strength'),
        layers,
        sae_id: residual ? null : source || null,
        model_id: $('modelName').value.trim() || modelId() || null,
        label: $('label').value.trim() || null,
        append
      };
    }

    async function setSteer(append) {
      const button = append ? $('appendSteer') : $('setSteer');
      setBusy(button, true);
      try {
        const state = await api('/api/state/items', {method: 'POST', body: JSON.stringify(steerPayload(append))});
        renderState(state);
        setStatus('stateStatus', append ? 'Steer appended.' : 'Steer set.', 'ok');
      } catch (err) {
        setStatus('stateStatus', err.message, 'error');
      } finally {
        setBusy(button, false);
      }
    }

    async function clearSteers() {
      if (!window.confirm('Clear all active steers?')) return;
      setBusy($('clearSteers'), true);
      try {
        const state = await api('/api/state', {method: 'DELETE'});
        renderState(state);
        setStatus('stateStatus', 'Steers cleared.', 'ok');
      } catch (err) {
        setStatus('stateStatus', err.message, 'error');
      } finally {
        setBusy($('clearSteers'), false);
      }
    }

    function hideTokenPopover() {
      clearTimeout(tokenPopoverTimer);
      $('tokenPopover').classList.add('hide');
    }

    function scheduleTokenPopoverHide() {
      clearTimeout(tokenPopoverTimer);
      tokenPopoverTimer = setTimeout(hideTokenPopover, 140);
    }

    function resetTokenInspection() {
      lastInspection = null;
      hideTokenPopover();
      $('tokenOutput').innerHTML = '';
      $('tokenOutput').classList.add('hide');
      $('output').classList.remove('hide');
      setStatus('inspectStatus', '');
    }

    function inspectPayloadForOutput() {
      const text = $('output').textContent;
      if (!text.trim()) throw new Error('Generate or load output before inspecting.');
      const payload = {
        text,
        prompt: $('prompt').value,
        cache_model_id: $('cacheModel').value.trim() || $('modelName').value.trim() || null,
        cache_source_id: $('sourceId').value.trim() || null,
        top_k: parseNumber('inspectTopK', 'Top features'),
        include_prompt: $('inspectPrompt').checked,
        mode: $('generationMode').value,
        system_prompt: $('systemPrompt').value.trim() || null,
        layers: []
      };
      if (!Number.isInteger(payload.top_k) || payload.top_k < 1 || payload.top_k > 20) {
        throw new Error('Top features must be a whole number from 1 to 20.');
      }
      const source = $('sourceId').value.trim();
      const residual = source.match(residualSourcePattern);
      const layerText = $('inspectLayers').value.trim() || (residual ? residual[1] : $('layers').value.trim());
      if (layerText) {
        payload.layers = parseLayers(layerText);
      } else if (source) {
        payload.sae_id = source;
      }
      if (!payload.layers.length && !payload.sae_id) {
        throw new Error('Provide inspect layers or an SAE/source id.');
      }
      return payload;
    }

    function featureSourceLabel(feature) {
      const layer = Number.isInteger(feature.layer) ? `L${feature.layer}` : 'SAE';
      return `${layer} | ${feature.source_id || feature.sae_id || feature.hook_name}`;
    }

    function cachedLabelSourceLabel(feature) {
      if (feature.description && (feature.label_model_id || feature.source_id)) {
        return `cached label: ${feature.label_model_id || $('cacheModel').value || modelId()}/${feature.source_id}`;
      }
      if (feature.label_lookup?.status === 'not_found') {
        return `no cached label found for ${featureSourceLabel(feature)}`;
      }
      return featureSourceLabel(feature);
    }

    function useInspectedFeature(feature, token) {
      const layer = Number.isInteger(feature.layer) ? String(feature.layer) : '';
      $('featureId').value = feature.feature_id;
      if (layer) {
        $('layers').value = layer;
        $('inspectLayers').value = layer;
        $('sourceId').value = `${layer}-res-jb`;
      } else if (feature.sae_id) {
        $('sourceId').value = feature.sae_id;
      }
      $('label').value = feature.description || `token ${String(token.text || '').trim() || token.token_id}`;
      setStatus('stateStatus', `Loaded feature ${feature.feature_id} from token ${String(token.text || '').trim() || token.token_id}.`, 'ok');
      hideTokenPopover();
    }

    function showTokenPopover(token, anchor) {
      clearTimeout(tokenPopoverTimer);
      const box = $('tokenPopover');
      box.innerHTML = '';

      const title = document.createElement('strong');
      title.textContent = token.text || '(empty token)';
      const meta = document.createElement('div');
      meta.className = 'muted';
      meta.textContent = `token ${token.token_id} | position ${token.position}${token.is_prompt ? ' | prompt' : ''}`;
      const list = document.createElement('div');
      list.className = 'feature-list';

      if (!token.features || !token.features.length) {
        const empty = document.createElement('div');
        empty.className = 'muted';
        empty.textContent = 'No positive SAE features in the top-k window.';
        list.appendChild(empty);
      } else {
        for (const feature of token.features) {
          const row = document.createElement('button');
          row.className = 'feature-hit';
          row.type = 'button';
          row.title = 'Load this feature into the Active Steer form';
          const id = document.createElement('b');
          const activation = document.createElement('span');
          const detail = document.createElement('span');
          const source = document.createElement('span');
          id.textContent = `#${feature.feature_id}`;
          activation.textContent = `activation ${Number(feature.activation).toFixed(3)}`;
          detail.className = 'feature-detail';
          detail.textContent = feature.description || featureSourceLabel(feature);
          source.className = 'feature-source';
          source.textContent = cachedLabelSourceLabel(feature);
          row.append(id, activation, detail, source);
          row.addEventListener('click', () => useInspectedFeature(feature, token));
          list.appendChild(row);
        }
      }

      box.append(title, meta, list);
      box.classList.remove('hide');
      const rect = anchor.getBoundingClientRect();
      const boxRect = box.getBoundingClientRect();
      const left = Math.min(window.innerWidth - boxRect.width - 12, Math.max(12, rect.left));
      let top = rect.bottom + 8;
      if (top + boxRect.height > window.innerHeight - 12) {
        top = Math.max(12, rect.top - boxRect.height - 8);
      }
      box.style.left = `${left}px`;
      box.style.top = `${top}px`;
    }

    function renderInspectedOutput(data) {
      lastInspection = data;
      const output = $('tokenOutput');
      output.innerHTML = '';
      for (const token of data.tokens || []) {
        const span = document.createElement('span');
        span.className = `token-chip${token.is_prompt ? ' prompt-token' : ''}`;
        span.tabIndex = 0;
        span.textContent = token.text;
        span.addEventListener('mouseenter', () => showTokenPopover(token, span));
        span.addEventListener('focus', () => showTokenPopover(token, span));
        span.addEventListener('mouseleave', scheduleTokenPopoverHide);
        span.addEventListener('blur', scheduleTokenPopoverHide);
        output.appendChild(span);
      }
      $('output').classList.add('hide');
      output.classList.remove('hide');
      $('outputMode').textContent = 'feature inspect';
      updateOutputMeta();
    }

    async function inspectOutput() {
      setBusy($('inspectOutput'), true);
      try {
        const payload = inspectPayloadForOutput();
        setStatus('inspectStatus', 'Inspecting output...', 'ok');
        const data = await api('/api/inspect/tokens', {
          method: 'POST',
          body: JSON.stringify(payload)
        });
        renderInspectedOutput(data);
        const sourceCount = (data.sources || []).length;
        const matchedLabels = (data.tokens || []).reduce((count, token) => {
          return count + (token.features || []).filter((feature) => feature.description).length;
        }, 0);
        setStatus(
          'inspectStatus',
          `Inspected ${data.token_count} tokens across ${sourceCount} ${sourceCount === 1 ? 'SAE source' : 'SAE sources'}; matched ${matchedLabels} cached ${matchedLabels === 1 ? 'label' : 'labels'}.`,
          'ok'
        );
      } catch (err) {
        setStatus('inspectStatus', err.message, 'error');
      } finally {
        setBusy($('inspectOutput'), false);
      }
    }

    function readGenerationSettings() {
      const max_new_tokens = parseNumber('maxTokens', 'Max tokens');
      const temperature = parseNumber('temperature', 'Temperature');
      const generation_mode = $('generationMode').value;
      if (!Number.isInteger(max_new_tokens) || max_new_tokens < 1 || max_new_tokens > 512) {
        throw new Error('Max tokens must be a whole number from 1 to 512.');
      }
      if (temperature < 0) throw new Error('Temperature must be >= 0.');
      if (!$('prompt').value.trim()) throw new Error('Prompt is required.');
      return {
        max_new_tokens,
        temperature,
        generation_mode,
        response_mode: responseModeLabel(generation_mode),
        system_prompt: $('systemPrompt').value.trim()
      };
    }

    function updateOutputMeta() {
      const chars = $('output').textContent.length;
      if (lastInspection && !$('tokenOutput').classList.contains('hide')) {
        $('outputTokens').textContent = `${lastInspection.token_count} tokens | ${chars} chars`;
      } else {
        $('outputTokens').textContent = `${chars} chars`;
      }
    }

    async function generate(options = {}) {
      const mode = options.mode || 'steered';
      const steersEnabled = options.steersEnabled !== false;
      if (generationController) generationController.abort();
      generationController = new AbortController();
      resetTokenInspection();
      $('output').textContent = '';
      updateOutputMeta();
      $('outputMode').textContent = `${steersEnabled ? 'steers on' : 'baseline'} | ${responseModeLabel($('generationMode').value)}`;
      setBusyMany(['generate', 'generateBaseline', 'compareRuns'], true);
      $('stopGenerate').disabled = false;
      try {
        await refreshState().catch(() => {});
        const settings = readGenerationSettings();
        const prompt = $('prompt').value;
        setStatus('generateStatus', mode === 'baseline' ? 'Generating baseline...' : 'Generating with current steer state...', 'ok');
        $('generateMeta').textContent = `${mode} | ${settings.response_mode} | ${settings.max_new_tokens} tokens | temperature ${settings.temperature}`;
        const response = await fetch('/generate', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            prompt,
            max_new_tokens: settings.max_new_tokens,
            temperature: settings.temperature,
            stream: true,
            steers_enabled: steersEnabled,
            mode: settings.generation_mode,
            system_prompt: settings.system_prompt || null,
            stop_on_eos: settings.generation_mode === 'auto' ? null : settings.generation_mode === 'chat'
          }),
          signal: generationController.signal
        });
        if (!response.ok) {
          const text = await response.text();
          let detail = text;
          try { detail = JSON.parse(text).detail || text; } catch {}
          throw new Error(detail || response.statusText);
        }
        if (!response.body) {
          $('output').textContent = await response.text();
        } else {
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          while (true) {
            const {value, done} = await reader.read();
            if (done) break;
            $('output').textContent += decoder.decode(value, {stream: true});
            updateOutputMeta();
            $('output').scrollTop = $('output').scrollHeight;
          }
          $('output').textContent += decoder.decode();
        }
        updateOutputMeta();
        const output = $('output').textContent;
        addResearchRun({mode, prompt, output, settings, steersEnabled});
        setStatus('generateStatus', 'Generation complete.', 'ok');
        setStatus('researchStatus', `Recorded ${mode} run.`, 'ok');
        await refreshHealth();
        return output;
      } catch (err) {
        if (err.name === 'AbortError') {
          setStatus('generateStatus', 'Generation stopped.', 'warn');
        } else {
          setStatus('generateStatus', err.message, 'error');
        }
        return null;
      } finally {
        generationController = null;
        setBusyMany(['generate', 'generateBaseline', 'compareRuns'], false);
        $('stopGenerate').disabled = true;
        $('outputMode').textContent = 'streaming off';
      }
    }

    async function generateBaseline() {
      return generate({mode: 'baseline', steersEnabled: false});
    }

    async function compareRuns() {
      setBusy($('compareRuns'), true);
      setStatus('researchStatus', 'Running baseline and steered comparison...', 'ok');
      const baseline = await generate({mode: 'baseline', steersEnabled: false});
      if (baseline === null) return;
      const steered = await generate({mode: 'steered', steersEnabled: true});
      if (steered === null) return;
      resetTokenInspection();
      $('output').textContent = `BASELINE\n${baseline}\n\nSTEERED\n${steered}`;
      updateOutputMeta();
      $('generateMeta').textContent = 'baseline vs steered comparison';
      setStatus('researchStatus', 'Comparison recorded as two research runs.', 'ok');
    }

    function stopGenerate() {
      if (generationController) generationController.abort();
    }

    async function copyOutput() {
      const text = $('output').textContent;
      if (!text) return setStatus('generateStatus', 'Nothing to copy.', 'warn');
      try {
        await navigator.clipboard.writeText(text);
        setStatus('generateStatus', 'Output copied.', 'ok');
      } catch (err) {
        setStatus('generateStatus', err.message, 'error');
      }
    }

    async function loadModel() {
      setBusy($('loadModel'), true);
      try {
        const data = await api('/api/model', {
          method: 'POST',
          body: JSON.stringify({
            model_name: $('modelName').value.trim(),
            sae_release: $('saeRelease').value.trim(),
            sae_id_template: $('saeTemplate').value.trim(),
            clear_steers: true
          })
        });
        renderHealth(data);
        await refreshState();
        setStatus('modelStatus', `Loaded ${data.model_name}.`, 'ok');
      } catch (err) {
        setStatus('modelStatus', err.message, 'error');
      } finally {
        setBusy($('loadModel'), false);
      }
    }

    function useSelectedModel() {
      const selected = selectedModelOption ? selectedModelOption.load_name : modelId();
      $('modelName').value = selected;
      $('cacheModel').value = cacheModelFromLoadName(selected);
      $('saeRelease').value = selectedModelOption?.sae_release_guess || `${selected.split('/').pop()}-res-jb`;
      highlightModelSelection();
      saveUiConfig();
      setStatus('modelStatus', `Prepared ${selected}; press Load model to switch.`, 'ok');
    }

    setupFieldHelp();
    setupWindowChrome();
    loadUiConfig();
    restoreSavedWindowLayout();

    $('moveWindows').addEventListener('click', () => setWindowMoveMode(!windowMoveMode));
    $('resetWindowLayout').addEventListener('click', resetWindowLayout);
    $('generate').addEventListener('click', () => generate({mode: 'steered', steersEnabled: true}));
    $('generateBaseline').addEventListener('click', generateBaseline);
    $('compareRuns').addEventListener('click', compareRuns);
    $('stopGenerate').addEventListener('click', stopGenerate);
    $('inspectOutput').addEventListener('click', inspectOutput);
    $('copyOutput').addEventListener('click', copyOutput);
    $('resetDefaults').addEventListener('click', resetUiConfig);
    $('copyLatestRun').addEventListener('click', copyLatestRun);
    $('exportRuns').addEventListener('click', exportRuns);
    $('clearRuns').addEventListener('click', clearResearchRuns);
    $('refreshState').addEventListener('click', () => refreshState().catch(err => setStatus('stateStatus', err.message, 'error')));
    $('reloadState').addEventListener('click', () => refreshState().catch(err => setStatus('stateStatus', err.message, 'error')));
    $('refreshHealth').addEventListener('click', refreshHealth);
    $('loadModel').addEventListener('click', loadModel);
    $('useSelectedModel').addEventListener('click', useSelectedModel);
    $('modelFilter').addEventListener('input', renderModelOptions);
    $('listModels').addEventListener('click', listModels);
    $('listSources').addEventListener('click', listSources);
    $('showCached').addEventListener('click', showCached);
    $('cacheLayers').addEventListener('click', cacheLayers);
    $('downloadSource').addEventListener('click', downloadSource);
    $('searchCache').addEventListener('click', searchCache);
    $('applyLabel').addEventListener('click', applySelectedLabel);
    $('setSteer').addEventListener('click', () => setSteer(false));
    $('appendSteer').addEventListener('click', () => setSteer(true));
    $('clearSteers').addEventListener('click', clearSteers);
    $('tokenPopover').addEventListener('mouseenter', () => clearTimeout(tokenPopoverTimer));
    $('tokenPopover').addEventListener('mouseleave', scheduleTokenPopoverHide);
    for (const id of fieldIdsForConfig()) {
      const node = $(id);
      const eventName = node.type === 'checkbox' || node.tagName === 'SELECT' ? 'change' : 'input';
      node.addEventListener(eventName, saveUiConfig);
    }
    $('search').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') searchCache();
    });
    $('sourceFilter').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') listSources();
    });
    document.addEventListener('keydown', (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        event.preventDefault();
        generate();
      }
      if (event.key === 'Escape') stopGenerate();
    });
    window.addEventListener('resize', clampWindowLayoutToViewport);

    refreshHealth();
    refreshState().catch(() => {});
    refreshCacheStatus().catch(() => {});
    loadResearchRuns();
    listModelOptions();
    updateOutputMeta();
  </script>
</body>
</html>
"""
