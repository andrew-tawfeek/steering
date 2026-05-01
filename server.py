from __future__ import annotations

from pathlib import Path
import threading
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from steering.feature_cache import (
    FeatureCache,
    FeatureCacheError,
    NeuronpediaDatasetClient,
    build_source_cache,
    default_feature_cache_path,
)
from steering.state import (
    SteerItem,
    SteeringError,
    clear_state,
    default_state_path,
    load_state,
    update_state,
)
from steering.tlens_backend import BackendConfig, TransformerLensSteeringBackend


app = FastAPI(title="Steering CLI TransformerLens Backend")
backend: TransformerLensSteeringBackend | None = None
backend_lock = threading.Lock()
state_path = Path(default_state_path())
feature_cache_path = Path(default_feature_cache_path())


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=60, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, allow_inf_nan=False)
    seed: int | None = None
    stream: bool = True


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


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        tokens = get_backend().generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            seed=request.seed,
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


WEB_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Steering</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #1c2330;
      --muted: #667085;
      --line: #d7dce3;
      --accent: #0f766e;
      --danger: #b42318;
      --ok: #067647;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      min-height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 10px 18px;
      background: #202936;
      color: #fff;
      border-bottom: 1px solid #151b24;
    }
    header h1 { margin: 0; font-size: 18px; font-weight: 650; }
    #health { color: #d0d5dd; font-size: 13px; text-align: right; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.2fr) minmax(420px, 0.8fr);
      gap: 14px;
      padding: 14px;
      max-width: 1500px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    h2 { margin: 0 0 12px; font-size: 15px; }
    h3 { margin: 16px 0 8px; font-size: 13px; color: #344054; }
    label { display: block; margin: 9px 0 5px; color: #344054; font-weight: 600; font-size: 12px; }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    textarea { min-height: 170px; resize: vertical; line-height: 1.45; }
    button {
      border: 1px solid #0d5f59;
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 8px 11px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      min-height: 36px;
    }
    button.secondary { color: #1f2937; background: #fff; border-color: var(--line); }
    button.danger { background: var(--danger); border-color: #912018; }
    button:disabled { opacity: .55; cursor: wait; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .status { min-height: 20px; color: var(--muted); margin-top: 8px; }
    .status.error { color: var(--danger); }
    .status.ok { color: var(--ok); }
    pre {
      min-height: 180px;
      white-space: pre-wrap;
      background: #101828;
      color: #f2f4f7;
      border-radius: 8px;
      padding: 12px;
      overflow: auto;
      line-height: 1.45;
    }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; vertical-align: top; }
    th { color: #475467; font-size: 12px; }
    tr.selectable { cursor: pointer; }
    tr.selectable:hover { background: #f2f4f7; }
    .muted { color: var(--muted); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; padding: 10px; }
      header { align-items: flex-start; flex-direction: column; }
      #health { text-align: left; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Steering</h1>
    <div id="health">Checking backend...</div>
  </header>
  <main>
    <section>
      <h2>Generate</h2>
      <label for="prompt">Prompt</label>
      <textarea id="prompt">Today the weather report says</textarea>
      <div class="grid">
        <div><label for="maxTokens">Max tokens</label><input id="maxTokens" type="number" min="1" max="512" value="40"></div>
        <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" step="0.1" value="0"></div>
      </div>
      <div class="actions">
        <button id="generate">Generate</button>
        <button class="secondary" id="refreshState">Refresh state</button>
      </div>
      <pre id="output"></pre>
      <div class="status" id="generateStatus"></div>

      <h2>Backend Model</h2>
      <div class="grid">
        <div><label for="modelName">TransformerLens model</label><input id="modelName" value="gpt2-small"></div>
        <div><label for="saeRelease">SAE Lens release</label><input id="saeRelease" value="gpt2-small-res-jb"></div>
      </div>
      <label for="saeTemplate">Layer SAE id template</label>
      <input id="saeTemplate" value="blocks.{layer}.hook_resid_pre">
      <div class="actions">
        <button id="loadModel">Load model</button>
        <button class="secondary" id="useSelectedModel">Use selected Neuronpedia model</button>
      </div>
      <div class="status" id="modelStatus"></div>
    </section>

    <section>
      <h2>Neuronpedia Sources</h2>
      <div class="grid">
        <div><label for="cacheModel">Model</label><input id="cacheModel" value="gpt2-small"></div>
        <div><label for="sourceFilter">Source filter</label><input id="sourceFilter" value="res-jb"></div>
      </div>
      <div class="actions">
        <button id="listModels">List models</button>
        <button id="listSources">List sources</button>
        <button id="downloadSource">Download selected source</button>
      </div>
      <table>
        <thead><tr><th>Model/source</th><th>Cached</th></tr></thead>
        <tbody id="sources"></tbody>
      </table>
      <div class="status" id="sourceStatus"></div>

      <h2>Search Cached Labels</h2>
      <label for="search">Search</label>
      <input id="search" value="time phrases">
      <div class="actions">
        <button id="searchCache">Search</button>
      </div>
      <table>
        <thead><tr><th>Feature</th><th>Source</th><th>Description</th></tr></thead>
        <tbody id="labels"></tbody>
      </table>

      <h2>Active Steer</h2>
      <div class="grid">
        <div><label for="featureId">Feature id</label><input id="featureId" type="number" min="0" value="204"></div>
        <div><label for="strength">Strength</label><input id="strength" type="number" step="0.1" value="10"></div>
      </div>
      <div class="grid">
        <div><label for="layers">Layers</label><input id="layers" value="6"></div>
        <div><label for="sourceId">SAE/source id</label><input id="sourceId" value="6-res-jb"></div>
      </div>
      <label for="label">Label</label>
      <input id="label">
      <div class="actions">
        <button id="setSteer">Set only</button>
        <button class="secondary" id="appendSteer">Append</button>
        <button class="danger" id="clearSteers">Clear</button>
      </div>
      <pre id="state"></pre>
      <div class="status" id="stateStatus"></div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    let selectedModel = null;
    let selectedSource = null;
    let cacheStatus = new Map();

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
    function setBusy(button, busy) { button.disabled = busy; }
    function renderHealth(data) {
      $('health').textContent = `${data.model_name} on ${data.device} | ${data.sae_release}`;
      $('modelName').value = data.model_name || $('modelName').value;
      $('saeRelease').value = data.sae_release || $('saeRelease').value;
      $('saeTemplate').value = data.sae_id_template || $('saeTemplate').value;
    }
    async function refreshHealth() {
      try { renderHealth(await api('/health')); }
      catch (err) { $('health').textContent = `Backend unavailable: ${err.message}`; }
    }
    async function refreshState() {
      const state = await api('/api/state');
      $('state').textContent = JSON.stringify(state, null, 2);
    }
    async function refreshCacheStatus() {
      const data = await api('/api/cache/status');
      cacheStatus = new Map(data.sources.map(row => [`${row.model_id}/${row.source_id}`, row]));
    }
    function renderSources(rows, mode) {
      const body = $('sources');
      body.innerHTML = '';
      for (const value of rows) {
        const tr = document.createElement('tr');
        tr.className = 'selectable';
        const cached = mode === 'sources' ? cacheStatus.get(`${$('cacheModel').value.trim()}/${value}`) : null;
        const nameCell = document.createElement('td');
        nameCell.textContent = value;
        const cacheCell = document.createElement('td');
        cacheCell.textContent = cached ? `${cached.label_count} labels` : '-';
        if (!cached) cacheCell.className = 'muted';
        tr.append(nameCell, cacheCell);
        tr.addEventListener('click', () => {
          if (mode === 'models') {
            selectedModel = value;
            selectedSource = null;
            $('cacheModel').value = value;
            $('modelName').value = value;
            $('saeRelease').value = `${value}-res-jb`;
            setStatus('sourceStatus', `Selected model ${value}.`, 'ok');
          } else {
            selectedSource = value;
            $('sourceId').value = value;
            if (/^\d+-res-jb$/.test(value)) $('layers').value = value.split('-')[0];
            setStatus('sourceStatus', `Selected source ${value}.`, 'ok');
          }
        });
        body.appendChild(tr);
      }
    }
    function selectedModelId() { return $('cacheModel').value.trim() || selectedModel || $('modelName').value.trim(); }
    async function listModels() {
      const button = $('listModels'); setBusy(button, true);
      try {
        const data = await api('/api/neuronpedia/models');
        renderSources(data.models, 'models');
        setStatus('sourceStatus', `Found ${data.count} Neuronpedia export models.`, 'ok');
      } catch (err) { setStatus('sourceStatus', err.message, 'error'); }
      finally { setBusy(button, false); }
    }
    async function listSources() {
      const button = $('listSources'); setBusy(button, true);
      try {
        await refreshCacheStatus();
        const params = new URLSearchParams({model_id: selectedModelId()});
        if ($('sourceFilter').value.trim()) params.set('contains', $('sourceFilter').value.trim());
        const data = await api(`/api/neuronpedia/sources?${params}`);
        renderSources(data.sources, 'sources');
        setStatus('sourceStatus', `Found ${data.count} sources for ${data.model_id}.`, 'ok');
      } catch (err) { setStatus('sourceStatus', err.message, 'error'); }
      finally { setBusy(button, false); }
    }
    async function downloadSource() {
      const source = selectedSource || $('sourceId').value.trim();
      if (!source) return setStatus('sourceStatus', 'Select or enter a source first.', 'error');
      const button = $('downloadSource'); setBusy(button, true);
      try {
        const row = await api('/api/cache/source', {
          method: 'POST',
          body: JSON.stringify({model_id: selectedModelId(), source_id: source})
        });
        await refreshCacheStatus();
        setStatus('sourceStatus', `Cached ${row.label_count} labels for ${row.model_id}/${row.source_id}.`, 'ok');
      } catch (err) { setStatus('sourceStatus', err.message, 'error'); }
      finally { setBusy(button, false); }
    }
    async function searchCache() {
      try {
        const params = new URLSearchParams({query: $('search').value.trim(), model_id: selectedModelId(), limit: '25'});
        const source = $('sourceId').value.trim();
        if (source) params.set('source_id', source);
        const data = await api(`/api/cache/search?${params}`);
        const body = $('labels');
        body.innerHTML = '';
        for (const label of data.labels) {
          const tr = document.createElement('tr');
          tr.className = 'selectable';
          const featureCell = document.createElement('td');
          const sourceCell = document.createElement('td');
          const descriptionCell = document.createElement('td');
          featureCell.textContent = label.feature_id;
          sourceCell.textContent = `${label.model_id}/${label.source_id}`;
          descriptionCell.textContent = label.description;
          tr.append(featureCell, sourceCell, descriptionCell);
          tr.addEventListener('click', () => {
            $('featureId').value = label.feature_id;
            $('sourceId').value = label.source_id;
            $('cacheModel').value = label.model_id;
            $('modelName').value = label.model_id;
            $('label').value = label.description;
            if (/^\d+-res-jb$/.test(label.source_id)) $('layers').value = label.source_id.split('-')[0];
          });
          body.appendChild(tr);
        }
        setStatus('sourceStatus', `Found ${data.count} cached labels.`, data.count ? 'ok' : '');
      } catch (err) { setStatus('sourceStatus', err.message, 'error'); }
    }
    function steerPayload(append) {
      const source = $('sourceId').value.trim();
      const residual = source.match(/^(\d+)-res-jb$/);
      const layers = $('layers').value.split(/[,\s]+/).filter(Boolean).map(Number);
      return {
        feature_id: Number($('featureId').value),
        strength: Number($('strength').value),
        layers: residual ? [Number(residual[1])] : layers,
        sae_id: residual ? null : source || null,
        model_id: $('modelName').value.trim() || selectedModelId(),
        label: $('label').value.trim() || null,
        append
      };
    }
    async function setSteer(append) {
      try {
        const state = await api('/api/state/items', {method: 'POST', body: JSON.stringify(steerPayload(append))});
        $('state').textContent = JSON.stringify(state, null, 2);
        setStatus('stateStatus', append ? 'Steer appended.' : 'Steer set.', 'ok');
      } catch (err) { setStatus('stateStatus', err.message, 'error'); }
    }
    async function clearSteers() {
      const state = await api('/api/state', {method: 'DELETE'});
      $('state').textContent = JSON.stringify(state, null, 2);
      setStatus('stateStatus', 'Steers cleared.', 'ok');
    }
    async function generate() {
      const button = $('generate'); setBusy(button, true);
      $('output').textContent = '';
      try {
        const data = await api('/generate', {
          method: 'POST',
          body: JSON.stringify({
            prompt: $('prompt').value,
            max_new_tokens: Number($('maxTokens').value),
            temperature: Number($('temperature').value),
            stream: false
          })
        });
        $('output').textContent = data.text;
        setStatus('generateStatus', 'Generation complete.', 'ok');
      } catch (err) { setStatus('generateStatus', err.message, 'error'); }
      finally { setBusy(button, false); }
    }
    async function loadModel() {
      const button = $('loadModel'); setBusy(button, true);
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
      } catch (err) { setStatus('modelStatus', err.message, 'error'); }
      finally { setBusy(button, false); }
    }
    $('generate').addEventListener('click', generate);
    $('refreshState').addEventListener('click', refreshState);
    $('loadModel').addEventListener('click', loadModel);
    $('useSelectedModel').addEventListener('click', () => {
      const model = selectedModelId();
      $('modelName').value = model;
      $('saeRelease').value = `${model}-res-jb`;
      setStatus('modelStatus', `Prepared ${model}; press Load model to download/switch.`, 'ok');
    });
    $('listModels').addEventListener('click', listModels);
    $('listSources').addEventListener('click', listSources);
    $('downloadSource').addEventListener('click', downloadSource);
    $('searchCache').addEventListener('click', searchCache);
    $('setSteer').addEventListener('click', () => setSteer(false));
    $('appendSteer').addEventListener('click', () => setSteer(true));
    $('clearSteers').addEventListener('click', clearSteers);
    refreshHealth();
    refreshState().catch(() => {});
    refreshCacheStatus().catch(() => {});
  </script>
</body>
</html>
"""
