from __future__ import annotations

from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from steering.state import SteeringError, default_state_path
from steering.tlens_backend import BackendConfig, TransformerLensSteeringBackend


app = FastAPI(title="Steering CLI TransformerLens Backend")
backend: TransformerLensSteeringBackend | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=60, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0)
    seed: int | None = None
    stream: bool = True


@app.on_event("startup")
def startup() -> None:
    global backend
    state_path = Path(default_state_path())
    backend = TransformerLensSteeringBackend(BackendConfig.from_env(state_path=state_path))


@app.get("/health")
def health() -> dict:
    return get_backend().health()


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
