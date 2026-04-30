from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any


STATE_VERSION = 1
DEFAULT_STATE_DIR = ".steering"
DEFAULT_STATE_FILE = "state.json"


class SteeringError(ValueError):
    """Raised for invalid steering state or CLI input."""


@dataclass(frozen=True)
class SteerItem:
    feature_id: int
    strength: float
    layers: tuple[int, ...]
    label: str | None = None

    def __post_init__(self) -> None:
        if self.feature_id < 0:
            raise SteeringError("feature_id must be >= 0")
        if not self.layers:
            raise SteeringError("at least one layer is required")
        if any(layer < 0 for layer in self.layers):
            raise SteeringError("layers must be >= 0")
        if self.label is not None and not self.label.strip():
            raise SteeringError("label cannot be blank")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SteerItem":
        try:
            feature_id = int(data["feature_id"])
            strength = float(data["strength"])
            layers = tuple(int(layer) for layer in data["layers"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SteeringError(f"invalid steer item: {data!r}") from exc

        label = data.get("label")
        if label is not None:
            label = str(label)
        return cls(
            feature_id=feature_id,
            strength=strength,
            layers=layers,
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "feature_id": self.feature_id,
            "strength": self.strength,
            "layers": list(self.layers),
        }
        if self.label:
            data["label"] = self.label
        return data


@dataclass(frozen=True)
class SteeringState:
    items: tuple[SteerItem, ...] = field(default_factory=tuple)
    updated_at: str = field(default_factory=lambda: now_iso())
    version: int = STATE_VERSION

    @classmethod
    def empty(cls) -> "SteeringState":
        return cls(items=tuple(), updated_at=now_iso(), version=STATE_VERSION)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SteeringState":
        version = int(data.get("version", STATE_VERSION))
        if version != STATE_VERSION:
            raise SteeringError(f"unsupported steering state version: {version}")
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            raise SteeringError("state 'items' must be a list")
        updated_at = str(data.get("updated_at") or now_iso())
        return cls(
            items=tuple(SteerItem.from_dict(item) for item in raw_items),
            updated_at=updated_at,
            version=version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "items": [item.to_dict() for item in self.items],
        }

    def replace(self, item: SteerItem) -> "SteeringState":
        return SteeringState(items=(item,), updated_at=now_iso())

    def append(self, item: SteerItem) -> "SteeringState":
        return SteeringState(items=(*self.items, item), updated_at=now_iso())

    def clear(self) -> "SteeringState":
        return SteeringState.empty()

    @property
    def is_empty(self) -> bool:
        return not self.items


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_state_path(cwd: Path | None = None) -> Path:
    env_path = os.environ.get("STEERING_STATE_PATH")
    if env_path:
        return Path(env_path).expanduser()
    base = cwd if cwd is not None else Path.cwd()
    return base / DEFAULT_STATE_DIR / DEFAULT_STATE_FILE


def parse_layers(raw: str) -> tuple[int, ...]:
    layers: list[int] = []
    for part in raw.replace(" ", ",").split(","):
        value = part.strip()
        if not value:
            continue
        try:
            layer = int(value)
        except ValueError as exc:
            raise SteeringError(f"invalid layer value: {value!r}") from exc
        if layer < 0:
            raise SteeringError("layers must be >= 0")
        layers.append(layer)

    if not layers:
        raise SteeringError("at least one layer is required")
    return tuple(dict.fromkeys(layers))


def load_state(path: Path | None = None) -> SteeringState:
    state_path = path or default_state_path()
    if not state_path.exists():
        return SteeringState.empty()

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SteeringError(f"invalid steering state JSON at {state_path}") from exc
    if not isinstance(data, dict):
        raise SteeringError(f"state file must contain an object: {state_path}")
    return SteeringState.from_dict(data)


def save_state(state: SteeringState, path: Path | None = None) -> Path:
    state_path = path or default_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp_path = state_path.with_name(f".{state_path.name}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, state_path)
    return state_path


def update_state(item: SteerItem, append: bool, path: Path | None = None) -> SteeringState:
    current = load_state(path)
    updated = current.append(item) if append else current.replace(item)
    save_state(updated, path)
    return updated


def clear_state(path: Path | None = None) -> SteeringState:
    cleared = SteeringState.empty()
    save_state(cleared, path)
    return cleared
