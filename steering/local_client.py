from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Iterator
from urllib import error, request


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_CLIENT_TIMEOUT = 60.0


class LocalServerError(RuntimeError):
    """Raised when the local steering server cannot satisfy a request."""


@dataclass(frozen=True)
class LocalServerClient:
    base_url: str = DEFAULT_SERVER_URL
    timeout: float = DEFAULT_CLIENT_TIMEOUT

    @classmethod
    def from_env(cls, base_url: str | None = None) -> "LocalServerClient":
        raw_url = base_url or os.environ.get("STEERING_SERVER_URL") or DEFAULT_SERVER_URL
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"http://{raw_url}"
        return cls(
            base_url=raw_url.rstrip("/"),
            timeout=parse_timeout_env("STEERING_CLIENT_TIMEOUT", DEFAULT_CLIENT_TIMEOUT),
        )

    def health(self) -> dict[str, Any]:
        return self._json_request("GET", "/health")

    def generate(
        self,
        *,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        seed: int | None,
        stream: bool,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if seed is not None:
            payload["seed"] = seed

        req = self._build_request("POST", "/generate", payload)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                if stream:
                    while True:
                        chunk = response.read(1)
                        if not chunk:
                            break
                        yield chunk.decode("utf-8", errors="replace")
                    return

                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LocalServerError(f"server returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LocalServerError(f"could not reach local steering server at {self.base_url}") from exc

        data = json.loads(body)
        if "error" in data:
            raise LocalServerError(str(data["error"]))
        yield str(data.get("text", ""))

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        req = self._build_request(method, path, payload)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LocalServerError(f"server returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise LocalServerError(f"could not reach local steering server at {self.base_url}") from exc

        data = json.loads(body or "{}")
        if not isinstance(data, dict):
            raise LocalServerError(f"unexpected server response: {data!r}")
        if "error" in data:
            raise LocalServerError(str(data["error"]))
        return data

    def _build_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> request.Request:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )


def parse_timeout_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise LocalServerError(f"{name} must be a number") from exc
    if value <= 0:
        raise LocalServerError(f"{name} must be greater than 0")
    return value
