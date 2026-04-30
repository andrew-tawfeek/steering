from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Iterator
from urllib import error, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    """Raised when the Ollama API cannot satisfy a request."""


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = DEFAULT_OLLAMA_URL
    timeout: float = 120.0

    @classmethod
    def from_env(cls, base_url: str | None = None) -> "OllamaClient":
        raw_url = base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"http://{raw_url}"
        return cls(base_url=raw_url.rstrip("/"))

    def tags(self) -> dict[str, Any]:
        return self._json_request("GET", "/api/tags")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str = "",
        max_tokens: int = 120,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        if stream:
            yield from self._stream_generate(payload)
            return

        response = self._json_request("POST", "/api/generate", payload)
        text = response.get("response", "")
        if text:
            yield str(text)

    def _stream_generate(self, payload: dict[str, Any]) -> Iterator[str]:
        req = self._build_request("POST", "/api/generate", payload)
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if "error" in event:
                        raise OllamaError(str(event["error"]))
                    token = event.get("response")
                    if token:
                        yield str(token)
                    if event.get("done"):
                        break
        except error.URLError as exc:
            raise OllamaError(f"could not reach Ollama at {self.base_url}") from exc

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
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise OllamaError(f"could not reach Ollama at {self.base_url}") from exc

        if not body:
            return {}
        data = json.loads(body)
        if not isinstance(data, dict):
            raise OllamaError(f"unexpected Ollama response: {data!r}")
        if "error" in data:
            raise OllamaError(str(data["error"]))
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
