from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    raise unittest.SkipTest("fastapi is not installed") from exc

from steering.state import SteeringError
import server


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    @staticmethod
    def health() -> dict:
        return {"model_name": "fake-model", "device": "cpu", "busy": False}

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int | None):
        self.requests.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
            }
        )
        yield "hi"
        yield "!"


class ErrorBackend(FakeBackend):
    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int | None):
        raise SteeringError("bad steer")


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_backend = server.backend
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        server.backend = self.previous_backend

    def test_health_returns_backend_status(self) -> None:
        server.backend = FakeBackend()

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "fake-model")

    def test_health_reports_starting_when_backend_is_absent(self) -> None:
        server.backend = None

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "backend is still starting")

    def test_generate_non_stream_returns_text_and_passes_options(self) -> None:
        backend = FakeBackend()
        server.backend = backend

        response = self.client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_new_tokens": 2,
                "temperature": 0,
                "seed": 123,
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hi!"})
        self.assertEqual(
            backend.requests,
            [{"prompt": "hello", "max_new_tokens": 2, "temperature": 0.0, "seed": 123}],
        )

    def test_generate_validates_request_shape(self) -> None:
        server.backend = FakeBackend()

        response = self.client.post("/generate", json={"prompt": "", "max_new_tokens": 0})

        self.assertEqual(response.status_code, 422)

    def test_generate_rejects_non_finite_temperature(self) -> None:
        with self.assertRaises(ValueError):
            server.GenerateRequest(prompt="hello", temperature=float("nan"))

    def test_generate_maps_steering_errors_to_bad_request(self) -> None:
        server.backend = ErrorBackend()

        response = self.client.post("/generate", json={"prompt": "hello", "stream": False})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad steer")


if __name__ == "__main__":
    unittest.main()
