from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    raise unittest.SkipTest("fastapi is not installed") from exc

from steering.feature_cache import CachedSource, FeatureCache, FeatureLabel
from steering.state import SteerItem, SteeringError, load_state, save_state
import server


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.config = server.BackendConfig(model_name="fake-model", device="cpu")

    def health(self) -> dict:
        return {
            "model_name": self.config.model_name,
            "sae_release": self.config.sae_release,
            "sae_id_template": self.config.sae_id_template,
            "device": self.config.device,
            "busy": False,
        }

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
        self.previous_state_path = server.state_path
        self.previous_feature_cache_path = server.feature_cache_path
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / "state.json"
        self.cache_path = Path(self.tmp.name) / "feature-cache.sqlite3"
        server.state_path = self.state_path
        server.feature_cache_path = self.cache_path
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        server.backend = self.previous_backend
        server.state_path = self.previous_state_path
        server.feature_cache_path = self.previous_feature_cache_path
        self.tmp.cleanup()

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

    def test_web_ui_exposes_model_and_neuronpedia_controls(self) -> None:
        server.backend = FakeBackend()

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.text
        self.assertIn("Backend Model", html)
        self.assertIn("Neuronpedia Sources", html)
        self.assertIn("/api/model", html)

    def test_state_api_sets_appends_and_clears_steers(self) -> None:
        server.backend = FakeBackend()

        response = self.client.post(
            "/api/state/items",
            json={
                "feature_id": 204,
                "strength": 10,
                "layers": [6],
                "model_id": "gpt2-small",
                "label": "time phrases",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"][0]["feature_id"], 204)
        self.assertEqual(load_state(self.state_path).items[0].model_id, "gpt2-small")

        response = self.client.post(
            "/api/state/items",
            json={"feature_id": 7, "strength": 1.5, "sae_id": "custom-sae", "append": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["items"]), 2)

        response = self.client.delete("/api/state")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])

    def test_neuronpedia_model_and_source_endpoints_use_dataset_exports(self) -> None:
        class FakeDatasetClient:
            @staticmethod
            def list_models() -> list[str]:
                return ["gpt2-small", "pythia-70m"]

            @staticmethod
            def list_sources(model_id: str) -> list[str]:
                self.assertEqual(model_id, "pythia-70m")
                return ["0-res-jb", "blocks.0.hook_resid_post", "1-res-jb"]

        with mock.patch.object(server, "NeuronpediaDatasetClient", return_value=FakeDatasetClient()):
            models_response = self.client.get("/api/neuronpedia/models")
            sources_response = self.client.get(
                "/api/neuronpedia/sources",
                params={"model_id": "pythia-70m", "contains": "res-jb"},
            )

        self.assertEqual(models_response.status_code, 200)
        self.assertEqual(models_response.json()["models"], ["gpt2-small", "pythia-70m"])
        self.assertEqual(sources_response.status_code, 200)
        self.assertEqual(sources_response.json()["sources"], ["0-res-jb", "1-res-jb"])

    def test_cache_status_source_and_search_endpoints(self) -> None:
        FeatureCache(self.cache_path).replace_source(
            "gpt2-small",
            "6-res-jb",
            [
                FeatureLabel(
                    model_id="gpt2-small",
                    source_id="6-res-jb",
                    feature_id=204,
                    description="time phrases and calendar dates",
                )
            ],
        )

        status_response = self.client.get("/api/cache/status")
        search_response = self.client.get(
            "/api/cache/search",
            params={"query": "calendar", "model_id": "gpt2-small", "source_id": "6-res-jb"},
        )

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["sources"][0]["source_id"], "6-res-jb")
        self.assertEqual(search_response.status_code, 200)
        self.assertEqual(search_response.json()["labels"][0]["feature_id"], 204)

        with mock.patch.object(
            server,
            "build_source_cache",
            return_value=CachedSource(
                model_id="gpt2-small",
                source_id="7-res-jb",
                label_count=3,
                feature_count=2,
                fetched_at="2026-05-01T00:00:00+00:00",
            ),
        ) as mocked_build:
            download_response = self.client.post(
                "/api/cache/source",
                json={"model_id": "gpt2-small", "source_id": "7-res-jb", "max_files": 1},
            )

        self.assertEqual(download_response.status_code, 200)
        self.assertEqual(download_response.json()["label_count"], 3)
        mocked_build.assert_called_once_with(
            model_id="gpt2-small",
            source_id="7-res-jb",
            cache_path=self.cache_path,
            max_files=1,
        )

    def test_load_model_replaces_backend_and_can_clear_state(self) -> None:
        server.backend = FakeBackend()
        save_state(
            load_state(self.state_path).append(
                SteerItem(feature_id=204, strength=10, layers=(6,), model_id="gpt2-small")
            ),
            self.state_path,
        )

        class ReplacementBackend(FakeBackend):
            def __init__(self, config: server.BackendConfig) -> None:
                self.requests = []
                self.config = config

        with mock.patch.object(server, "TransformerLensSteeringBackend", ReplacementBackend):
            response = self.client.post(
                "/api/model",
                json={
                    "model_name": "pythia-70m",
                    "sae_release": "pythia-70m-deduped-res-sm",
                    "sae_id_template": "blocks.{layer}.hook_resid_post",
                    "clear_steers": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_name"], "pythia-70m")
        self.assertEqual(response.json()["sae_release"], "pythia-70m-deduped-res-sm")
        self.assertEqual(response.json()["sae_id_template"], "blocks.{layer}.hook_resid_post")
        self.assertEqual(load_state(self.state_path).items, tuple())
        self.assertEqual(server.backend.config.model_name, "pythia-70m")


if __name__ == "__main__":
    unittest.main()
