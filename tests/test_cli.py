from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import steer
from steering import __version__
from steering.feature_cache import CachedSource, FeatureCache, FeatureLabel


class CliParserTests(unittest.TestCase):
    def test_version_flag_prints_package_version(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = steer.main(["--version"])

        self.assertEqual(result, 0)
        self.assertTrue(stdout.getvalue().strip().endswith(f" {__version__}"))

    def test_generate_rejects_zero_max_tokens_before_request(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["generate", "hello", "--max-tokens", "0"])

        self.assertEqual(result, 2)
        self.assertIn("must be >= 1", stderr.getvalue())

    def test_generate_rejects_too_many_max_tokens_before_request(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["generate", "hello", "--max-tokens", "513"])

        self.assertEqual(result, 2)
        self.assertIn("must be <= 512", stderr.getvalue())

    def test_generate_rejects_negative_temperature_before_request(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["generate", "hello", "--temperature", "-0.1"])

        self.assertEqual(result, 2)
        self.assertIn("must be >= 0", stderr.getvalue())

    def test_generate_rejects_non_finite_temperature_before_request(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["generate", "hello", "--temperature", "nan"])

        self.assertEqual(result, 2)
        self.assertIn("must be finite", stderr.getvalue())

    def test_update_rejects_non_finite_strength_before_saving(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["update", "--feature-id", "204", "--strength", "inf", "--layers", "6"])

        self.assertEqual(result, 2)
        self.assertIn("must be finite", stderr.getvalue())

    def test_feature_rejects_negative_feature_id_before_api_call(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(steer.NeuronpediaClient, "from_env") as from_env:
            with contextlib.redirect_stderr(stderr):
                result = steer.main(["feature", "--layer", "6", "--feature-id", "-1"])

        self.assertEqual(result, 2)
        self.assertFalse(from_env.called)
        self.assertIn("must be >= 0", stderr.getvalue())

    def test_feature_cache_search_rejects_zero_limit(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["feature-cache", "search", "time", "--limit", "0"])

        self.assertEqual(result, 2)
        self.assertIn("must be >= 1", stderr.getvalue())

    def test_serve_runs_uvicorn_with_cli_options(self) -> None:
        calls: list[dict] = []

        class FakeUvicorn:
            @staticmethod
            def run(app: str, *, host: str, port: int, reload: bool) -> None:
                calls.append({"app": app, "host": host, "port": port, "reload": reload})

        with mock.patch.dict(sys.modules, {"uvicorn": FakeUvicorn}):
            result = steer.main(["serve", "--host", "0.0.0.0", "--port", "8123", "--reload"])

        self.assertEqual(result, 0)
        self.assertEqual(calls, [{"app": "server:app", "host": "0.0.0.0", "port": 8123, "reload": True}])

    def test_serve_rejects_invalid_port(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = steer.main(["serve", "--port", "0"])

        self.assertEqual(result, 2)
        self.assertIn("must be >= 1", stderr.getvalue())

    def test_health_prints_backend_json(self) -> None:
        class FakeClient:
            @staticmethod
            def health() -> dict:
                return {"model_name": "fake-model", "device": "cpu"}

        stdout = io.StringIO()
        with mock.patch.object(steer.LocalServerClient, "from_env", return_value=FakeClient()):
            with contextlib.redirect_stdout(stdout):
                result = steer.main(["health", "--server-url", "http://fake-server"])

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"model_name": "fake-model", "device": "cpu"})

    def test_generate_prints_streamed_chunks(self) -> None:
        class FakeClient:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                yield "he"
                yield "llo"

        fake_client = FakeClient()
        stdout = io.StringIO()
        with mock.patch.object(steer.LocalServerClient, "from_env", return_value=fake_client):
            with contextlib.redirect_stdout(stdout):
                result = steer.main(
                    [
                        "generate",
                        "prompt",
                        "--server-url",
                        "http://fake-server",
                        "--max-tokens",
                        "3",
                        "--temperature",
                        "0",
                        "--seed",
                        "12",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "hello\n")
        self.assertEqual(
            fake_client.kwargs,
            {
                "prompt": "prompt",
                "max_new_tokens": 3,
                "temperature": 0.0,
                "seed": 12,
                "stream": True,
            },
        )

    def test_generate_reads_prompt_from_stdin_when_omitted(self) -> None:
        class FakeClient:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                yield "ok"

        fake_client = FakeClient()
        stdout = io.StringIO()
        with mock.patch.object(steer.LocalServerClient, "from_env", return_value=fake_client):
            with mock.patch.object(steer.sys, "stdin", io.StringIO("stdin prompt\n")):
                with contextlib.redirect_stdout(stdout):
                    result = steer.main(["generate", "--no-stream"])

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "ok\n")
        self.assertEqual(fake_client.kwargs["prompt"], "stdin prompt")
        self.assertFalse(fake_client.kwargs["stream"])

    def test_update_show_and_clear_state_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                update_result = steer.main(
                    [
                        "--state-path",
                        str(state_path),
                        "update",
                        "--feature-id",
                        "204",
                        "--strength",
                        "10",
                        "--layers",
                        "6,8",
                        "--label",
                        "time phrases",
                        "--json",
                    ]
                )

            self.assertEqual(update_result, 0)
            updated = json.loads(stdout.getvalue())
            self.assertEqual(updated["items"][0]["feature_id"], 204)
            self.assertEqual(updated["items"][0]["layers"], [6, 8])
            self.assertEqual(updated["items"][0]["label"], "time phrases")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                show_result = steer.main(["--state-path", str(state_path), "show", "--json"])

            self.assertEqual(show_result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["items"][0]["feature_id"], 204)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                clear_result = steer.main(["--state-path", str(state_path), "clear", "--json"])

            self.assertEqual(clear_result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["items"], [])

    def test_update_append_adds_second_steer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    steer.main(
                        [
                            "--state-path",
                            str(state_path),
                            "update",
                            "--feature-id",
                            "204",
                            "--strength",
                            "10",
                            "--layers",
                            "6",
                        ]
                    ),
                    0,
                )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                append_result = steer.main(
                    [
                        "--state-path",
                        str(state_path),
                        "update",
                        "--feature-id",
                        "7",
                        "--strength",
                        "-3",
                        "--layers",
                        "8",
                        "--append",
                        "--json",
                    ]
                )

            self.assertEqual(append_result, 0)
            self.assertEqual([item["feature_id"] for item in json.loads(stdout.getvalue())["items"]], [204, 7])

    def test_feature_cache_search_show_and_status_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "features.sqlite3"
            FeatureCache(cache_path).replace_source(
                "gpt2-small",
                "6-res-jb",
                [
                    FeatureLabel(
                        "gpt2-small",
                        "6-res-jb",
                        204,
                        "time-related phrases",
                        type_name="test-type",
                        explanation_model_name="test-explainer",
                    )
                ],
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                search_result = steer.main(
                    [
                        "feature-cache",
                        "search",
                        "time",
                        "--model-id",
                        "gpt2-small",
                        "--source",
                        "6-res-jb",
                        "--cache-path",
                        str(cache_path),
                        "--json",
                    ]
                )

            self.assertEqual(search_result, 0)
            labels = json.loads(stdout.getvalue())
            self.assertEqual(labels[0]["feature_id"], 204)
            self.assertEqual(labels[0]["explanation_model_name"], "test-explainer")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                show_result = steer.main(
                    [
                        "feature-cache",
                        "show",
                        "--model-id",
                        "gpt2-small",
                        "--source",
                        "6-res-jb",
                        "--feature-id",
                        "204",
                        "--cache-path",
                        str(cache_path),
                        "--json",
                    ]
                )

            self.assertEqual(show_result, 0)
            self.assertEqual(json.loads(stdout.getvalue())[0]["description"], "time-related phrases")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status_result = steer.main(["feature-cache", "status", "--cache-path", str(cache_path), "--json"])

            self.assertEqual(status_result, 0)
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["sources"][0]["source_id"], "6-res-jb")
            self.assertEqual(status["sources"][0]["label_count"], 1)

    def test_feature_cache_models_and_sources_json(self) -> None:
        class FakeDatasetClient:
            @staticmethod
            def list_models() -> list[str]:
                return ["gpt2-small"]

            @staticmethod
            def list_sources(model_id: str) -> list[str]:
                return ["6-res-jb", "mlp-out"]

        stdout = io.StringIO()
        with mock.patch.object(steer, "NeuronpediaDatasetClient", return_value=FakeDatasetClient()):
            with contextlib.redirect_stdout(stdout):
                models_result = steer.main(["feature-cache", "models", "--json"])

        self.assertEqual(models_result, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"models": ["gpt2-small"], "count": 1})

        stdout = io.StringIO()
        with mock.patch.object(steer, "NeuronpediaDatasetClient", return_value=FakeDatasetClient()):
            with contextlib.redirect_stdout(stdout):
                sources_result = steer.main(
                    ["feature-cache", "sources", "--model-id", "gpt2-small", "--contains", "res", "--json"]
                )

        self.assertEqual(sources_result, 0)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"model_id": "gpt2-small", "sources": ["6-res-jb"], "count": 1},
        )

    def test_feature_cache_download_json_reports_cached_sources(self) -> None:
        class FakeDatasetClient:
            ...

        calls: list[dict] = []

        def fake_build_source_cache(**kwargs):
            calls.append(kwargs)
            return CachedSource(
                model_id=kwargs["model_id"],
                source_id=kwargs["source_id"],
                label_count=2,
                feature_count=1,
                fetched_at="2026-05-01T00:00:00+00:00",
            )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "features.sqlite3"
            stdout = io.StringIO()
            with mock.patch.object(steer, "NeuronpediaDatasetClient", return_value=FakeDatasetClient()):
                with mock.patch.object(steer, "build_source_cache", side_effect=fake_build_source_cache):
                    with contextlib.redirect_stdout(stdout):
                        result = steer.main(
                            [
                                "feature-cache",
                                "download",
                                "--model-id",
                                "gpt2-small",
                                "--source",
                                "6-res-jb",
                                "--cache-path",
                                str(cache_path),
                                "--json",
                            ]
                        )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["cache_path"], str(cache_path))
        self.assertEqual(payload["sources"][0]["source_id"], "6-res-jb")
        self.assertEqual(payload["sources"][0]["label_count"], 2)
        self.assertEqual(calls[0]["source_id"], "6-res-jb")
        self.assertEqual(calls[0]["cache_path"], cache_path)

    def test_doctor_json_reports_missing_dependencies_without_server_check(self) -> None:
        stdout = io.StringIO()

        def fake_find_spec(module_name: str):
            if module_name == "textual":
                return None
            return object()

        with mock.patch.object(steer.importlib.util, "find_spec", side_effect=fake_find_spec):
            with contextlib.redirect_stdout(stdout):
                result = steer.main(["doctor", "--skip-server", "--json"])

        self.assertEqual(result, 1)
        checks = json.loads(stdout.getvalue())
        names = {check["name"] for check in checks}
        self.assertIn("python", names)
        self.assertIn("textual", names)
        self.assertIn("backend", names)
        self.assertFalse(next(check for check in checks if check["name"] == "textual")["ok"])
        self.assertEqual(next(check for check in checks if check["name"] == "backend")["detail"], "skipped")

    def test_doctor_can_pass_with_dependencies_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = io.StringIO()

            def fake_find_spec(module_name: str):
                return object()

            class FakeDoctorClient:
                @staticmethod
                def health() -> dict:
                    return {"model_name": "fake-model", "device": "cpu"}

            with mock.patch.object(steer.importlib.util, "find_spec", side_effect=fake_find_spec):
                with mock.patch.object(steer.LocalServerClient, "from_env", return_value=FakeDoctorClient()):
                    with contextlib.redirect_stdout(stdout):
                        result = steer.main(
                            [
                                "--state-path",
                                str(Path(tmp) / "state.json"),
                                "doctor",
                                "--server-url",
                                "http://fake-server",
                            ]
                        )

        self.assertEqual(result, 0)
        output = stdout.getvalue()
        self.assertIn("[ok] backend: fake-model on cpu", output)
        self.assertNotIn("[missing]", output)


if __name__ == "__main__":
    unittest.main()
