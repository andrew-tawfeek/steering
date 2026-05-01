from __future__ import annotations

import unittest
from unittest import mock
from urllib import request

from steering.neuronpedia_client import NeuronpediaClient, NeuronpediaError, state_to_neuronpedia_features
from steering.state import SteerItem, SteeringState


class NeuronpediaClientTests(unittest.TestCase):
    def test_state_to_neuronpedia_features_uses_layer_template(self) -> None:
        state = SteeringState(items=(SteerItem(204, 5, (6, 8)),))

        features = state_to_neuronpedia_features(
            state,
            default_model_id="gpt2-small",
            sae_id_template="{layer}-res-jb",
        )

        self.assertEqual(
            features,
            [
                {"modelId": "gpt2-small", "layer": "6-res-jb", "index": 204, "strength": 5},
                {"modelId": "gpt2-small", "layer": "8-res-jb", "index": 204, "strength": 5},
            ],
        )

    def test_state_to_neuronpedia_features_prefers_explicit_sae_id(self) -> None:
        state = SteeringState(
            items=(SteerItem(650, 3, tuple(), model_id="gpt2-small", sae_id="6-res_scefr-ajt"),)
        )

        features = state_to_neuronpedia_features(
            state,
            default_model_id="ignored",
            sae_id_template="{layer}-res-jb",
        )

        self.assertEqual(
            features,
            [{"modelId": "gpt2-small", "layer": "6-res_scefr-ajt", "index": 650, "strength": 3}],
        )

    def test_client_timeout_can_come_from_environment(self) -> None:
        with mock.patch.dict("os.environ", {"STEERING_NEURONPEDIA_TIMEOUT": "3.5"}):
            client = NeuronpediaClient.from_env("neuronpedia.local")

        self.assertEqual(client.base_url, "https://neuronpedia.local")
        self.assertEqual(client.timeout, 3.5)

    def test_client_timeout_rejects_non_positive_values(self) -> None:
        with mock.patch.dict("os.environ", {"STEERING_NEURONPEDIA_TIMEOUT": "0"}):
            with self.assertRaisesRegex(NeuronpediaError, "greater than 0"):
                NeuronpediaClient.from_env()

    def test_feature_reports_invalid_json_response(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                return None

            @staticmethod
            def read() -> bytes:
                return b"not-json"

        with mock.patch.object(request, "urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(NeuronpediaError, "invalid JSON response"):
                NeuronpediaClient().feature("gpt2-small", "6-res-jb", 204)


if __name__ == "__main__":
    unittest.main()
