from __future__ import annotations

import unittest

from steering.neuronpedia_client import state_to_neuronpedia_features
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


if __name__ == "__main__":
    unittest.main()
