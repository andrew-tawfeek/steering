from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from steering.feature_cache import FeatureCache, FeatureLabel, NeuronpediaDatasetClient, sort_export_keys


class FakeDatasetClient(NeuronpediaDatasetClient):
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def _list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.files if key.startswith(prefix)]

    def _read_bytes(self, path: str) -> bytes:
        return self.files[path.lstrip("/")]


class FakeListingClient(NeuronpediaDatasetClient):
    def _list_common_prefixes(self, prefix: str) -> list[str]:
        if prefix == "v1/":
            return ["v1/config/", "v1/gpt2-small/", "v1/gemma-2-2b/"]
        return []


def gzip_jsonl(rows: list[dict]) -> bytes:
    payload = "\n".join(json.dumps(row) for row in rows).encode("utf-8")
    return gzip.compress(payload)


class FeatureCacheTests(unittest.TestCase):
    def test_list_models_excludes_export_config_directory(self) -> None:
        client = FakeListingClient()

        self.assertEqual(client.list_models(), ["gemma-2-2b", "gpt2-small"])

    def test_download_source_labels_parses_explanations_export(self) -> None:
        key = "v1/gpt2-small/6-res-jb/explanations/batch-0.jsonl.gz"
        client = FakeDatasetClient(
            {
                key: gzip_jsonl(
                    [
                        {
                            "modelId": "gpt2-small",
                            "layer": "6-res-jb",
                            "index": "204",
                            "description": "time-related phrases",
                            "typeName": "oai_token-act-pair",
                            "explanationModelName": "gpt-4o-mini",
                            "embedding": "[0.0, 1.0]",
                        }
                    ]
                )
            }
        )

        labels = client.download_source_labels("gpt2-small", "6-res-jb")

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].feature_id, 204)
        self.assertEqual(labels[0].description, "time-related phrases")
        self.assertEqual(labels[0].type_name, "oai_token-act-pair")

    def test_download_source_labels_uses_natural_batch_order_for_limits(self) -> None:
        keys = [
            "v1/gpt2-small/6-res-jb/explanations/batch-10.jsonl.gz",
            "v1/gpt2-small/6-res-jb/explanations/batch-2.jsonl.gz",
            "v1/gpt2-small/6-res-jb/explanations/batch-1.jsonl.gz",
        ]

        self.assertEqual(
            sort_export_keys(keys),
            [
                "v1/gpt2-small/6-res-jb/explanations/batch-1.jsonl.gz",
                "v1/gpt2-small/6-res-jb/explanations/batch-2.jsonl.gz",
                "v1/gpt2-small/6-res-jb/explanations/batch-10.jsonl.gz",
            ],
        )

    def test_cache_search_and_get(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FeatureCache(Path(tmp) / "features.sqlite3")
            cache.replace_source(
                "gpt2-small",
                "6-res-jb",
                [
                    FeatureLabel(
                        "gpt2-small",
                        "6-res-jb",
                        204,
                        "time-related phrases and expressions",
                        explanation_model_name="gpt-4o-mini",
                    ),
                    FeatureLabel("gpt2-small", "6-res-jb", 1, "grant funding references"),
                ],
            )

            results = cache.search("time phrases", model_id="gpt2-small", source_id="6-res-jb")
            self.assertEqual([result.feature_id for result in results], [204])

            labels = cache.get(model_id="gpt2-small", source_id="6-res-jb", feature_id=1)
            self.assertEqual(labels[0].description, "grant funding references")

    def test_cache_search_can_filter_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FeatureCache(Path(tmp) / "features.sqlite3")
            cache.replace_source(
                "gpt2-small",
                "6-res-jb",
                [FeatureLabel("gpt2-small", "6-res-jb", 204, "time-related phrases")],
            )
            cache.replace_source(
                "gpt2-small",
                "8-res-jb",
                [FeatureLabel("gpt2-small", "8-res-jb", 205, "time-related phrases")],
            )
            cache.replace_source(
                "gpt2-small",
                "mlp-out",
                [FeatureLabel("gpt2-small", "mlp-out", 206, "time-related phrases")],
            )

            results = cache.search("time phrases", model_id="gpt2-small", source_ids=("6-res-jb", "8-res-jb"))

            self.assertEqual([(result.source_id, result.feature_id) for result in results], [
                ("6-res-jb", 204),
                ("8-res-jb", 205),
            ])


if __name__ == "__main__":
    unittest.main()
