from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import io
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable
from urllib import error, parse, request
import xml.etree.ElementTree as ET


DATASET_BASE_URL = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
DATASET_VERSION = "v1"
DEFAULT_CACHE_PATH = ".steering/feature-cache.sqlite3"


class FeatureCacheError(RuntimeError):
    """Raised when feature cache operations fail."""


@dataclass(frozen=True)
class CachedSource:
    model_id: str
    source_id: str
    label_count: int
    feature_count: int
    fetched_at: str


@dataclass(frozen=True)
class FeatureLabel:
    model_id: str
    source_id: str
    feature_id: int
    description: str
    type_name: str | None = None
    explanation_model_name: str | None = None


def default_feature_cache_path(cwd: Path | None = None) -> Path:
    env_path = os.environ.get("STEERING_FEATURE_CACHE_PATH")
    if env_path:
        return Path(env_path).expanduser()
    base = cwd if cwd is not None else Path.cwd()
    return base / DEFAULT_CACHE_PATH


class FeatureCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_feature_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS labels (
                    model_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    feature_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    type_name TEXT NOT NULL DEFAULT '',
                    explanation_model_name TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (
                        model_id,
                        source_id,
                        feature_id,
                        description,
                        type_name,
                        explanation_model_name
                    )
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cached_sources (
                    model_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    label_count INTEGER NOT NULL,
                    feature_count INTEGER NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (model_id, source_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_feature ON labels(model_id, source_id, feature_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_model_source ON labels(model_id, source_id)")

    def replace_source(self, model_id: str, source_id: str, labels: Iterable[FeatureLabel]) -> CachedSource:
        if not model_id.strip():
            raise FeatureCacheError("model_id cannot be blank")
        if not source_id.strip():
            raise FeatureCacheError("source_id cannot be blank")

        now = now_iso()
        rows = [
            (
                model_id,
                source_id,
                label.feature_id,
                label.description,
                label.type_name or "",
                label.explanation_model_name or "",
                now,
            )
            for label in labels
            if label.description.strip()
        ]
        feature_count = len({row[2] for row in rows})
        with self._connect() as conn:
            conn.execute("DELETE FROM labels WHERE model_id = ? AND source_id = ?", (model_id, source_id))
            conn.executemany(
                """
                INSERT OR IGNORE INTO labels (
                    model_id,
                    source_id,
                    feature_id,
                    description,
                    type_name,
                    explanation_model_name,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.execute(
                """
                INSERT INTO cached_sources (model_id, source_id, label_count, feature_count, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_id, source_id) DO UPDATE SET
                    label_count = excluded.label_count,
                    feature_count = excluded.feature_count,
                    fetched_at = excluded.fetched_at
                """,
                (model_id, source_id, len(rows), feature_count, now),
            )
        return CachedSource(model_id, source_id, len(rows), feature_count, now)

    def status(self) -> list[CachedSource]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_id, source_id, label_count, feature_count, fetched_at
                FROM cached_sources
                ORDER BY model_id, source_id
                """
            ).fetchall()
        return [
            CachedSource(
                model_id=row["model_id"],
                source_id=row["source_id"],
                label_count=int(row["label_count"]),
                feature_count=int(row["feature_count"]),
                fetched_at=row["fetched_at"],
            )
            for row in rows
        ]

    def search(
        self,
        query: str,
        *,
        model_id: str | None = None,
        source_id: str | None = None,
        limit: int = 20,
    ) -> list[FeatureLabel]:
        if limit < 1:
            raise FeatureCacheError("limit must be >= 1")
        terms = [term.casefold() for term in query.split() if term.strip()]
        if not terms:
            raise FeatureCacheError("search query cannot be empty")

        where = []
        params: list[Any] = []
        if model_id:
            where.append("model_id = ?")
            params.append(model_id)
        if source_id:
            where.append("source_id = ?")
            params.append(source_id)
        for term in terms:
            where.append("LOWER(description) LIKE ?")
            params.append(f"%{term}%")
        sql = """
            SELECT model_id, source_id, feature_id, description, type_name, explanation_model_name
            FROM labels
        """
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY model_id, source_id, feature_id LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [row_to_label(row) for row in rows]

    def get(
        self,
        *,
        model_id: str,
        source_id: str,
        feature_id: int,
    ) -> list[FeatureLabel]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT model_id, source_id, feature_id, description, type_name, explanation_model_name
                FROM labels
                WHERE model_id = ? AND source_id = ? AND feature_id = ?
                ORDER BY explanation_model_name, type_name, description
                """,
                (model_id, source_id, feature_id),
            ).fetchall()
        return [row_to_label(row) for row in rows]


class NeuronpediaDatasetClient:
    def __init__(self, base_url: str = DATASET_BASE_URL, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        prefixes = self._list_common_prefixes(f"{DATASET_VERSION}/")
        return sorted(
            model_id
            for model_id in (prefix.split("/")[1] for prefix in prefixes if prefix.count("/") >= 2)
            if model_id != "config"
        )

    def list_sources(self, model_id: str) -> list[str]:
        prefixes = self._list_common_prefixes(f"{DATASET_VERSION}/{model_id}/")
        return sorted(prefix.split("/")[2] for prefix in prefixes if prefix.count("/") >= 3)

    def download_source_labels(
        self,
        model_id: str,
        source_id: str,
        *,
        max_files: int | None = None,
    ) -> list[FeatureLabel]:
        if max_files is not None and max_files < 1:
            raise FeatureCacheError("max_files must be >= 1")

        prefix = f"{DATASET_VERSION}/{model_id}/{source_id}/explanations/"
        keys = sort_export_keys(key for key in self._list_keys(prefix) if key.endswith(".jsonl.gz"))
        if max_files is not None:
            keys = keys[:max_files]
        if not keys:
            raise FeatureCacheError(f"no explanation export files found for {model_id}/{source_id}")

        labels: list[FeatureLabel] = []
        for key in keys:
            labels.extend(self._download_label_file(key, model_id, source_id))
        return labels

    def _download_label_file(self, key: str, model_id: str, source_id: str) -> list[FeatureLabel]:
        data = self._read_bytes(f"/{key}")
        labels: list[FeatureLabel] = []
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
            for raw_line in gz:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                row = json.loads(line)
                description = str(row.get("description") or "").strip()
                if not description:
                    continue
                try:
                    feature_id = int(row["index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise FeatureCacheError(f"invalid explanation row in {key}: {row!r}") from exc
                labels.append(
                    FeatureLabel(
                        model_id=str(row.get("modelId") or model_id),
                        source_id=str(row.get("layer") or source_id),
                        feature_id=feature_id,
                        description=description,
                        type_name=optional_str(row.get("typeName")),
                        explanation_model_name=optional_str(row.get("explanationModelName")),
                    )
                )
        return labels

    def _list_common_prefixes(self, prefix: str) -> list[str]:
        return self._list_bucket(prefix, delimiter="/")[1]

    def _list_keys(self, prefix: str) -> list[str]:
        return self._list_bucket(prefix, delimiter=None)[0]

    def _list_bucket(self, prefix: str, delimiter: str | None) -> tuple[list[str], list[str]]:
        continuation: str | None = None
        keys: list[str] = []
        prefixes: list[str] = []
        while True:
            query: dict[str, str] = {"list-type": "2", "prefix": prefix}
            if delimiter is not None:
                query["delimiter"] = delimiter
            if continuation is not None:
                query["continuation-token"] = continuation
            path = "/?" + parse.urlencode(query)
            xml = self._read_text(path)
            root = ET.fromstring(xml)
            namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
            keys.extend(node.text or "" for node in root.findall("s3:Contents/s3:Key", namespace))
            prefixes.extend(node.text or "" for node in root.findall("s3:CommonPrefixes/s3:Prefix", namespace))
            is_truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=namespace) or "").lower()
            if is_truncated != "true":
                break
            continuation = root.findtext("s3:NextContinuationToken", namespaces=namespace)
            if not continuation:
                break
        return keys, prefixes

    def _read_text(self, path: str) -> str:
        return self._read_bytes(path).decode("utf-8")

    def _read_bytes(self, path: str) -> bytes:
        req = request.Request(f"{self.base_url}{path}", headers={"Accept": "*/*"})
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FeatureCacheError(f"dataset export returned HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise FeatureCacheError(f"could not reach dataset export at {self.base_url}") from exc


def build_source_cache(
    *,
    model_id: str,
    source_id: str,
    cache_path: Path | None = None,
    dataset_client: NeuronpediaDatasetClient | None = None,
    max_files: int | None = None,
) -> CachedSource:
    client = dataset_client or NeuronpediaDatasetClient()
    labels = client.download_source_labels(model_id, source_id, max_files=max_files)
    return FeatureCache(cache_path).replace_source(model_id, source_id, labels)


def row_to_label(row: sqlite3.Row) -> FeatureLabel:
    return FeatureLabel(
        model_id=row["model_id"],
        source_id=row["source_id"],
        feature_id=int(row["feature_id"]),
        description=row["description"],
        type_name=row["type_name"] or None,
        explanation_model_name=row["explanation_model_name"] or None,
    )


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def sort_export_keys(keys: Iterable[str]) -> list[str]:
    def key_parts(key: str) -> tuple[str, int]:
        match = re.search(r"/batch-(\d+)\.jsonl\.gz$", key)
        if not match:
            return key, -1
        return key[: match.start()], int(match.group(1))

    return sorted(keys, key=key_parts)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
