from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_ingestion.common import (  # noqa: E402
    CachedFile,
    download_url,
    extract_tar_gz,
    sha256_file,
    write_provenance,
)
from data_ingestion.config_utils import (  # noqa: E402
    as_path,
    load_toml,
    optional_int,
    optional_list_of_str,
    optional_str,
    require_str,
    require_table,
 )

PipelineName = Literal["sentiment_embeddings"]


def load_config(config_path: Path) -> dict[str, Any]:
    cfg = load_toml(config_path)

    pipeline_tbl = require_table(cfg, "pipeline", path=config_path)
    paths_tbl = require_table(cfg, "paths", path=config_path)
    download_tbl = require_table(cfg, "download", path=config_path)
    extract_tbl = require_table(cfg, "extract", path=config_path)

    pipeline_name = require_str(
        pipeline_tbl, "name", path=config_path, table_name="pipeline")
    if pipeline_name != "sentiment_embeddings":
        raise ValueError(
            f"Config pipeline name mismatch: expected 'sentiment_embeddings', got '{pipeline_name}'"
        )

    cache_root_str = require_str(
        paths_tbl, "cache_root", path=config_path, table_name="paths")
    archive_filename = require_str(
        paths_tbl, "archive_filename", path=config_path, table_name="paths")
    raw_dirname = require_str(paths_tbl, "raw_dirname",
                              path=config_path, table_name="paths")
    provenance_filename = require_str(
        paths_tbl, "provenance_filename", path=config_path, table_name="paths"
    )

    url = require_str(download_tbl, "url", path=config_path,
                      table_name="download")
    expected_md5 = optional_str(download_tbl, "expected_md5")
    expected_sha256 = optional_str(download_tbl, "expected_sha256")
    expected_bytes_min = optional_int(download_tbl, "expected_bytes_min")
    expected_bytes_max = optional_int(download_tbl, "expected_bytes_max")
    user_agent = require_str(download_tbl, "user_agent",
                             path=config_path, table_name="download")
    timeout_seconds = optional_int(download_tbl, "timeout_seconds")

    sentinel_relpath = require_str(
        extract_tbl, "sentinel_relpath", path=config_path, table_name="extract")
    expected_dirs = optional_list_of_str(extract_tbl, "expected_dirs")
    if not expected_dirs:
        raise ValueError(
            "Config must include [extract].expected_dirs as a non-empty list")

    return {
        "pipeline": pipeline_name,
        "cache_root": as_path(cache_root_str),
        "archive_filename": archive_filename,
        "raw_dirname": raw_dirname,
        "provenance_filename": provenance_filename,
        "url": url,
        "expected_md5": expected_md5,
        "expected_sha256": expected_sha256,
        "expected_bytes_min": expected_bytes_min,
        "expected_bytes_max": expected_bytes_max,
        "user_agent": user_agent,
        "timeout_seconds": timeout_seconds,
        "sentinel_relpath": sentinel_relpath,
        "expected_dirs": expected_dirs,
    }


def ingest(*, config: dict[str, Any], cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "sentiment_embeddings"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / str(config["archive_filename"])
    raw_dir = pipeline_cache / str(config["raw_dirname"])

    url = str(config["url"])

    print(f"[sentiment_embeddings] Downloading: {url}")
    print(f"[sentiment_embeddings] Cache file: {archive_path}")

    download_url(
        url=url,
        dst=archive_path,
        expected_md5=config.get("expected_md5"),
        expected_sha256=config.get("expected_sha256"),
        expected_bytes_min=config.get("expected_bytes_min"),
        expected_bytes_max=config.get("expected_bytes_max"),
        user_agent=str(config["user_agent"]),
        timeout_seconds=int(config["timeout_seconds"] or 60),
        force=force,
    )

    print(f"[sentiment_embeddings] Extracting into: {raw_dir}")
    # Archive contains 'aclImdb/' folder.
    extract_tar_gz(
        archive_path=archive_path,
        dst_dir=raw_dir,
        sentinel_relpath=str(config["sentinel_relpath"]),
    )

    # Basic sanity checks (expected files)
    sentinel = raw_dir / str(config["sentinel_relpath"])
    expected_dirs = [raw_dir / rel for rel in list(config["expected_dirs"])]
    if not sentinel.exists() or any(not d.exists() for d in expected_dirs):
        raise RuntimeError(
            "IMDB extraction sanity check failed; expected aclImdb/README, aclImdb/train, aclImdb/test under "
            f"{raw_dir}"
        )

    provenance_path = pipeline_cache / str(config["provenance_filename"])
    files: list[CachedFile] = [
        CachedFile(
            src=url,
            dst=str(archive_path),
            method="download",
            bytes=archive_path.stat().st_size,
            sha256=sha256_file(archive_path),
        ),
        CachedFile(
            src=str(archive_path),
            dst=str(sentinel),
            method="extract",
            bytes=sentinel.stat().st_size,
            sha256=sha256_file(sentinel),
        ),
    ]
    write_provenance(pipeline=pipeline, cache_root=cache_root,
                     files=files, out_path=provenance_path)
    return provenance_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download + cache IMDB sentiment dataset into .cache/sentiment_embeddings/"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
        help="Path to TOML config file. Default: data_ingestion/sentiment_embeddings/config.toml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Override cache root directory (otherwise from config).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached files exist and pass verification.",
    )
    args = parser.parse_args()

    config_path: Path = args.config
    config = load_config(config_path)
    cache_root = args.cache_root if args.cache_root is not None else Path(
        config["cache_root"])

    provenance_path = ingest(
        config=config, cache_root=cache_root, force=args.force)
    print(f"Wrote provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    main()
