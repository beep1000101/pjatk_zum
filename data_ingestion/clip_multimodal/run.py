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
    write_json,
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

PipelineName = Literal["clip_multimodal"]


def load_config(config_path: Path) -> dict[str, Any]:
    cfg = load_toml(config_path)

    pipeline_tbl = require_table(cfg, "pipeline", path=config_path)
    paths_tbl = require_table(cfg, "paths", path=config_path)
    download_tbl = require_table(cfg, "download", path=config_path)
    extract_tbl = require_table(cfg, "extract", path=config_path)
    dataset_tbl = require_table(cfg, "dataset", path=config_path)

    pipeline_name = require_str(
        pipeline_tbl, "name", path=config_path, table_name="pipeline")
    if pipeline_name != "clip_multimodal":
        raise ValueError(
            f"Config pipeline name mismatch: expected 'clip_multimodal', got '{pipeline_name}'"
        )

    cache_root_str = require_str(
        paths_tbl, "cache_root", path=config_path, table_name="paths")
    archive_filename = require_str(
        paths_tbl, "archive_filename", path=config_path, table_name="paths")
    raw_dirname = require_str(paths_tbl, "raw_dirname",
                              path=config_path, table_name="paths")
    label_texts_filename = require_str(
        paths_tbl, "label_texts_filename", path=config_path, table_name="paths"
    )
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
    extracted_root_dirname = require_str(
        extract_tbl, "extracted_root_dirname", path=config_path, table_name="extract"
    )
    expected_files = optional_list_of_str(extract_tbl, "expected_files")
    if not expected_files:
        raise ValueError(
            "Config must include [extract].expected_files as a non-empty list")

    label_texts = optional_list_of_str(dataset_tbl, "label_texts")
    if not label_texts:
        raise ValueError(
            "Config must include [dataset].label_texts as a non-empty list")

    return {
        "pipeline": pipeline_name,
        "cache_root": as_path(cache_root_str),
        "archive_filename": archive_filename,
        "raw_dirname": raw_dirname,
        "label_texts_filename": label_texts_filename,
        "provenance_filename": provenance_filename,
        "url": url,
        "expected_md5": expected_md5,
        "expected_sha256": expected_sha256,
        "expected_bytes_min": expected_bytes_min,
        "expected_bytes_max": expected_bytes_max,
        "user_agent": user_agent,
        "timeout_seconds": timeout_seconds,
        "sentinel_relpath": sentinel_relpath,
        "extracted_root_dirname": extracted_root_dirname,
        "expected_files": expected_files,
        "label_texts": label_texts,
    }


def ingest(*, config: dict[str, Any], cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "clip_multimodal"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / str(config["archive_filename"])
    raw_dir = pipeline_cache / str(config["raw_dirname"])

    url = str(config["url"])

    print(f"[clip_multimodal] Downloading: {url}")
    print(f"[clip_multimodal] Cache file: {archive_path}")

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

    print(f"[clip_multimodal] Extracting into: {raw_dir}")
    extract_tar_gz(
        archive_path=archive_path,
        dst_dir=raw_dir,
        sentinel_relpath=str(config["sentinel_relpath"]),
    )

    # Basic sanity checks (expected files)
    base = raw_dir / str(config["extracted_root_dirname"])
    expected = [base / rel for rel in list(config["expected_files"])]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"CIFAR-10 extraction sanity check failed; missing files: {missing}")

    # Cache a deterministic mapping from label id -> text prompt.
    labels_path = pipeline_cache / str(config["label_texts_filename"])
    write_json(labels_path, {"labels": list(config["label_texts"])})

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
            dst=str(base / "batches.meta"),
            method="extract",
            bytes=(base / "batches.meta").stat().st_size,
            sha256=sha256_file(base / "batches.meta"),
        ),
        CachedFile(
            src="(generated) label_texts.json",
            dst=str(labels_path),
            method="generated",
            bytes=labels_path.stat().st_size,
            sha256=sha256_file(labels_path),
        ),
    ]
    write_provenance(pipeline=pipeline, cache_root=cache_root,
                     files=files, out_path=provenance_path)
    return provenance_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download + cache CIFAR-10 (for CLIP multimodal pipeline) into .cache/clip_multimodal/"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
        help="Path to TOML config file. Default: data_ingestion/clip_multimodal/config.toml",
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
