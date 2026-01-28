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
    extract_zip,
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

PipelineName = Literal["asr_commands"]


def load_config(config_path: Path) -> dict[str, Any]:
    cfg = load_toml(config_path)

    pipeline_tbl = require_table(cfg, "pipeline", path=config_path)
    paths_tbl = require_table(cfg, "paths", path=config_path)
    download_tbl = require_table(cfg, "download", path=config_path)
    extract_tbl = require_table(cfg, "extract", path=config_path)
    dataset_tbl = require_table(cfg, "dataset", path=config_path)

    pipeline_name = require_str(
        pipeline_tbl, "name", path=config_path, table_name="pipeline")
    if pipeline_name != "asr_commands":
        raise ValueError(
            f"Config pipeline name mismatch: expected 'asr_commands', got '{pipeline_name}'"
        )

    archive_filename = require_str(
        paths_tbl, "archive_filename", path=config_path, table_name="paths")
    raw_dirname = require_str(paths_tbl, "raw_dirname",
                              path=config_path, table_name="paths")
    labels_filename = require_str(
        paths_tbl, "labels_filename", path=config_path, table_name="paths")
    provenance_filename = require_str(
        paths_tbl, "provenance_filename", path=config_path, table_name="paths")

    url = require_str(download_tbl, "url", path=config_path,
                      table_name="download")
    user_agent = require_str(download_tbl, "user_agent",
                             path=config_path, table_name="download")
    timeout_seconds = optional_int(download_tbl, "timeout_seconds")

    expected_bytes_min = optional_int(download_tbl, "expected_bytes_min")
    expected_bytes_max = optional_int(download_tbl, "expected_bytes_max")
    expected_sha256 = optional_str(download_tbl, "expected_sha256")
    expected_md5 = optional_str(download_tbl, "expected_md5")

    sentinel_relpath = require_str(
        extract_tbl, "sentinel_relpath", path=config_path, table_name="extract")
    extracted_root_dirname = require_str(
        extract_tbl, "extracted_root_dirname", path=config_path, table_name="extract"
    )

    labels = optional_list_of_str(dataset_tbl, "labels")
    if not labels:
        raise ValueError(
            "Config must include [dataset].labels as a non-empty list")

    cache_root_str = require_str(
        paths_tbl, "cache_root", path=config_path, table_name="paths")

    return {
        "pipeline": pipeline_name,
        "cache_root": as_path(cache_root_str),
        "archive_filename": archive_filename,
        "raw_dirname": raw_dirname,
        "labels_filename": labels_filename,
        "provenance_filename": provenance_filename,
        "url": url,
        "expected_bytes_min": expected_bytes_min,
        "expected_bytes_max": expected_bytes_max,
        "expected_sha256": expected_sha256,
        "expected_md5": expected_md5,
        "user_agent": user_agent,
        "timeout_seconds": timeout_seconds,
        "sentinel_relpath": sentinel_relpath,
        "extracted_root_dirname": extracted_root_dirname,
        "labels": labels,
    }


def ingest(*, config: dict[str, Any], cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "asr_commands"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / str(config["archive_filename"])
    raw_dir = pipeline_cache / str(config["raw_dirname"])

    url = str(config["url"])

    print(f"[asr_commands] Downloading: {url}")
    print(f"[asr_commands] Cache file: {archive_path}")

    download_url(
        url=url,
        dst=archive_path,
        expected_bytes_min=config.get("expected_bytes_min"),
        expected_bytes_max=config.get("expected_bytes_max"),
        expected_sha256=config.get("expected_sha256"),
        expected_md5=config.get("expected_md5"),
        user_agent=str(config["user_agent"]),
        timeout_seconds=int(config["timeout_seconds"] or 60),
        force=force,
    )

    print(f"[asr_commands] Extracting into: {raw_dir}")
    # Zip contains a 'mini_speech_commands/' root.
    extract_zip(
        archive_path=archive_path,
        dst_dir=raw_dir,
        sentinel_relpath=str(config["sentinel_relpath"]),
    )

    base = raw_dir / str(config["extracted_root_dirname"])
    labels = list(config["labels"])
    missing = [lbl for lbl in labels if not (base / lbl).exists()]
    if missing:
        raise RuntimeError(
            f"mini_speech_commands extraction sanity check failed; missing label directories: {missing}"
        )

    labels_path = pipeline_cache / str(config["labels_filename"])
    write_json(labels_path, {"labels": labels})

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
            dst=str(base / "yes"),
            method="extract",
            bytes=0,
            sha256="(directory)",
        ),
        CachedFile(
            src="(generated) labels.json",
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
        description="Download + cache mini_speech_commands into .cache/asr_commands/"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.toml"),
        help="Path to TOML config file. Default: data_ingestion/asr_commands/config.toml",
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
        help="Force re-download even if cached files exist and pass basic verification.",
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
