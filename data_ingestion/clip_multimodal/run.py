from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

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

PipelineName = Literal["clip_multimodal"]

# CIFAR-10 (python version), official host (Toronto)
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"

# Canonical CIFAR-10 class names (used as text prompts for CLIP retrieval)
CIFAR10_LABEL_TEXTS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def ingest(*, cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "clip_multimodal"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / "cifar-10-python.tar.gz"
    raw_dir = pipeline_cache / "raw"

    print(f"[clip_multimodal] Downloading: {CIFAR10_URL}")
    print(f"[clip_multimodal] Cache file: {archive_path}")

    download_url(
        url=CIFAR10_URL,
        dst=archive_path,
        expected_md5=CIFAR10_MD5,
        expected_bytes_min=100 * 1024 * 1024,
        expected_bytes_max=300 * 1024 * 1024,
        force=force,
    )

    print(f"[clip_multimodal] Extracting into: {raw_dir}")
    extract_tar_gz(
        archive_path=archive_path,
        dst_dir=raw_dir,
        sentinel_relpath="cifar-10-batches-py/batches.meta",
    )

    # Basic sanity checks (expected files)
    base = raw_dir / "cifar-10-batches-py"
    expected = [
        base / "batches.meta",
        base / "test_batch",
        base / "data_batch_1",
        base / "data_batch_2",
        base / "data_batch_3",
        base / "data_batch_4",
        base / "data_batch_5",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(
            f"CIFAR-10 extraction sanity check failed; missing files: {missing}")

    # Cache a deterministic mapping from label id -> text prompt.
    labels_path = pipeline_cache / "label_texts.json"
    write_json(labels_path, {"labels": CIFAR10_LABEL_TEXTS})

    provenance_path = pipeline_cache / "provenance.json"
    files: list[CachedFile] = [
        CachedFile(
            src=CIFAR10_URL,
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
        "--cache-root",
        type=Path,
        default=Path(".cache"),
        help="Cache root directory (single source of truth for raw data). Default: .cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached files exist and pass verification.",
    )
    args = parser.parse_args()

    provenance_path = ingest(cache_root=args.cache_root, force=args.force)
    print(f"Wrote provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    main()
