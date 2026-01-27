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
    write_provenance,
)

PipelineName = Literal["sentiment_embeddings"]

# Stanford Large Movie Review Dataset (IMDB)
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
# Widely published checksum for the official archive.
IMDB_MD5 = "7c2ac02c03563afcf9b574c7e56c153a"


def ingest(*, cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "sentiment_embeddings"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / "aclImdb_v1.tar.gz"
    raw_dir = pipeline_cache / "raw"

    print(f"[sentiment_embeddings] Downloading: {IMDB_URL}")
    print(f"[sentiment_embeddings] Cache file: {archive_path}")

    download_url(
        url=IMDB_URL,
        dst=archive_path,
        expected_md5=IMDB_MD5,
        expected_bytes_min=50 * 1024 * 1024,
        expected_bytes_max=500 * 1024 * 1024,
        force=force,
    )

    print(f"[sentiment_embeddings] Extracting into: {raw_dir}")
    # Archive contains 'aclImdb/' folder.
    extract_tar_gz(archive_path=archive_path, dst_dir=raw_dir,
                   sentinel_relpath="aclImdb/README")

    # Basic sanity checks (expected files)
    sentinel = raw_dir / "aclImdb/README"
    train_dir = raw_dir / "aclImdb/train"
    test_dir = raw_dir / "aclImdb/test"
    if not sentinel.exists() or not train_dir.exists() or not test_dir.exists():
        raise RuntimeError(
            "IMDB extraction sanity check failed; expected aclImdb/README, aclImdb/train, aclImdb/test under "
            f"{raw_dir}"
        )

    provenance_path = pipeline_cache / "provenance.json"
    files: list[CachedFile] = [
        CachedFile(
            src=IMDB_URL,
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
