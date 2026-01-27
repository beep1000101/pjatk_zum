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
    extract_zip,
    sha256_file,
    write_json,
    write_provenance,
)

PipelineName = Literal["asr_commands"]

# TensorFlow official tutorial dataset (small subset of Speech Commands)
MINI_SPEECH_COMMANDS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"

# Expected label folders in the mini dataset.
COMMAND_LABELS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]


def ingest(*, cache_root: Path, force: bool = False) -> Path:
    pipeline: PipelineName = "asr_commands"
    pipeline_cache = cache_root / pipeline
    pipeline_cache.mkdir(parents=True, exist_ok=True)

    archive_path = pipeline_cache / "mini_speech_commands.zip"
    raw_dir = pipeline_cache / "raw"

    print(f"[asr_commands] Downloading: {MINI_SPEECH_COMMANDS_URL}")
    print(f"[asr_commands] Cache file: {archive_path}")

    download_url(
        url=MINI_SPEECH_COMMANDS_URL,
        dst=archive_path,
        expected_bytes_min=10 * 1024 * 1024,
        expected_bytes_max=500 * 1024 * 1024,
        force=force,
    )

    print(f"[asr_commands] Extracting into: {raw_dir}")
    # Zip contains a 'mini_speech_commands/' root.
    extract_zip(archive_path=archive_path, dst_dir=raw_dir,
                sentinel_relpath="mini_speech_commands/yes")

    base = raw_dir / "mini_speech_commands"
    missing = [lbl for lbl in COMMAND_LABELS if not (base / lbl).exists()]
    if missing:
        raise RuntimeError(
            f"mini_speech_commands extraction sanity check failed; missing label directories: {missing}"
        )

    labels_path = pipeline_cache / "labels.json"
    write_json(labels_path, {"labels": COMMAND_LABELS})

    provenance_path = pipeline_cache / "provenance.json"
    files: list[CachedFile] = [
        CachedFile(
            src=MINI_SPEECH_COMMANDS_URL,
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
        "--cache-root",
        type=Path,
        default=Path(".cache"),
        help="Cache root directory (single source of truth for raw data). Default: .cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached files exist and pass basic verification.",
    )
    args = parser.parse_args()

    provenance_path = ingest(cache_root=args.cache_root, force=args.force)
    print(f"Wrote provenance: {provenance_path}")
    return 0


if __name__ == "__main__":
    main()
