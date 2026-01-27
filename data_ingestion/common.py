from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from urllib.request import Request, urlopen
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any


@dataclass(frozen=True)
class CachedFile:
    src: str
    dst: str
    method: str
    bytes: int
    sha256: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - used for dataset integrity checks only
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_or_hardlink(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        copy2(src, dst)
        return "copy"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def verify_file(
    *,
    path: Path,
    expected_bytes_min: int | None = None,
    expected_bytes_max: int | None = None,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

    size = path.stat().st_size
    if expected_bytes_min is not None and size < expected_bytes_min:
        raise ValueError(
            f"File too small: {path} ({size} bytes < {expected_bytes_min})")
    if expected_bytes_max is not None and size > expected_bytes_max:
        raise ValueError(
            f"File too large: {path} ({size} bytes > {expected_bytes_max})")

    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"SHA256 mismatch for {path}: {actual} != {expected_sha256}")

    if expected_md5 is not None:
        actual = md5_file(path)
        if actual.lower() != expected_md5.lower():
            raise ValueError(
                f"MD5 mismatch for {path}: {actual} != {expected_md5}")


def download_url(
    *,
    url: str,
    dst: Path,
    expected_bytes_min: int | None = None,
    expected_bytes_max: int | None = None,
    expected_sha256: str | None = None,
    expected_md5: str | None = None,
    user_agent: str = "pjatk_zum-ingestion/1.0",
    timeout_seconds: int = 60,
    force: bool = False,
) -> Path:
    """Download a URL to dst atomically and verify basic integrity.

    Idempotent behavior:
    - If dst exists and passes verification, it is reused.
    - If dst exists but fails verification (or force=True), it is re-downloaded.
    """
    ensure_dir(dst.parent)

    if dst.exists() and not force:
        try:
            verify_file(
                path=dst,
                expected_bytes_min=expected_bytes_min,
                expected_bytes_max=expected_bytes_max,
                expected_sha256=expected_sha256,
                expected_md5=expected_md5,
            )
            return dst
        except Exception:
            dst.unlink()

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout_seconds) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)

    tmp.replace(dst)
    verify_file(
        path=dst,
        expected_bytes_min=expected_bytes_min,
        expected_bytes_max=expected_bytes_max,
        expected_sha256=expected_sha256,
        expected_md5=expected_md5,
    )
    return dst


def extract_tar_gz(*, archive_path: Path, dst_dir: Path, sentinel_relpath: str | None = None) -> Path:
    """Extract a .tar.gz into dst_dir; reuse existing extraction if sentinel exists."""
    ensure_dir(dst_dir)
    if sentinel_relpath is not None:
        sentinel = dst_dir / sentinel_relpath
        if sentinel.exists():
            return dst_dir

    with tarfile.open(archive_path, mode="r:gz") as tf:
        tf.extractall(dst_dir)
    return dst_dir


def extract_zip(*, archive_path: Path, dst_dir: Path, sentinel_relpath: str | None = None) -> Path:
    """Extract a .zip into dst_dir; reuse existing extraction if sentinel exists."""
    ensure_dir(dst_dir)
    if sentinel_relpath is not None:
        sentinel = dst_dir / sentinel_relpath
        if sentinel.exists():
            return dst_dir

    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dst_dir)
    return dst_dir


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(
        data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cached_file_record(*, src: Path, dst: Path, method: str) -> CachedFile:
    return CachedFile(
        src=str(src),
        dst=str(dst),
        method=method,
        bytes=dst.stat().st_size,
        sha256=sha256_file(dst),
    )


def write_provenance(*, pipeline: str, cache_root: Path, files: list[CachedFile], out_path: Path) -> None:
    provenance: dict[str, Any] = {
        "pipeline": pipeline,
        "created_at": utc_now_iso(),
        "cache_root": str(cache_root),
        "files": [asdict(f) for f in files],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(provenance, indent=2,
                        sort_keys=True) + "\n", encoding="utf-8")
