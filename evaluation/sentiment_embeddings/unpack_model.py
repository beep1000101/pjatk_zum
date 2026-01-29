"""
Unpack archived sentiment embeddings model to .cache/sentiment_embeddings for evaluation.
"""
from pathlib import Path
import shutil


def unpack_model(archive_path, unpack_dir):
    unpack_dir = Path(unpack_dir)
    unpack_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == '.zip':
        import zipfile
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(unpack_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tar.gz', '.tgz']:
        import tarfile
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(unpack_dir)
    else:
        # Assume it's a directory, just copy
        if unpack_dir.exists():
            shutil.rmtree(unpack_dir)
        shutil.copytree(archive_path, unpack_dir)
    return unpack_dir
