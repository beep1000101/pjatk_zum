from pathlib import Path
from typing import Final

ROOT_PATH: Final[Path] = Path(__file__).parents[1]

CACHE_PATH: Final[Path] = ROOT_PATH / ".cache"

# Add outputs path for clip_multimodal
CLIP_OUTPUTS_PATH: Final[Path] = ROOT_PATH / "outputs/clip_multimodal"
