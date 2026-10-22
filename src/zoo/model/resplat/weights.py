"""Pretrained ReSplat checkpoint management: HF download + verified cache.

Checkpoints follow the upstream MODEL_ZOO convention: the 8-hex sha256
prefix is embedded in the file name and verified on every cache hit — a
partial or corrupted download is re-fetched automatically.

Downloads try hf-mirror.com first (HuggingFace direct stalls on this
network) with resume (`curl -C -`) and bounded retries.
"""

import hashlib
import subprocess
from pathlib import Path

_MODELS_DIR = Path(__file__).parents[4] / "models" / "resplat"

# name -> (sha256 prefix from MODEL_ZOO.md, huggingface repo path)
CHECKPOINTS = {
    "resplat-base-dl3dv-512x960-view8-8179ed87.pth": (
        "8179ed87",
        "haofeixu/resplat/resolve/main/resplat-base-dl3dv-512x960-view8-8179ed87.pth",
    ),
}

DEFAULT_CHECKPOINT = "resplat-base-dl3dv-512x960-view8-8179ed87.pth"

_MIRRORS = (
    "https://hf-mirror.com/{repo}",
    "https://huggingface.co/{repo}",
)
_MAX_ATTEMPTS = 8


def _sha256_prefix(path: Path, length: int = 8) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def ensure_checkpoint(name: str = DEFAULT_CHECKPOINT) -> Path:
    """Return a verified checkpoint path, downloading it if needed.

    Raises RuntimeError when the download cannot be completed or the
    checksum does not match after retrying.
    """
    prefix, repo = CHECKPOINTS[name]
    path = _MODELS_DIR / name
    if path.exists() and _sha256_prefix(path) == prefix:
        return path

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[resplat.weights] fetching {name} (sha256 prefix {prefix})...")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        for mirror in _MIRRORS:
            url = mirror.format(repo=repo)
            subprocess.run(
                [
                    "curl",
                    "-sL",
                    "-C",
                    "-",
                    "-o",
                    str(path),
                    "--max-time",
                    "900",
                    "--speed-time",
                    "45",
                    "--speed-limit",
                    "20480",
                    url,
                ],
                check=False,
            )
            if not path.exists():
                continue
            if _sha256_prefix(path) == prefix:
                print(f"[resplat.weights] cached at {path}")
                return path
        size = path.stat().st_size if path.exists() else 0
        print(
            f"[resplat.weights] attempt {attempt}/{_MAX_ATTEMPTS}: "
            f"{size / 1e6:.0f} MB so far, checksum not met — resuming"
        )
    raise RuntimeError(
        f"could not fetch a checkpoint matching sha256 prefix {prefix} "
        f"for {name} after {_MAX_ATTEMPTS} attempts"
    )
