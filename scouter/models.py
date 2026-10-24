"""Downloads the ONNX models from the OpenCV model zoo on first use and checks them."""

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str
    size_mb: float


FACE_DETECTOR = Model(
    "face_detection_yunet_2023mar.onnx",
    ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    0.2,
)
FACE_RECOGNIZER = Model(
    "face_recognition_sface_2021dec.onnx",
    ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    38.7,
)
OBJECT_DETECTOR = Model(
    "object_detection_yolox_2022nov.onnx",
    ZOO + "object_detection_yolox/object_detection_yolox_2022nov.onnx",
    "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    35.9,
)
ALL = (FACE_DETECTOR, FACE_RECOGNIZER, OBJECT_DETECTOR)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(model: Model) -> str:
    """Returns the local path of [model], downloading it first if needed."""
    path = MODELS_DIR / model.filename
    if path.exists():
        return str(path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"Downloading {model.filename} ({model.size_mb:.1f} MB)...", file=sys.stderr)

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            print(f"\r  {pct:3d}%", end="", file=sys.stderr)

    urllib.request.urlretrieve(model.url, tmp, progress)
    print(file=sys.stderr)
    digest = _sha256(tmp)
    if digest != model.sha256:
        tmp.unlink()
        raise RuntimeError(f"{model.filename}: checksum mismatch (got {digest}). Try again later.")
    tmp.replace(path)
    return str(path)
