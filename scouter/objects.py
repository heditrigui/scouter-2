"""Object detection with YOLOX (80 COCO classes) through OpenCV's DNN module."""

from dataclasses import dataclass

import cv2
import numpy as np

COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

INPUT_SIZE = 640
STRIDES = (8, 16, 32)


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]  # x, y, w, h in the original image


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Scales [image] to fit a size x size square (top-left aligned, grey padding) as YOLOX expects."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114.0, np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, ratio


def _grids(size: int = INPUT_SIZE) -> tuple[np.ndarray, np.ndarray]:
    grids, strides = [], []
    for s in STRIDES:
        n = size // s
        xs, ys = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        strides.append(np.full((n * n, 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)


GRIDS, GRID_STRIDES = _grids()


def decode(raw: np.ndarray, ratio: float, min_score: float, nms: float) -> list[Detection]:
    """Turns the raw (8400, 85) YOLOX output into boxes in original-image pixels."""
    centers = (raw[:, :2] + GRIDS) * GRID_STRIDES
    sizes = np.exp(raw[:, 2:4]) * GRID_STRIDES
    scores = raw[:, 4:5] * raw[:, 5:]
    classes = scores.argmax(1)
    best = scores[np.arange(len(scores)), classes]

    candidates = np.where(best >= min_score)[0]
    if len(candidates) == 0:
        return []
    xywh = np.concatenate((centers - sizes / 2, sizes), 1)[candidates] / ratio
    keep = cv2.dnn.NMSBoxesBatched(
        xywh.tolist(), best[candidates].tolist(), classes[candidates].tolist(), min_score, nms)
    out = []
    for k in np.array(keep).flatten():
        x, y, w, h = xywh[k]
        i = candidates[k]
        out.append(Detection(COCO[classes[i]], float(best[i]), (int(x), int(y), int(w), int(h))))
    return sorted(out, key=lambda d: -d.score)


class ObjectDetector:
    def __init__(self, model_path: str, min_score: float = 0.5, nms: float = 0.5):
        self.net = cv2.dnn.readNet(model_path)
        self.min_score = min_score
        self.nms = nms

    def detect(self, image: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        padded, ratio = letterbox(rgb)
        self.net.setInput(padded.transpose(2, 0, 1)[np.newaxis])
        raw = self.net.forward()
        return decode(raw.reshape(-1, 85), ratio, self.min_score, self.nms)
