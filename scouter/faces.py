"""Face detection (YuNet), face recognition (SFace) and the database of known people."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Cosine similarity above which two SFace embeddings are treated as the same person.
# 0.363 is the threshold the SFace authors give for their LFW benchmark.
MATCH_THRESHOLD = 0.363


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h
    score: float
    raw: np.ndarray  # YuNet row: box, 5 landmarks, score (needed for alignment)
    name: str | None = None  # None = not recognised
    similarity: float = 0.0


class FaceEngine:
    def __init__(self, detector_path: str, recognizer_path: str, min_score: float = 0.8):
        self.detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320), min_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")

    def detect(self, image: np.ndarray) -> list[Face]:
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(image)
        if rows is None:
            return []
        faces = []
        for r in rows:
            x, y, bw, bh = (int(v) for v in r[:4])
            faces.append(Face((x, y, bw, bh), float(r[14]), r))
        return faces

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        """A unit-length 128-d embedding of [face]; similar faces have a high dot product."""
        aligned = self.recognizer.alignCrop(image, face.raw)
        feature = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return feature / (np.linalg.norm(feature) + 1e-9)


class FaceDB:
    """Known people: several embeddings per name, kept in one .npz file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.names: list[str] = []
        self.features = np.zeros((0, 128), np.float32)
        if self.path.exists():
            data = np.load(self.path, allow_pickle=False)
            self.names = [str(n) for n in data["names"]]
            self.features = data["features"].astype(np.float32)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.path, names=np.array(self.names, dtype=str), features=self.features)

    def add(self, name: str, features: list[np.ndarray]) -> None:
        if not features:
            return
        self.names.extend([name] * len(features))
        self.features = np.vstack([self.features, np.stack(features).astype(np.float32)])

    def forget(self, name: str) -> int:
        keep = [i for i, n in enumerate(self.names) if n.lower() != name.lower()]
        removed = len(self.names) - len(keep)
        self.names = [self.names[i] for i in keep]
        self.features = self.features[keep] if keep else np.zeros((0, 128), np.float32)
        return removed

    def people(self) -> dict[str, int]:
        """Name -> number of stored samples."""
        out: dict[str, int] = {}
        for n in self.names:
            out[n] = out.get(n, 0) + 1
        return out

    def match(self, feature: np.ndarray, threshold: float = MATCH_THRESHOLD) -> tuple[str | None, float]:
        """Best matching name and its similarity, or (None, best) when nobody is close enough."""
        if not self.names:
            return None, 0.0
        sims = self.features @ feature
        i = int(np.argmax(sims))
        best = float(sims[i])
        return (self.names[i], best) if best >= threshold else (None, best)


def largest(faces: list[Face]) -> Face | None:
    return max(faces, key=lambda f: f.box[2] * f.box[3], default=None)
