"""Scouter-style overlay: corner brackets around what it sees, a status bar and the last sentence spoken."""

import cv2
import numpy as np

GREEN = (90, 255, 120)
AMBER = (40, 190, 255)
CYAN = (255, 220, 90)
WHITE = (240, 240, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def brackets(img, box, color, thickness=2):
    x, y, w, h = box
    k = max(8, min(w, h) // 5)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * k, py), color, thickness, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * k), color, thickness, cv2.LINE_AA)


def tag(img, text, x, y, color):
    """A filled label above (x, y), kept inside the image."""
    (tw, th), base = cv2.getTextSize(text, FONT, 0.55, 1)
    x = max(0, min(x, img.shape[1] - tw - 8))
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), color, -1)
    cv2.putText(img, text, (x + 4, y - 5), FONT, 0.55, (20, 20, 20), 1, cv2.LINE_AA)


def bar(img, y0, y1, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (img.shape[1], y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw(img: np.ndarray, faces, objects, status: list[str], last_said: str) -> np.ndarray:
    for d in objects:
        brackets(img, d.box, CYAN, 1)
        tag(img, f"{d.label} {d.score:.0%}", d.box[0], d.box[1], CYAN)
    for f in faces:
        color = GREEN if f.name else AMBER
        brackets(img, f.box, color, 2)
        label = f.name or ("unknown" if f.similarity else "face")
        if f.name or f.similarity:
            label += f" {f.similarity:.2f}"
        tag(img, label, f.box[0], f.box[1], color)

    h, w = img.shape[:2]
    bar(img, 0, 28)
    cv2.putText(img, "SCOUTER", (8, 20), FONT, 0.6, GREEN, 2, cv2.LINE_AA)
    cv2.putText(img, "  ".join(status), (110, 20), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    if last_said:
        bar(img, h - 30, h)
        cv2.putText(img, last_said[:90], (8, h - 10), FONT, 0.5, GREEN, 1, cv2.LINE_AA)
    return img
