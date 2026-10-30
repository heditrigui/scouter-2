"""Remembers where and when each object and person was last seen."""

import threading
import time
from dataclasses import dataclass, field

UNKNOWN = "someone I don't know"  # the name stored for unrecognised faces


@dataclass
class Sighting:
    kind: str  # "object" or "person"
    name: str  # COCO label, person's name, or UNKNOWN
    ts: float  # wall-clock time (time.time())
    region: str  # e.g. "bottom left", "middle"
    near: list[str] = field(default_factory=list)  # labels of objects next to it


def region(box, width: int, height: int) -> str:
    """Which ninth of the view the centre of [box] is in: 'top left', 'left', 'middle', ..."""
    x, y, w, h = box
    cx, cy = (x + w / 2) / max(width, 1), (y + h / 2) / max(height, 1)
    row = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else ""
    col = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ""
    return f"{row} {col}".strip() or "middle"


def touching(a, b, margin: float = 0.5) -> bool:
    """True when box [a], grown by [margin] of its smaller side, overlaps box [b]."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    m = margin * min(aw, ah)
    return ax - m < bx + bw and bx < ax + aw + m and ay - m < by + bh and by < ay + ah + m


def nearby(box, others, exclude: str = "") -> list[str]:
    """Labels of the [others] detections touching [box], biggest first, without duplicates."""
    out = []
    for d in sorted(others, key=lambda d: -d.box[2] * d.box[3]):
        if d.label != exclude and d.label not in out and touching(box, d.box):
            out.append(d.label)
    return out[:2]


class Memory:
    """The last sighting of every object label and every person.

    With a [store] (the EventLog), sightings survive restarts; each one is written at most
    every [write_every] seconds unless it moved, to keep the disk quiet.
    """

    def __init__(self, store=None, clock=time.time, write_every: float = 10.0):
        self.store = store
        self.clock = clock
        self.write_every = write_every
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, str], Sighting] = {}
        self._written: dict[tuple[str, str], Sighting] = {}
        if store is not None:
            for s in store.load_sightings():
                self._seen[(s.kind, s.name)] = self._written[(s.kind, s.name)] = s

    def _put(self, s: Sighting) -> None:
        key = (s.kind, s.name)
        self._seen[key] = s
        if self.store is None:
            return
        w = self._written.get(key)
        if w is None or s.ts - w.ts >= self.write_every or w.region != s.region:
            self.store.save_sighting(s)
            self._written[key] = s

    def saw_objects(self, dets, width: int, height: int) -> None:
        now = self.clock()
        with self._lock:
            for d in dets:
                if d.label != "person":
                    self._put(Sighting("object", d.label, now, region(d.box, width, height), nearby(d.box, dets, d.label)))

    def saw_people(self, faces, objects, width: int, height: int) -> None:
        """[faces] with .name (None = unknown) and .box; [objects] give the 'next to' context."""
        now = self.clock()
        others = [o for o in objects if o.label != "person"]
        with self._lock:
            for f in faces:
                self._put(Sighting("person", f.name or UNKNOWN, now, region(f.box, width, height), nearby(f.box, others)))

    def last(self, kind: str, name: str) -> Sighting | None:
        with self._lock:
            if kind == "person":  # names are matched case-insensitively
                return next((s for (k, n), s in self._seen.items() if k == kind and n.lower() == name.lower()), None)
            return self._seen.get((kind, name))

    def people_since(self, ts: float) -> list[Sighting]:
        with self._lock:
            return sorted((s for (k, _), s in self._seen.items() if k == "person" and s.ts >= ts), key=lambda s: -s.ts)

    def known_names(self) -> list[str]:
        with self._lock:
            return [n for (k, n) in self._seen if k == "person" and n != UNKNOWN]

    def rename(self, old: str, new: str | None) -> None:
        """Follows a rename in the face database; [new] = None forgets the person."""
        with self._lock:
            for key in [k for k in self._seen if k[0] == "person" and k[1].lower() == old.lower()]:
                s = self._seen.pop(key)
                self._written.pop(key, None)
                if new:
                    s.name = new
                    self._seen[("person", new)] = self._written[("person", new)] = s

    def flush(self) -> None:
        """Writes every sighting not yet stored (call on exit)."""
        if self.store is None:
            return
        with self._lock:
            for key, s in self._seen.items():
                if self._written.get(key) is not s:
                    self.store.save_sighting(s)
                    self._written[key] = s
