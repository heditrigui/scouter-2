"""Arrivals and departures, kept in a local SQLite log with JPEG snapshots."""

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .memory import Sighting


class Presence:
    """Turns "what's in view this frame" into arrive / leave events.

    A key arrives once it has been in view for its [arrive_after] seconds (so a face that
    flickers past isn't logged), and leaves after [leave_after] seconds out of view.
    """

    def __init__(self, arrive_after: dict[str, float] | float = 0.0, leave_after: float = 15.0, clock=time.monotonic):
        self.arrive_after = arrive_after
        self.leave_after = leave_after
        self.clock = clock
        self._since: dict[str, float] = {}  # in view (not yet arrived) since
        self._present: dict[str, float] = {}  # arrived; value = last time in view

    def _delay(self, key: str) -> float:
        if isinstance(self.arrive_after, dict):
            return self.arrive_after.get(key.split(":")[0], 0.0)
        return self.arrive_after

    def update(self, keys) -> list[tuple[str, str]]:
        """Call with the keys in view now (e.g. 'person:Alex', 'unknown', 'object:cup')."""
        now = self.clock()
        keys = set(keys)
        events = []
        for k in keys:
            if k in self._present:
                self._present[k] = now
                continue
            start = self._since.setdefault(k, now)
            if now - start >= self._delay(k):
                del self._since[k]
                self._present[k] = now
                events.append(("arrive", k))
        for k in [k for k in self._since if k not in keys]:
            del self._since[k]
        for k, last in list(self._present.items()):
            if k not in keys and now - last >= self.leave_after:
                del self._present[k]
                events.append(("leave", k))
        return sorted(events)

    @property
    def present(self) -> list[str]:
        return sorted(self._present)


@dataclass
class Event:
    id: int
    ts: float
    action: str  # "arrive", "leave" or "alert"
    kind: str  # "person", "unknown" or "object"
    name: str
    snapshot: str | None  # path relative to the snapshot folder

    @property
    def time(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d %H:%M:%S")

    def describe(self) -> str:
        who = {"unknown": "unknown face", "object": self.name, "person": self.name}[self.kind]
        verb = {"arrive": "arrived" if self.kind != "object" else "appeared", "leave": "left",
                "alert": "ALERT"}.get(self.action, self.action)
        return f"{who} {verb}" if self.action != "alert" else f"ALERT: {self.name}"

    def as_dict(self) -> dict:
        return {"id": self.id, "ts": self.ts, "time": self.time, "action": self.action, "kind": self.kind,
                "name": self.name, "text": self.describe(), "snapshot": self.snapshot}


def parse_since(text: str, now: float | None = None) -> float:
    """'30m', '2h', '7d', 'today', 'yesterday' or '2026-10-30' -> a timestamp."""
    now = time.time() if now is None else now
    text = text.strip().lower()
    midnight = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    if text == "today":
        return midnight
    if text == "yesterday":
        return midnight - 86400
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if text[:-1].replace(".", "", 1).isdigit() and text[-1] in units:
        return now - float(text[:-1]) * units[text[-1]]
    return datetime.fromisoformat(text).timestamp()


class EventLog:
    """SQLite file with the event log and the last sightings; snapshots are JPEG files next to it.

    Retention: events older than [keep_days] are deleted, and the oldest snapshots are
    removed whenever they take more than [keep_mb] megabytes.
    """

    def __init__(self, path: Path, snapshots: Path | None = None, keep_days: float = 30, keep_mb: float = 500,
                 clock=time.time):
        self.path = Path(path)
        self.snapshots = Path(snapshots) if snapshots else self.path.parent / "snapshots"
        self.keep_days = keep_days
        self.keep_mb = keep_mb
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, ts REAL NOT NULL, action TEXT NOT NULL, kind TEXT NOT NULL,
                name TEXT NOT NULL, snapshot TEXT);
            CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
            CREATE TABLE IF NOT EXISTS sightings (
                kind TEXT NOT NULL, name TEXT NOT NULL, ts REAL NOT NULL, region TEXT NOT NULL,
                near TEXT NOT NULL, PRIMARY KEY (kind, name));
        """)
        self._db.commit()
        self._pruned = 0.0

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add(self, action: str, kind: str, name: str, image: np.ndarray | None = None) -> Event:
        now = self.clock()
        snap = None
        if image is not None:
            snap = self._save_snapshot(image, now, kind, name)
        with self._lock:
            cur = self._db.execute("INSERT INTO events (ts, action, kind, name, snapshot) VALUES (?, ?, ?, ?, ?)",
                                   (now, action, kind, name, snap))
            self._db.commit()
            event = Event(cur.lastrowid, now, action, kind, name, snap)
        if now - self._pruned > 3600:
            self.prune()
        return event

    def _save_snapshot(self, image: np.ndarray, ts: float, kind: str, name: str) -> str | None:
        h, w = image.shape[:2]
        if w > 640:
            image = cv2.resize(image, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        when = datetime.fromtimestamp(ts)
        slug = "".join(c if c.isalnum() else "-" for c in name.lower())[:30]
        rel = f"{when:%Y-%m-%d}/{when:%H%M%S}-{int(ts * 1000) % 1000:03d}-{uuid.uuid4().hex[:4]}-{kind}-{slug}.jpg"
        out = self.snapshots / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(jpg.tobytes())
        return rel

    def search(self, text: str = "", since: float | None = None, until: float | None = None,
               kind: str | None = None, limit: int = 50) -> list[Event]:
        """Newest first. [text] matches the name or action (case-insensitive)."""
        sql, args = "SELECT id, ts, action, kind, name, snapshot FROM events WHERE 1=1", []
        if text:
            sql += " AND (name LIKE ? OR action LIKE ?)"
            args += [f"%{text}%", f"%{text}%"]
        if since is not None:
            sql += " AND ts >= ?"
            args.append(since)
        if until is not None:
            sql += " AND ts < ?"
            args.append(until)
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [Event(*row) for row in self._db.execute(sql, args)]

    def snapshot_path(self, rel: str) -> Path | None:
        """The file for a snapshot name from the log, or None if it isn't inside the snapshot folder."""
        root = self.snapshots.resolve()
        p = (root / rel).resolve()
        return p if p.is_file() and root in p.parents else None

    def prune(self) -> tuple[int, int]:
        """Applies the retention limits; returns (events deleted, snapshots deleted)."""
        now = self.clock()
        self._pruned = now
        cutoff = now - self.keep_days * 86400
        with self._lock:
            old = self._db.execute("SELECT snapshot FROM events WHERE ts < ? AND snapshot IS NOT NULL", (cutoff,)).fetchall()
            n_events = self._db.execute("DELETE FROM events WHERE ts < ?", (cutoff,)).rowcount
            self._db.commit()
        n_snaps = 0
        for (rel,) in old:
            n_snaps += self._unlink(rel)

        files = sorted(self.snapshots.rglob("*.jpg")) if self.snapshots.exists() else []
        total = sum(f.stat().st_size for f in files)
        limit = self.keep_mb * 1024 * 1024
        doomed = []
        for f in files:  # names start with the date and time, so sorted = oldest first
            if total <= limit:
                break
            total -= f.stat().st_size
            doomed.append(f)
        if doomed:
            rels = [f.relative_to(self.snapshots).as_posix() for f in doomed]
            with self._lock:
                self._db.executemany("UPDATE events SET snapshot = NULL WHERE snapshot = ?", [(r,) for r in rels])
                self._db.commit()
            for r in rels:
                n_snaps += self._unlink(r)
        for d in sorted(self.snapshots.glob("*"), reverse=True) if self.snapshots.exists() else []:
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        return n_events, n_snaps

    def _unlink(self, rel: str) -> int:
        p = self.snapshots / rel
        if p.exists():
            p.unlink()
            return 1
        return 0

    # Sightings, used by Memory so "where is my phone?" works after a restart.

    def save_sighting(self, s: Sighting) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO sightings (kind, name, ts, region, near) VALUES (?, ?, ?, ?, ?)",
                             (s.kind, s.name, s.ts, s.region, "|".join(s.near)))
            self._db.commit()

    def load_sightings(self) -> list[Sighting]:
        with self._lock:
            rows = self._db.execute("SELECT kind, name, ts, region, near FROM sightings").fetchall()
        return [Sighting(k, n, ts, r, near.split("|") if near else []) for k, n, ts, r, near in rows]

    def rename(self, old: str, new: str | None) -> None:
        """Follows a person being renamed in the face database; [new] = None forgets their last sighting."""
        with self._lock:
            if new:
                self._db.execute("UPDATE events SET name = ? WHERE kind = 'person' AND name = ? COLLATE NOCASE", (new, old))
                if new.lower() != old.lower():
                    self._db.execute("DELETE FROM sightings WHERE kind = 'person' AND name = ? COLLATE NOCASE", (new,))
                self._db.execute("UPDATE sightings SET name = ? WHERE kind = 'person' AND name = ? COLLATE NOCASE",
                                 (new, old))
            else:
                self._db.execute("DELETE FROM sightings WHERE kind = 'person' AND name = ? COLLATE NOCASE", (old,))
            self._db.commit()
