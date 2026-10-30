from datetime import datetime

import numpy as np

from scouter.events import EventLog, Presence, parse_since


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_presence_arrive_and_leave():
    clock = Clock()
    p = Presence({"person": 0.0, "unknown": 1.5}, leave_after=10, clock=clock)
    assert p.update({"person:Alex", "unknown"}) == [("arrive", "person:Alex")]
    clock.t = 1.0
    assert p.update({"person:Alex", "unknown"}) == []
    clock.t = 1.6
    assert p.update({"person:Alex", "unknown"}) == [("arrive", "unknown")]
    clock.t = 5
    assert p.update(set()) == []  # out of view, but not long enough to have left
    clock.t = 8
    assert p.update({"person:Alex"}) == []  # back: still the same visit
    clock.t = 16
    assert p.update({"person:Alex"}) == [("leave", "unknown")]
    clock.t = 30
    assert p.update(set()) == [("leave", "person:Alex")]
    assert p.present == []


def test_flicker_never_arrives():
    clock = Clock()
    p = Presence(1.0, clock=clock)
    for t in (0, 0.5, 1.0, 1.5):
        clock.t = t
        assert p.update({"object:cup"} if t % 1 == 0 else set()) == []


def test_parse_since():
    now = datetime(2026, 10, 30, 15, 0).timestamp()
    assert parse_since("2h", now) == now - 7200
    assert parse_since("30m", now) == now - 1800
    assert parse_since("today", now) == datetime(2026, 10, 30).timestamp()
    assert parse_since("yesterday", now) == datetime(2026, 10, 29).timestamp()
    assert parse_since("2026-10-01", now) == datetime(2026, 10, 1).timestamp()


def test_log_search_and_snapshots(tmp_path):
    clock = Clock(datetime(2026, 10, 30, 9, 0).timestamp())
    log = EventLog(tmp_path / "log.db", clock=clock)
    img = np.zeros((480, 640, 3), np.uint8)
    a = log.add("arrive", "person", "Alex", img)
    clock.t += 60
    log.add("arrive", "unknown", "unknown face", img)
    clock.t += 60
    log.add("leave", "person", "Alex")
    log.add("arrive", "object", "cup", img)

    assert [e.describe() for e in log.search()] == ["cup appeared", "Alex left", "unknown face arrived", "Alex arrived"]
    assert [e.action for e in log.search("alex")] == ["leave", "arrive"]
    assert [e.name for e in log.search(kind="object")] == ["cup"]
    assert len(log.search(since=clock.t - 30)) == 2
    assert log.snapshot_path(a.snapshot).read_bytes()[:2] == b"\xff\xd8"  # a JPEG
    assert log.snapshot_path("../log.db") is None  # can't escape the snapshot folder

    log.rename("Alex", "Alexandra")
    assert len(log.search("Alexandra")) == 2


def test_retention_by_age_and_size(tmp_path):
    day = 86400
    clock = Clock(datetime(2026, 10, 1).timestamp())
    log = EventLog(tmp_path / "log.db", keep_days=7, keep_mb=1, clock=clock)
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8)  # big, incompressible JPEGs
    old = log.add("arrive", "person", "Old", noise)
    clock.t += 10 * day
    recent = []
    for _ in range(12):
        recent.append(log.add("arrive", "unknown", "unknown face", noise))
        clock.t += 1

    log.prune()  # add() also prunes once an hour, so the old event went already
    assert log.search("Old") == [] and log.snapshot_path(old.snapshot) is None
    left = [e for e in log.search(limit=100) if e.snapshot]
    total = sum(log.snapshot_path(e.snapshot).stat().st_size for e in left)
    assert total <= 1024 * 1024 and 0 < len(left) < 12
    assert len(log.search(limit=100)) == 12  # events stay, only their oldest pictures went
    assert left[-1].id > recent[0].id  # it's the oldest snapshots that went
