from datetime import datetime

from scouter.ask import ago, answer, labels_for
from scouter.events import EventLog
from scouter.memory import Memory, nearby, region
from scouter.objects import Detection


class Face:
    def __init__(self, name, box):
        self.name, self.box = name, box


NOW = datetime(2026, 10, 30, 15, 0, 0).timestamp()


class Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def test_regions_and_neighbours():
    assert region((0, 0, 50, 50), 600, 600) == "top left"
    assert region((275, 275, 50, 50), 600, 600) == "middle"
    assert region((550, 300, 50, 50), 600, 600) == "right"
    assert region((10, 550, 50, 50), 600, 600) == "bottom left"
    phone = Detection("cell phone", 0.9, (100, 100, 40, 80))
    laptop = Detection("laptop", 0.9, (150, 90, 200, 150))
    cup = Detection("cup", 0.9, (500, 500, 30, 30))
    assert nearby(phone.box, [phone, laptop, cup], "cell phone") == ["laptop"]


def test_labels_for_everyday_words():
    assert labels_for("my phone") == ["cell phone"]
    assert labels_for("the cups") == ["cup"]
    assert labels_for("laptop") == ["laptop"]
    assert labels_for("keys") == []


def test_ago():
    assert ago(NOW - 3, NOW) == "just now"
    assert ago(NOW - 45, NOW) == "45 seconds ago"
    assert ago(NOW - 60, NOW) == "a minute ago"
    assert ago(NOW - 12 * 60, NOW) == "12 minutes ago"
    assert ago(NOW - 2 * 3600, NOW) == "2 hours ago, at 13:00"
    assert ago(datetime(2026, 10, 29, 9, 15).timestamp(), NOW) == "yesterday at 09:15"
    assert ago(datetime(2026, 10, 3, 18, 40).timestamp(), NOW) == "on 3 October at 18:40"


def test_where_is_my_phone():
    clock = Clock(NOW - 5 * 60)
    m = Memory(clock=clock)
    m.saw_objects([Detection("cell phone", 0.9, (10, 400, 40, 60)), Detection("laptop", 0.9, (40, 350, 200, 120))], 640, 480)
    assert answer("Where is my phone?", m, NOW) == \
        "I last saw your phone 5 minutes ago, at the bottom left of the view, next to the laptop."
    assert answer("where's the laptop", m, NOW).startswith("I last saw the laptop 5 minutes ago")
    assert answer("where is my phone", m, NOW - 5 * 60 + 1).startswith("Your phone is in view right now")
    assert answer("where are my keys?", m, NOW).startswith("I can't recognise keys")
    assert answer("where is my cup", m, NOW) == "I haven't seen a cup yet."


def test_who_did_you_see_today():
    clock = Clock(NOW - 3600)
    m = Memory(clock=clock)
    m.saw_people([Face("Sam", (300, 100, 80, 80))], [], 640, 480)
    clock.t = NOW - 120
    m.saw_people([Face("Alex", (10, 10, 80, 80)), Face(None, (400, 10, 80, 80))], [], 640, 480)
    assert answer("Scouter who did you see today", m, NOW) == \
        "Today I saw Alex, 2 minutes ago, someone I don't know, 2 minutes ago and Sam, an hour ago, at 14:00."
    assert answer("when did you last see Sam", m, NOW) == "I last saw Sam an hour ago, at 14:00, at the top of the view."
    assert answer("who did you see today", Memory(), NOW) == "I haven't seen anyone today."


def test_what_do_you_see_and_help():
    m = Memory()
    assert answer("what do you see", m, NOW, describe=lambda: "I see a cup.") == "I see a cup."
    assert answer("sing me a song", m, NOW).startswith("You can ask me")


def test_sightings_survive_a_restart(tmp_path):
    clock = Clock(NOW - 600)
    log = EventLog(tmp_path / "s.db", clock=clock)
    m = Memory(log, clock=clock)
    m.saw_objects([Detection("cup", 0.9, (600, 10, 30, 30))], 640, 480)
    clock.t += 1
    m.saw_objects([Detection("cup", 0.9, (600, 10, 30, 30))], 640, 480)  # throttled: not written again
    m.flush()
    log.close()

    m2 = Memory(EventLog(tmp_path / "s.db"))
    s = m2.last("object", "cup")
    assert s.region == "top right" and s.ts == NOW - 599
    assert answer("where is my mug", m2, NOW).startswith("I last saw your mug 9 minutes ago, at the top right")


def test_rename_and_forget_follow_the_face_db():
    m = Memory(clock=Clock())
    m.saw_people([Face("Alex", (10, 10, 80, 80))], [], 640, 480)
    m.rename("alex", "Alexandra")
    assert m.last("person", "Alexandra") and not m.last("person", "Alex")
    m.rename("Alexandra", None)
    assert m.known_names() == []
