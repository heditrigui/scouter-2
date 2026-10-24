from scouter.voice import Announcer, join, phrase


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_phrases():
    assert phrase("cup", 1) == "a cup"
    assert phrase("apple", 1) == "an apple"
    assert phrase("cup", 2) == "two cups"
    assert phrase("person", 3) == "three people"
    assert phrase("wine glass", 2) == "two wine glasses"
    assert phrase("cup", 12) == "many cups"
    assert join(["a", "b", "c"]) == "a, b and c"


def test_known_person_greeted_once_per_cooldown():
    clock = Clock()
    a = Announcer(person_cooldown=30, clock=clock)
    assert a.people(["Alex"]) == "Hello Alex."
    clock.t = 10
    assert a.people(["Alex"]) is None
    clock.t = 31
    assert a.people(["Alex"]) == "Hello Alex."


def test_two_people_in_one_sentence():
    a = Announcer(clock=Clock())
    assert a.people(["Sam", "Alex"]) == "Hello Alex and Sam."


def test_unknown_face_needs_to_stay_in_view():
    clock = Clock()
    a = Announcer(unknown_delay=1.5, clock=clock)
    assert a.people([None]) is None
    clock.t = 1.0
    assert a.people([None]) is None
    clock.t = 1.6
    assert a.people([None]) == "someone I don't know is here."
    clock.t = 2.0
    assert a.people([None]) is None


def test_flickering_unknown_face_is_not_announced():
    clock = Clock()
    a = Announcer(unknown_delay=1.5, clock=clock)
    for t in (0.0, 1.0, 2.0, 3.0):
        clock.t = t
        assert a.people([None]) is None or t >= 1.5
        clock.t = t + 0.5
        a.people([])  # face lost: the timer starts over


def test_object_must_be_confirmed_then_cools_down():
    clock = Clock()
    a = Announcer(object_confirm=2, object_cooldown=60, clock=clock)
    assert a.objects(["cup"]) is None
    assert a.objects(["cup", "cup", "laptop"]) == "I see two cups."
    assert a.objects(["cup", "laptop"]) == "I see a laptop."
    assert a.objects(["cup", "laptop"]) is None
    clock.t = 61
    assert a.objects(["cup"]) == "I see a cup."


def test_object_streak_resets_when_it_disappears():
    a = Announcer(object_confirm=2, clock=Clock())
    assert a.objects(["dog"]) is None
    assert a.objects([]) is None
    assert a.objects(["dog"]) is None
    assert a.objects(["dog"]) == "I see a dog."


def test_describe_ignores_cooldowns():
    a = Announcer(clock=Clock())
    a.people(["Alex"])
    assert a.describe(["Alex", None], ["cup", "cup", "book"]) == "I see Alex, a person I don't know, two cups and a book."
    assert a.describe([], []) == "I don't see anything I recognise."
