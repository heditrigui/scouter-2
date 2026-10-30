from scouter.listen import WakeWord


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_question_after_wake_word():
    w = WakeWord(clock=Clock())
    assert w.feed("scouter where is my phone") == "where is my phone"
    assert w.feed("hey scooter who did you see today") == "who did you see today"
    assert w.feed("where is my phone") is None  # no wake word


def test_wake_word_alone_waits_for_the_question():
    clock = Clock()
    w = WakeWord(window=6, clock=clock)
    assert w.feed("scouter") is None and w.armed
    clock.t = 3
    assert w.feed("where are my keys") == "where are my keys"
    assert not w.armed
    w.feed("scouter")
    clock.t = 10
    assert w.feed("too late") is None
