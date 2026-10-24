"""Speaking out loud, and deciding what is worth saying."""

import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter

NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
PLURALS = {"person": "people", "knife": "knives", "mouse": "mice", "sheep": "sheep", "skis": "skis",
           "scissors": "scissors", "bus": "buses", "couch": "couches", "sandwich": "sandwiches",
           "toothbrush": "toothbrushes", "glass": "glasses"}


def plural(label: str) -> str:
    words = label.split()
    last = words[-1]
    words[-1] = PLURALS.get(last, last if last.endswith("s") else last + "s")
    return " ".join(words)


def phrase(label: str, count: int) -> str:
    """'a cup', 'an apple', 'two cups', 'many people'."""
    if count == 1:
        return ("an " if label[0] in "aeiou" else "a ") + label
    return f"{NUMBERS.get(count, 'many')} {plural(label)}"


def join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


class Announcer:
    """Turns what the camera sees into short sentences, without repeating itself.

    - A known person is greeted when they show up, then not again for [person_cooldown] seconds.
    - An unknown face must stay in view for [unknown_delay] seconds before it's mentioned,
      so a face that flickers in and out isn't announced.
    - An object must be seen in [object_confirm] detection passes in a row before it's
      mentioned, then not again for [object_cooldown] seconds.
    """

    def __init__(self, person_cooldown=30.0, unknown_delay=1.5, object_cooldown=60.0, object_confirm=2,
                 clock=time.monotonic):
        self.person_cooldown = person_cooldown
        self.unknown_delay = unknown_delay
        self.object_cooldown = object_cooldown
        self.object_confirm = object_confirm
        self.clock = clock
        self._said: dict[str, float] = {}
        self._unknown_since: float | None = None
        self._streak: Counter = Counter()

    def _due(self, key: str, cooldown: float, now: float) -> bool:
        last = self._said.get(key)
        return last is None or now - last >= cooldown

    def people(self, names: list[str | None]) -> str | None:
        """Call every frame with the recognised names (None for an unknown face)."""
        now = self.clock()
        known = sorted({n for n in names if n})
        new = [n for n in known if self._due("person:" + n, self.person_cooldown, now)]
        for n in new:
            self._said["person:" + n] = now

        unknown = sum(1 for n in names if n is None)
        say_unknown = False
        if unknown:
            if self._unknown_since is None:
                self._unknown_since = now
            if now - self._unknown_since >= self.unknown_delay and self._due("unknown", self.person_cooldown, now):
                self._said["unknown"] = now
                say_unknown = True
        else:
            self._unknown_since = None

        parts = [f"Hello {join(new)}"] if new else []
        if say_unknown:
            parts.append("someone I don't know is here" if unknown == 1 else f"{phrase('stranger', unknown)} are here")
        return ". ".join(parts) + "." if parts else None

    def objects(self, labels: list[str]) -> str | None:
        """Call after every object detection pass with the labels it found."""
        now = self.clock()
        counts = Counter(labels)
        for label in list(self._streak):
            if label not in counts:
                del self._streak[label]
        for label in counts:
            self._streak[label] += 1
        ready = [l for l in counts if self._streak[l] >= self.object_confirm and self._due("obj:" + l, self.object_cooldown, now)]
        if not ready:
            return None
        for l in ready:
            self._said["obj:" + l] = now
        ready.sort(key=lambda l: (-counts[l], l))
        return "I see " + join([phrase(l, counts[l]) for l in ready]) + "."

    def describe(self, names: list[str | None], labels: list[str]) -> str:
        """Everything in view right now, ignoring cooldowns (the 'what do you see?' key)."""
        parts = []
        known = sorted({n for n in names if n})
        if known:
            parts.append(join(known))
        unknown = sum(1 for n in names if n is None)
        if unknown:
            parts.append(phrase("person I don't know", 1) if unknown == 1 else f"{NUMBERS.get(unknown, 'many')} people I don't know")
        counts = Counter(labels)
        parts += [phrase(l, counts[l]) for l in sorted(counts, key=lambda l: (-counts[l], l))]
        return ("I see " + join(parts) + ".") if parts else "I don't see anything I recognise."


class Voice:
    """Speaks sentences one at a time on a background thread, dropping stale ones."""

    def __init__(self, enabled: bool = True, rate: int = 175):
        self.enabled = enabled
        self.muted = False
        self.last = ""
        self._q: queue.Queue[str] = queue.Queue()
        self._rate = rate
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str | None) -> None:
        if not text:
            return
        self.last = text
        print("[say]", text, flush=True)
        if self.enabled and not self.muted:
            self._q.put(text)

    def _run(self) -> None:
        speak = self._backend()
        while True:
            text = self._q.get()
            # If several sentences piled up while speaking, only the newest is still relevant.
            while not self._q.empty():
                text = self._q.get()
            try:
                speak(text)
            except Exception as e:  # never let speech take the app down
                print("[voice] error:", e, file=sys.stderr)

    def _backend(self):
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)

            def speak(text):
                engine.say(text)
                engine.runAndWait()

            return speak
        except Exception as e:
            exe = shutil.which("espeak-ng") or shutil.which("espeak")
            if exe:
                return lambda text: subprocess.run([exe, "-s", str(self._rate), text], check=False)
            print(f"[voice] no speech engine available ({e}); printing only", file=sys.stderr)
            return lambda text: None
