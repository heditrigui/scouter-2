"""Answers spoken or typed questions: "where is my phone?", "who did you see today?", ..."""

import re
import time
from datetime import datetime, timedelta

from .memory import UNKNOWN, Memory, Sighting
from .objects import COCO
from .voice import join, phrase

# Everyday words for COCO labels. A word that maps to several labels means "any of these".
SYNONYMS = {
    "phone": ["cell phone"], "mobile": ["cell phone"], "cellphone": ["cell phone"], "smartphone": ["cell phone"],
    "mug": ["cup"], "glass": ["wine glass", "cup"], "bag": ["backpack", "handbag", "suitcase"],
    "purse": ["handbag"], "rucksack": ["backpack"], "tv": ["tv"], "television": ["tv"], "telly": ["tv"],
    "remote": ["remote"], "remote control": ["remote"], "controller": ["remote"], "sofa": ["couch"],
    "table": ["dining table"], "plant": ["potted plant"], "computer": ["laptop"], "notebook": ["laptop", "book"],
    "bike": ["bicycle"], "ball": ["sports ball"], "teddy": ["teddy bear"], "fridge": ["refrigerator"],
    "hairdryer": ["hair drier"], "hair dryer": ["hair drier"], "drink": ["cup", "bottle", "wine glass"],
    "kitty": ["cat"], "puppy": ["dog"], "doggy": ["dog"],
}
# Things people ask about that YOLOX has no class for.
UNKNOWABLE = {"key", "keys", "wallet", "glasses", "sunglasses", "watch", "ring", "pen", "charger", "earbuds",
              "headphones", "headset", "card", "passport", "money", "lighter", "jacket", "coat", "shoe", "shoes",
              "hat", "tablet", "ipad"}

HELP = "You can ask me: where is my phone, who did you see today, when did you last see a name, or what do you see."


def _singular(word: str) -> str:
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def labels_for(thing: str) -> list[str]:
    """COCO labels meant by [thing] ('my phone' -> ['cell phone']), or [] if it isn't one we know."""
    thing = re.sub(r"^(my|the|a|an|our|your)\s+", "", thing.strip().lower())
    for t in (thing, _singular(thing)):
        if t in COCO:
            return [t]
        if t in SYNONYMS:
            return SYNONYMS[t]
    return []


def ago(ts: float, now: float) -> str:
    """'just now', '5 minutes ago', 'at 14:03', 'yesterday at 09:15', 'on 28 October at 18:40'."""
    dt = max(0.0, now - ts)
    when = datetime.fromtimestamp(ts)
    today = datetime.fromtimestamp(now).date()
    if dt < 10:
        return "just now"
    if dt < 60:
        return f"{int(dt)} seconds ago"
    if dt < 3600:
        m = int(dt // 60)
        return "a minute ago" if m == 1 else f"{m} minutes ago"
    clock = when.strftime("%H:%M")
    if when.date() == today:
        h = int(dt // 3600)
        return f"{'an hour' if h == 1 else f'{h} hours'} ago, at {clock}"
    if when.date() == today - timedelta(days=1):
        return f"yesterday at {clock}"
    return f"on {when.day} {when.strftime('%B')} at {clock}"


def where(s: Sighting) -> str:
    place = "in the middle of the view" if s.region == "middle" else f"at the {s.region} of the view"
    if s.near:
        place += ", next to " + join([phrase(n, 1).replace("a ", "the ", 1).replace("an ", "the ", 1) for n in s.near])
    return place


def start_of_day(now: float) -> float:
    return datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def answer(question: str, memory: Memory, now: float | None = None, describe=None, live: float = 3.0) -> str:
    """One spoken sentence answering [question]. [describe] returns what is in view right now."""
    now = time.time() if now is None else now
    q = re.sub(r"[^a-z0-9' ]+", " ", question.lower()).strip()
    q = re.sub(r"\s+", " ", q)
    if not q:
        return HELP

    if re.search(r"\bwho\b.*\b(see|seen|saw|here|visit|visited|came|been)\b", q):
        since = start_of_day(now)
        seen = memory.people_since(since)
        if not seen:
            return "I haven't seen anyone today."
        parts = []
        for s in seen:
            name = s.name if s.name != UNKNOWN else "someone I don't know"
            parts.append(f"{name}, {'right now' if now - s.ts < live else ago(s.ts, now)}")
        return "Today I saw " + join(parts) + "."

    if re.search(r"\bwhat (do|can) you see\b|\bwhat'?s (there|in view|in front)\b|\bdescribe\b", q):
        return describe() if describe else "I'm not looking right now."

    m = re.search(r"\b(?:where(?: is| are|'s|s)?|have you seen|did you see|when did you (?:last )?see|"
                  r"when was|last seen|find|seen)\s+(.+?)(?:\s+(?:last seen|last|today|recently|anywhere|now|please))*$", q)
    if not m:
        return HELP
    thing = m.group(1).strip()
    thing = re.sub(r"\s+(?:is|are|was)$", "", thing)
    mine = bool(re.match(r"(my|our)\s", thing))
    bare = re.sub(r"^(my|the|a|an|our|your)\s+", "", thing)

    # A person?
    for name in memory.known_names():
        if bare == name.lower():
            s = memory.last("person", name)
            if now - s.ts < live:
                return f"{name} is here right now, {where(s)}."
            return f"I last saw {name} {ago(s.ts, now)}, {where(s)}."

    labels = labels_for(bare)
    if not labels:
        if _singular(bare) in UNKNOWABLE or bare in UNKNOWABLE:
            return f"I can't recognise {bare} yet. I know 80 kinds of things, like phones, cups, bags and laptops."
        return f"I don't know what {bare} is. I know people I've been taught, and 80 kinds of things like phones, cups and bags."

    seen = [s for s in (memory.last("object", l) for l in labels) if s]
    word = "your" if mine else "the"
    if not seen:
        return f"I haven't seen {phrase(labels[0], 1)} yet."
    s = max(seen, key=lambda s: s.ts)
    subject = f"{word} {s.name if len(labels) > 1 or bare not in SYNONYMS else bare}"
    if now - s.ts < live:
        return f"{subject[0].upper()}{subject[1:]} is in view right now, {where(s)}."
    return f"I last saw {subject} {ago(s.ts, now)}, {where(s)}."
