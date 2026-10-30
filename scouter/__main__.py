"""Scouter 2: recognises people and objects through a camera and says what it sees.

    python -m scouter                      run with the default webcam
    python -m scouter enroll NAME          teach it a face (from the webcam, or --photos DIR)
    python -m scouter people               list known people
    python -m scouter forget NAME          remove a person
    python -m scouter log [TEXT]           search the event log
    python -m scouter ask "where is my phone?"
    python -m scouter download             fetch the models ahead of time
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import hud, models
from .app import WINDOW, face_engine
from .camera import Camera
from .faces import FaceDB, largest

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "faces.npz"
DEFAULT_LOG = ROOT / "data" / "scouter.db"

try:  # OpenCV 5 prints a harmless warning for every DNN model loaded
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def run(args) -> int:
    from .app import App

    return App(args).run()


def enroll(args) -> int:
    engine = face_engine()
    db = FaceDB(args.db)
    samples: list[np.ndarray] = []

    if args.photos:
        paths = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        for p in paths:
            img = cv2.imread(str(p))
            face = largest(engine.detect(img)) if img is not None else None
            if face is None:
                print(f"  {p.name}: no face found, skipped")
                continue
            samples.append(engine.embed(img, face))
            print(f"  {p.name}: ok")
    else:
        cam = Camera(args.camera, args.width, args.height)
        print(f"Look at the camera and slowly turn your head a little. Collecting {args.samples} samples...")
        last = 0.0
        try:
            while len(samples) < args.samples:
                frame = cam.read()
                if frame is None:
                    break
                faces = engine.detect(frame)
                good = len(faces) == 1 and faces[0].score >= 0.9 and faces[0].box[2] >= 80
                if good and time.time() - last >= 0.25:
                    samples.append(engine.embed(frame, faces[0]))
                    last = time.time()
                    if args.headless:
                        print(f"  {len(samples)}/{args.samples}")
                if args.headless:
                    continue
                for f in faces:
                    hud.brackets(frame, f.box, hud.GREEN if good else hud.AMBER, 2)
                msg = (f"{len(samples)}/{args.samples}  hold still..." if good
                       else "one face, closer to the camera" if faces else "no face in view")
                hud.bar(frame, 0, 28)
                cv2.putText(frame, f"ENROLLING {args.name}: {msg}", (8, 20), hud.FONT, 0.55, hud.GREEN, 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Cancelled.")
                    return 1
        finally:
            cam.close()
            cv2.destroyAllWindows()

    if not samples:
        print("No usable face samples; nothing saved.")
        return 1
    # Warn if this face already belongs to someone else in the database.
    for s in samples[:3]:
        other, sim = db.match(s, args.threshold)
        if other and other.lower() != args.name.lower():
            print(f"Warning: this face looks like {other} (similarity {sim:.2f}).")
            break
    db.add(args.name, samples)
    db.save()
    print(f"Saved {len(samples)} samples for {args.name}. Known people: {', '.join(db.people())}")
    return 0


def people(args) -> int:
    known = FaceDB(args.db).people()
    if not known:
        print("Nobody enrolled yet.")
    for name, n in known.items():
        print(f"{name}  ({n} samples)")
    return 0


def forget(args) -> int:
    db = FaceDB(args.db)
    removed = db.forget(args.name)
    db.save()
    if removed and Path(args.log_db).exists():  # also drop where they were last seen
        from .events import EventLog

        EventLog(args.log_db, keep_days=float("inf"), keep_mb=float("inf")).rename(args.name, None)
    print(f"Removed {removed} samples for {args.name}." if removed else f"{args.name} wasn't enrolled.")
    return 0 if removed else 1


def open_log(args):
    from .events import EventLog

    if not Path(args.log_db).exists():
        print("No event log yet: it's created the first time Scouter runs.")
        return None
    return EventLog(args.log_db, keep_days=float("inf"), keep_mb=float("inf"))


def log(args) -> int:
    from .events import parse_since

    events = open_log(args)
    if events is None:
        return 1
    try:
        since = parse_since(args.since) if args.since else None
        until = parse_since(args.until) if args.until else None
    except ValueError:
        print("--since/--until take e.g. 30m, 2h, 7d, today, yesterday or 2026-10-30.")
        return 2
    found = events.search(" ".join(args.text), since, until, args.type, args.limit)
    for e in reversed(found):
        snap = f"  {events.snapshots / e.snapshot}" if e.snapshot and args.snapshots else ""
        print(f"{e.time}  {e.describe()}{snap}")
    if not found:
        print("No matching events.")
    return 0


def ask(args) -> int:
    from .ask import answer
    from .memory import Memory

    events = open_log(args)
    if events is None:
        return 1
    print(answer(" ".join(args.question), Memory(events)))
    return 0


def download(args) -> int:
    for m in models.ALL:
        print(models.ensure(m))
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=DEFAULT_DB, help="face database file (default: data/faces.npz)")
    common.add_argument("--threshold", type=float, default=0.363, help="face match threshold, higher = stricter (default 0.363)")
    cam = argparse.ArgumentParser(add_help=False)
    cam.add_argument("--camera", default="0", help="webcam index, stream URL, or 'picamera' (default 0)")
    cam.add_argument("--width", type=int, default=640)
    cam.add_argument("--height", type=int, default=480)
    logs = argparse.ArgumentParser(add_help=False)
    logs.add_argument("--log-db", type=Path, default=DEFAULT_LOG, help="event log file (default: data/scouter.db)")
    cam.add_argument("--headless", action="store_true", help="no window (e.g. a Raspberry Pi without a screen)")

    p = argparse.ArgumentParser(prog="scouter", description="Recognises people and objects and says what it sees.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", parents=[common, cam, logs], help="start the scouter (default)")
    r.add_argument("--quiet", action="store_true", help="don't speak, only print")
    r.add_argument("--no-faces", action="store_true")
    r.add_argument("--no-objects", action="store_true")
    r.add_argument("--object-score", type=float, default=0.5, help="minimum object confidence (default 0.5)")
    r.add_argument("--cooldown", type=float, default=30, help="seconds before repeating a person (objects: twice this)")
    g = r.add_argument_group("event log")
    g.add_argument("--no-log", action="store_true", help="don't keep an event log, snapshots or sightings")
    g.add_argument("--keep-days", type=float, default=30, help="delete events older than this (default 30)")
    g.add_argument("--keep-mb", type=float, default=500, help="max megabytes of snapshots (default 500)")
    g.add_argument("--leave-after", type=float, default=15, help="seconds out of view before someone has left (default 15)")
    g = r.add_argument_group("questions")
    g.add_argument("--listen", action="store_true", help='answer spoken questions after "Scouter" (needs vosk)')
    g.add_argument("--vosk-model", type=Path, help="Vosk model folder (default models/vosk-model-small-en-us-0.15)")
    g = r.add_argument_group("alerts")
    g.add_argument("--alert-after", type=float, metavar="SECONDS", help="alert when an unknown face stays this long")
    g.add_argument("--alert-cooldown", type=float, default=300, help="seconds between alerts (default 300)")
    g.add_argument("--quiet-hours", metavar="HH:MM-HH:MM", help="no alerts in this range, e.g. 08:00-18:00")
    g.add_argument("--no-desktop-alert", action="store_true", help="don't show a desktop notification")
    g.add_argument("--telegram-token", help="Telegram bot token (or SCOUTER_TELEGRAM_TOKEN)")
    g.add_argument("--telegram-chat", help="Telegram chat id (or SCOUTER_TELEGRAM_CHAT)")
    g.add_argument("--webhook", help="URL to POST alerts to as JSON (or SCOUTER_WEBHOOK)")
    g = r.add_argument_group("web dashboard")
    g.add_argument("--web", action="store_true", help="serve the dashboard (default http://localhost:8080)")
    g.add_argument("--web-port", type=int, default=8080)
    g.add_argument("--web-lan", action="store_true", help="let other devices on the network in (needs a password)")
    g.add_argument("--web-password", help="dashboard password (or SCOUTER_WEB_PASSWORD)")
    r.set_defaults(func=run)

    e = sub.add_parser("enroll", parents=[common, cam], help="teach it a face")
    e.add_argument("name")
    e.add_argument("--photos", help="folder of photos of this person instead of the webcam")
    e.add_argument("--samples", type=int, default=20, help="webcam samples to collect (default 20)")
    e.set_defaults(func=enroll)

    sub.add_parser("people", parents=[common], help="list known people").set_defaults(func=people)
    f = sub.add_parser("forget", parents=[common, logs], help="remove a person")
    f.add_argument("name")
    f.set_defaults(func=forget)
    lg = sub.add_parser("log", parents=[logs], help="search the event log")
    lg.add_argument("text", nargs="*", help="name or word to look for, e.g. Alex, unknown, cup, alert")
    lg.add_argument("--since", help="30m, 2h, 7d, today, yesterday or a date like 2026-10-30")
    lg.add_argument("--until", help="same formats as --since")
    lg.add_argument("--type", choices=("person", "unknown", "object"))
    lg.add_argument("--limit", type=int, default=50)
    lg.add_argument("--snapshots", action="store_true", help="show the snapshot file of each event")
    lg.set_defaults(func=log)
    a = sub.add_parser("ask", parents=[logs], help="ask about what it saw (uses the log)")
    a.add_argument("question", nargs="+")
    a.set_defaults(func=ask)
    sub.add_parser("download", help="download the models now").set_defaults(func=download)

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]
    args = p.parse_args(argv)
    if args.cmd == "run":
        args.web_password = args.web_password or os.environ.get("SCOUTER_WEB_PASSWORD")
        args.web_host = "0.0.0.0" if args.web_lan else "127.0.0.1"
        if args.web_lan and not args.web_password:
            p.error("--web-lan needs --web-password (or SCOUTER_WEB_PASSWORD)")
        if args.quiet_hours:
            from .alerts import parse_quiet

            try:
                parse_quiet(args.quiet_hours)
            except ValueError:
                p.error("--quiet-hours looks like 23:00-07:00")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
