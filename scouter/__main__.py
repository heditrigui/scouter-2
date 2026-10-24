"""Scouter 2: recognises people and objects through a camera and says what it sees.

    python -m scouter                      run with the default webcam
    python -m scouter enroll NAME          teach it a face (from the webcam, or --photos DIR)
    python -m scouter people               list known people
    python -m scouter forget NAME          remove a person
    python -m scouter download             fetch the models ahead of time
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import hud, models
from .camera import Camera
from .faces import FaceDB, FaceEngine, largest
from .objects import ObjectDetector
from .voice import Announcer, Voice

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "faces.npz"
WINDOW = "Scouter"
MIN_FACE_PX = 40  # smaller faces are too far away to recognise reliably

try:  # OpenCV 5 prints a harmless warning for every DNN model loaded
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def face_engine() -> FaceEngine:
    return FaceEngine(models.ensure(models.FACE_DETECTOR), models.ensure(models.FACE_RECOGNIZER))


class ObjectWorker:
    """Runs the (slow) object detector on the newest frame in the background."""

    def __init__(self, detector: ObjectDetector, on_result):
        self.detector = detector
        self.on_result = on_result
        self.results = []
        self._frame = None
        self._wake = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, frame: np.ndarray) -> None:
        self._frame = frame
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            frame, self._frame = self._frame, None
            if frame is None:
                continue
            self.results = self.detector.detect(frame)
            self.on_result(self.results)


def run(args) -> int:
    faces_on = not args.no_faces
    objects_on = not args.no_objects
    engine = face_engine()
    db = FaceDB(args.db)
    detector = ObjectDetector(models.ensure(models.OBJECT_DETECTOR), min_score=args.object_score)
    voice = Voice(enabled=not args.quiet)
    announcer = Announcer(person_cooldown=args.cooldown, object_cooldown=args.cooldown * 2)
    lock = threading.Lock()
    faces_in_view = 0

    if not db.people():
        print("No known people yet. Teach it a face with:  python -m scouter enroll YOUR_NAME", file=sys.stderr)

    def on_objects(dets):
        labels = [d.label for d in dets]
        if faces_in_view:  # people with a visible face are announced by name instead
            labels = [l for l in labels if l != "person"]
        with lock:
            voice.say(announcer.objects(labels))

    worker = ObjectWorker(detector, on_objects)
    cam = Camera(args.camera, args.width, args.height)
    print("Running. Keys: q quit, space say what you see, m mute, f faces on/off, o objects on/off", file=sys.stderr)
    voice.say("Scouter ready.")

    fps, last_t = 0.0, time.time()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Camera stopped.", file=sys.stderr)
                return 1

            faces = []
            if faces_on:
                for f in engine.detect(frame):
                    if f.box[2] >= MIN_FACE_PX:
                        f.name, f.similarity = db.match(engine.embed(frame, f), args.threshold)
                    faces.append(f)
                close = [f for f in faces if f.box[2] >= MIN_FACE_PX]
                faces_in_view = len(close)
                with lock:
                    voice.say(announcer.people([f.name for f in close]))
            else:
                faces_in_view = 0
            objects = worker.results if objects_on else []
            if objects_on:
                worker.submit(frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last_t, 1e-3)
            last_t = now

            if args.headless:
                continue
            status = [
                f"FACES {'ON' if faces_on else 'OFF'}",
                f"OBJECTS {'ON' if objects_on else 'OFF'}",
                "VOICE " + ("OFF" if args.quiet else "MUTED" if voice.muted else "ON"),
                f"{fps:4.1f} FPS",
            ]
            cv2.imshow(WINDOW, hud.draw(frame, faces, objects, status, voice.last))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return 0
            if key == ord(" "):
                with lock:
                    names = [f.name for f in faces if f.box[2] >= MIN_FACE_PX]
                    labels = [o.label for o in objects if not (names and o.label == "person")]
                    voice.say(announcer.describe(names, labels))
            elif key == ord("m"):
                voice.muted = not voice.muted
            elif key == ord("f"):
                faces_on = not faces_on
            elif key == ord("o"):
                objects_on = not objects_on
                worker.results = []
    except KeyboardInterrupt:
        return 0
    finally:
        cam.close()
        cv2.destroyAllWindows()


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
    print(f"Removed {removed} samples for {args.name}." if removed else f"{args.name} wasn't enrolled.")
    return 0 if removed else 1


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
    cam.add_argument("--headless", action="store_true", help="no window (e.g. a Raspberry Pi without a screen)")

    p = argparse.ArgumentParser(prog="scouter", description="Recognises people and objects and says what it sees.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", parents=[common, cam], help="start the scouter (default)")
    r.add_argument("--quiet", action="store_true", help="don't speak, only print")
    r.add_argument("--no-faces", action="store_true")
    r.add_argument("--no-objects", action="store_true")
    r.add_argument("--object-score", type=float, default=0.5, help="minimum object confidence (default 0.5)")
    r.add_argument("--cooldown", type=float, default=30, help="seconds before repeating a person (objects: twice this)")
    r.set_defaults(func=run)

    e = sub.add_parser("enroll", parents=[common, cam], help="teach it a face")
    e.add_argument("name")
    e.add_argument("--photos", help="folder of photos of this person instead of the webcam")
    e.add_argument("--samples", type=int, default=20, help="webcam samples to collect (default 20)")
    e.set_defaults(func=enroll)

    sub.add_parser("people", parents=[common], help="list known people").set_defaults(func=people)
    f = sub.add_parser("forget", parents=[common], help="remove a person")
    f.add_argument("name")
    f.set_defaults(func=forget)
    sub.add_parser("download", help="download the models now").set_defaults(func=download)

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
