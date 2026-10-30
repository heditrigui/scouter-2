"""The running scouter: camera loop, announcements, memory, event log, alerts, questions."""

import os
import sys
import threading
import time

import cv2
import numpy as np

from . import hud, models
from .alerts import AlertRule, Notifier, parse_quiet
from .ask import answer
from .camera import Camera
from .events import EventLog, Presence
from .faces import FaceDB, FaceEngine
from .memory import Memory
from .objects import ObjectDetector
from .voice import Announcer, Voice

WINDOW = "Scouter"
MIN_FACE_PX = 40  # smaller faces are too far away to recognise reliably


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
            self.on_result(self.results, frame)


def _subject(key: str) -> tuple[str, str]:
    """Presence key -> (kind, name) for the event log."""
    if key == "unknown":
        return "unknown", "unknown face"
    kind, name = key.split(":", 1)
    return kind, name


class App:
    def __init__(self, args):
        self.args = args
        self.faces_on = not args.no_faces
        self.objects_on = not args.no_objects
        self.engine = face_engine()
        self.db = FaceDB(args.db)
        self.db_lock = threading.Lock()
        self.detector = ObjectDetector(models.ensure(models.OBJECT_DETECTOR), min_score=args.object_score)
        self.voice = Voice(enabled=not args.quiet)
        self.announcer = Announcer(person_cooldown=args.cooldown, object_cooldown=args.cooldown * 2)
        self.lock = threading.Lock()  # around the announcer
        self.log = None if args.no_log else EventLog(args.log_db, keep_days=args.keep_days, keep_mb=args.keep_mb)
        if self.log:
            self.log.prune()
        self.memory = Memory(self.log)
        self.people_presence = Presence({"person": 0.0, "unknown": 1.5}, leave_after=args.leave_after)
        self.object_presence = Presence(1.0, leave_after=max(60.0, args.leave_after))
        self.alert = None
        if args.alert_after is not None:
            self.alert = AlertRule(args.alert_after, args.alert_cooldown, parse_quiet(args.quiet_hours))
            self.notifier = Notifier(
                desktop=not args.no_desktop_alert,
                telegram_token=args.telegram_token or os.environ.get("SCOUTER_TELEGRAM_TOKEN"),
                telegram_chat=args.telegram_chat or os.environ.get("SCOUTER_TELEGRAM_CHAT"),
                webhook=args.webhook or os.environ.get("SCOUTER_WEBHOOK"))
        self.hub = None  # set by the web dashboard
        self.fps = 0.0
        self.faces_in_view = 0
        self.faces: list = []
        self.objects: list = []
        self._enroll: dict | None = None
        self.enroll_result = ""
        self.listener = None

    # What is in view, for "what do you see?" and the space key.

    def describe_now(self) -> str:
        names = [f.name for f in self.faces if f.box[2] >= MIN_FACE_PX]
        labels = [o.label for o in self.objects if not (names and o.label == "person")]
        with self.lock:
            return self.announcer.describe(names, labels)

    def ask(self, question: str) -> str:
        reply = answer(question, self.memory, describe=self.describe_now)
        print(f"[ask] {question}", flush=True)
        self.voice.say(reply)
        return reply

    # People management (also used by the web dashboard).

    def people(self) -> dict[str, int]:
        with self.db_lock:
            return self.db.people()

    def rename(self, old: str, new: str) -> bool:
        new = new.strip()
        if not new:
            return False
        with self.db_lock:
            moved = self.db.rename(old, new)
            if moved:
                self.db.save()
        if moved:
            self.memory.rename(old, new)
            if self.log:
                self.log.rename(old, new)
        return bool(moved)

    def forget(self, name: str) -> bool:
        with self.db_lock:
            removed = self.db.forget(name)
            if removed:
                self.db.save()
        if removed:
            self.memory.rename(name, None)
            if self.log:
                self.log.rename(name, None)
        return bool(removed)

    def start_enroll(self, name: str, samples: int = 15) -> None:
        self._enroll = {"name": name.strip(), "samples": [], "target": samples, "last": 0.0}
        self.enroll_result = ""

    @property
    def enroll_status(self) -> dict:
        e = self._enroll
        if e is None:
            return {"active": False, "result": self.enroll_result}
        return {"active": True, "name": e["name"], "done": len(e["samples"]), "target": e["target"]}

    def _enroll_step(self, frame, faces) -> None:
        e = self._enroll
        good = len(faces) == 1 and faces[0].score >= 0.9 and faces[0].box[2] >= 80
        if not good or time.time() - e["last"] < 0.25:
            return
        e["samples"].append(self.engine.embed(frame, faces[0]))
        e["last"] = time.time()
        if len(e["samples"]) >= e["target"]:
            with self.db_lock:
                self.db.add(e["name"], e["samples"])
                self.db.save()
            self.enroll_result = f"Saved {len(e['samples'])} samples for {e['name']}."
            print(self.enroll_result, file=sys.stderr)
            self._enroll = None

    # Events.

    def _log(self, events, snapshot) -> None:
        for action, key in events:
            kind, name = _subject(key)
            if kind == "object" and action == "leave":
                continue  # objects come and go with the detector's confidence; only new ones are logged
            if self.log:
                e = self.log.add(action, kind, name, snapshot() if action == "arrive" else None)
                print(f"[log] {e.time}  {e.describe()}", flush=True)

    def _on_objects(self, dets, frame) -> None:
        labels = [d.label for d in dets]
        if self.faces_in_view:  # people with a visible face are announced by name instead
            labels = [l for l in labels if l != "person"]
        with self.lock:
            self.voice.say(self.announcer.objects(labels))
        h, w = frame.shape[:2]
        self.memory.saw_objects(dets, w, h)
        events = self.object_presence.update({"object:" + d.label for d in dets if d.label != "person"})

        def snapshot():
            img = frame.copy()
            for d in dets:
                hud.brackets(img, d.box, hud.CYAN, 2)
                hud.tag(img, d.label, d.box[0], d.box[1], hud.CYAN)
            return img

        self._log(events, snapshot)

    def _step(self, frame: np.ndarray, show: bool) -> np.ndarray | None:
        h, w = frame.shape[:2]
        faces = []
        if self.faces_on:
            for f in self.engine.detect(frame):
                if f.box[2] >= MIN_FACE_PX:
                    with self.db_lock:
                        f.name, f.similarity = self.db.match(self.engine.embed(frame, f), self.args.threshold)
                faces.append(f)
            if self._enroll is not None:
                self._enroll_step(frame, faces)
        close = [f for f in faces if f.box[2] >= MIN_FACE_PX]
        self.faces, self.faces_in_view = faces, len(close)
        objects = self.objects = self.worker.results if self.objects_on else []
        if self.faces_on:
            with self.lock:
                self.voice.say(self.announcer.people([f.name for f in close]))
            self.memory.saw_people(close, objects, w, h)
        if self.objects_on:
            self.worker.submit(frame)

        annotated = None

        def snapshot():
            nonlocal annotated
            if annotated is None:
                annotated = hud.draw(frame.copy(), faces, objects, self.status(), self.voice.last)
            return annotated

        keys = {"person:" + f.name if f.name else "unknown" for f in close}
        self._log(self.people_presence.update(keys), snapshot)

        unknown = sum(1 for f in close if f.name is None)
        if self.alert and self.alert.update(unknown):
            msg = ("Someone I don't know" if unknown == 1 else f"{unknown} people I don't know") + \
                  f" has been in view for {self.alert.after:g} seconds."
            ok, jpg = cv2.imencode(".jpg", snapshot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if self.log:
                self.log.add("alert", "unknown", msg, snapshot())
            print("[alert]", msg, flush=True)
            self.notifier.send(msg, jpg.tobytes() if ok else None)

        if self.hub is not None and self.hub.wanted():
            self.hub.publish(snapshot())
        return snapshot() if show else None

    def status(self) -> list[str]:
        out = [
            f"FACES {'ON' if self.faces_on else 'OFF'}",
            f"OBJECTS {'ON' if self.objects_on else 'OFF'}",
            "VOICE " + ("OFF" if self.args.quiet else "MUTED" if self.voice.muted else "ON"),
            f"{self.fps:4.1f} FPS",
        ]
        if self._enroll is not None:
            out.append(f"ENROLLING {len(self._enroll['samples'])}/{self._enroll['target']}")
        return out

    def _read_stdin(self) -> None:
        for line in sys.stdin:
            if line.strip():
                print("[answer]", self.ask(line.strip()), flush=True)

    def run(self) -> int:
        args = self.args
        if not self.db.people():
            print("No known people yet. Teach it a face with:  python -m scouter enroll YOUR_NAME", file=sys.stderr)
        self.worker = ObjectWorker(self.detector, self._on_objects)
        cam = Camera(args.camera, args.width, args.height)
        if args.web:
            from .web import serve

            serve(self, args.web_host, args.web_port, args.web_password)
        if args.listen:
            from .listen import Listener

            try:
                self.listener = Listener(self.ask, args.vosk_model, busy=lambda: self.voice.speaking,
                                         on_wake=lambda: self.voice.say("Yes?"))
                print('Listening: say "Scouter, where is my phone?"', file=sys.stderr)
            except RuntimeError as e:
                print(f"[listen] {e}", file=sys.stderr)
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=self._read_stdin, daemon=True).start()
            print("Type a question (e.g. where is my phone?) and press Enter.", file=sys.stderr)
        if not args.headless:
            print("Keys: q quit, space say what you see, m mute, f faces on/off, o objects on/off", file=sys.stderr)
        self.voice.say("Scouter ready.")

        last_t = time.time()
        try:
            while True:
                frame = cam.read()
                if frame is None:
                    print("Camera stopped.", file=sys.stderr)
                    return 1
                shown = self._step(frame, show=not args.headless)
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 / max(now - last_t, 1e-3)
                last_t = now
                if args.headless:
                    continue
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                if key == ord(" "):
                    self.voice.say(self.describe_now())
                elif key == ord("m"):
                    self.voice.muted = not self.voice.muted
                elif key == ord("f"):
                    self.faces_on = not self.faces_on
                elif key == ord("o"):
                    self.objects_on = not self.objects_on
                    self.worker.results = []
        except KeyboardInterrupt:
            return 0
        finally:
            cam.close()
            cv2.destroyAllWindows()
            if self.listener:
                self.listener.close()
            self.memory.flush()
            if self.log:
                self.log.close()
