"""Offline speech recognition (Vosk) listening for "Scouter, <question>".

Optional: pip install vosk sounddevice, and unzip a Vosk model (e.g. vosk-model-small-en-us-0.15
from https://alphacephei.com/vosk/models) into models/.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

WAKE_WORDS = ("scouter", "scooter", "scout")  # small models often hear "scooter"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "vosk-model-small-en-us-0.15"


class WakeWord:
    """Pulls the question out of recognised text.

    "scouter where is my phone" -> "where is my phone". Saying just "Scouter" opens a
    [window]-second gap in which the next sentence is taken as the question.
    """

    def __init__(self, words=WAKE_WORDS, window: float = 6.0, clock=time.monotonic):
        self.words = words
        self.window = window
        self.clock = clock
        self._armed: float | None = None

    def feed(self, text: str) -> str | None:
        words = text.lower().split()
        now = self.clock()
        for i, w in enumerate(words):
            if w in self.words:
                rest = " ".join(words[i + 1:])
                if rest:
                    self._armed = None
                    return rest
                self._armed = now
                return None
        if self._armed is not None and words and now - self._armed <= self.window:
            self._armed = None
            return " ".join(words)
        return None

    @property
    def armed(self) -> bool:
        return self._armed is not None and self.clock() - self._armed <= self.window


class Listener:
    """Listens on the default microphone and calls on_question(text) for each question."""

    def __init__(self, on_question, model_dir: Path | None = None, on_wake=None, busy=lambda: False):
        try:
            import sounddevice as sd
            import vosk
        except ImportError as e:
            raise RuntimeError("Voice questions need: pip install vosk sounddevice") from e
        model_dir = Path(model_dir or DEFAULT_MODEL)
        if not model_dir.is_dir():
            raise RuntimeError(f"No Vosk model at {model_dir}. Download a small English model from "
                               "https://alphacephei.com/vosk/models and unzip it there (or pass --vosk-model).")
        vosk.SetLogLevel(-1)
        self._sd = sd
        self._model = vosk.Model(str(model_dir))
        self._rec = vosk.KaldiRecognizer(self._model, 16000)
        self.on_question = on_question
        self.on_wake = on_wake
        self.wake = WakeWord()
        self.busy = busy  # true while Scouter is speaking, so it doesn't hear itself
        self._audio: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _callback(self, data, frames, t, status) -> None:
        if not self.busy():
            try:
                self._audio.put_nowait(bytes(data))
            except queue.Full:
                pass

    def _run(self) -> None:
        try:
            with self._sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16", channels=1,
                                         callback=self._callback):
                while self._running:
                    try:
                        data = self._audio.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if self._rec.AcceptWaveform(data):
                        text = json.loads(self._rec.Result()).get("text", "")
                        if not text:
                            continue
                        question = self.wake.feed(text)
                        if question:
                            self.on_question(question)
                        elif self.wake.armed and self.on_wake:
                            self.on_wake()
        except Exception as e:
            print("[listen] microphone stopped:", e, file=sys.stderr)

    def close(self) -> None:
        self._running = False
