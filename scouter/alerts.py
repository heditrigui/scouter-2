"""Alerts when someone unknown stays in view: a desktop notification, and optionally Telegram or a webhook.

Nothing leaves the machine unless a Telegram bot or a webhook URL is configured.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime
from datetime import time as dtime


def parse_quiet(text: str | None) -> tuple[dtime, dtime] | None:
    """'23:00-07:00' -> (23:00, 07:00). The range may wrap past midnight."""
    if not text:
        return None
    start, end = (dtime.fromisoformat(p.strip()) for p in text.split("-"))
    return start, end


def in_quiet(quiet: tuple[dtime, dtime] | None, t: dtime) -> bool:
    if not quiet:
        return False
    start, end = quiet
    return start <= t < end if start <= end else (t >= start or t < end)


class AlertRule:
    """Fires when an unknown face has been in view for [after] seconds without a break.

    After firing it stays silent for [cooldown] seconds. During [quiet] hours it never fires
    (the event log still records the visit).
    """

    def __init__(self, after: float = 10.0, cooldown: float = 300.0, quiet: tuple[dtime, dtime] | None = None,
                 clock=time.monotonic, wall=datetime.now):
        self.after = after
        self.cooldown = cooldown
        self.quiet = quiet
        self.clock = clock
        self.wall = wall
        self._since: float | None = None
        self._fired: float | None = None

    def update(self, unknown_in_view: int) -> bool:
        now = self.clock()
        if not unknown_in_view:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        if now - self._since < self.after:
            return False
        if self._fired is not None and now - self._fired < self.cooldown:
            return False
        if in_quiet(self.quiet, self.wall().time()):
            return False
        self._fired = now
        return True


def desktop_notify(title: str, message: str) -> None:
    """Best effort: plyer if installed, else the platform's own tool. Never raises."""
    try:
        from plyer import notification  # optional

        notification.notify(title=title, message=message, app_name="Scouter", timeout=10)
        return
    except Exception:
        pass
    try:
        if sys.platform == "darwin":
            script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
            subprocess.Popen(["osascript", "-e", script])
        elif os.name == "nt":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n = New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon = [System.Drawing.SystemIcons]::Warning; $n.Visible = $true;"
                "$n.ShowBalloonTip(10000, $env:SCOUTER_TITLE, $env:SCOUTER_MSG, 'Warning');"
                "Start-Sleep -Seconds 11; $n.Dispose()"
            )
            env = {**os.environ, "SCOUTER_TITLE": title, "SCOUTER_MSG": message}
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps], env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        elif shutil.which("notify-send"):
            subprocess.Popen(["notify-send", title, message])
    except Exception as e:
        print("[alert] desktop notification failed:", e, file=sys.stderr)


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """A multipart/form-data body and its content type."""
    boundary = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (filename, data, ctype) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{filename}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class Notifier:
    """Sends an alert everywhere configured, on a background thread so the camera never waits."""

    def __init__(self, desktop: bool = True, telegram_token: str | None = None, telegram_chat: str | None = None,
                 webhook: str | None = None, timeout: float = 10.0):
        self.desktop = desktop
        self.telegram_token = telegram_token
        self.telegram_chat = telegram_chat
        self.webhook = webhook
        self.timeout = timeout

    def send(self, message: str, jpeg: bytes | None = None, when: float | None = None) -> threading.Thread:
        t = threading.Thread(target=self._send, args=(message, jpeg, when or time.time()), daemon=True)
        t.start()
        return t

    def _send(self, message: str, jpeg: bytes | None, when: float) -> None:
        if self.desktop:
            desktop_notify("Scouter", message)
        if self.telegram_token and self.telegram_chat:
            self._try("Telegram", self._telegram, message, jpeg)
        if self.webhook:
            self._try("webhook", self._webhook, message, jpeg, when)

    def _try(self, what, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as e:
            print(f"[alert] {what} failed: {e}", file=sys.stderr)

    def _telegram(self, message: str, jpeg: bytes | None) -> None:
        base = f"https://api.telegram.org/bot{self.telegram_token}/"
        if jpeg:
            body, ctype = multipart({"chat_id": self.telegram_chat, "caption": message},
                                    {"photo": ("scouter.jpg", jpeg, "image/jpeg")})
            req = urllib.request.Request(base + "sendPhoto", body, {"Content-Type": ctype})
        else:
            body = json.dumps({"chat_id": self.telegram_chat, "text": message}).encode()
            req = urllib.request.Request(base + "sendMessage", body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self.timeout).close()

    def webhook_payload(self, message: str, jpeg: bytes | None, when: float) -> dict:
        import base64

        return {"source": "scouter", "event": "unknown_person", "message": message, "time": when,
                "iso_time": datetime.fromtimestamp(when).isoformat(timespec="seconds"),
                "snapshot_jpeg_base64": base64.b64encode(jpeg).decode() if jpeg else None}

    def _webhook(self, message: str, jpeg: bytes | None, when: float) -> None:
        body = json.dumps(self.webhook_payload(message, jpeg, when)).encode()
        req = urllib.request.Request(self.webhook, body, {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=self.timeout).close()
