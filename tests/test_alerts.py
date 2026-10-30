import json
import threading
from datetime import datetime
from datetime import time as dtime
from http.server import BaseHTTPRequestHandler, HTTPServer

from scouter.alerts import AlertRule, Notifier, in_quiet, multipart, parse_quiet


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_quiet_hours_wrap_midnight():
    q = parse_quiet("23:00-07:00")
    assert in_quiet(q, dtime(23, 30)) and in_quiet(q, dtime(3, 0))
    assert not in_quiet(q, dtime(7, 0)) and not in_quiet(q, dtime(12, 0))
    day = parse_quiet("08:00-18:00")
    assert in_quiet(day, dtime(12, 0)) and not in_quiet(day, dtime(20, 0))
    assert not in_quiet(None, dtime(12, 0))


def test_alert_after_n_seconds_then_cooldown():
    clock = Clock()
    rule = AlertRule(after=10, cooldown=300, clock=clock, wall=lambda: datetime(2026, 10, 30, 12, 0))
    assert not rule.update(1)
    clock.t = 9
    assert not rule.update(1)
    clock.t = 10
    assert rule.update(1)
    clock.t = 20
    assert not rule.update(1)  # cooling down
    clock.t = 25
    assert not rule.update(0)  # they left
    clock.t = 400
    assert not rule.update(1)  # back, but the timer starts over
    clock.t = 410
    assert rule.update(1)


def test_no_alert_during_quiet_hours():
    clock = Clock()
    rule = AlertRule(after=1, quiet=parse_quiet("23:00-07:00"), clock=clock, wall=lambda: datetime(2026, 10, 30, 2, 0))
    rule.update(1)
    clock.t = 5
    assert not rule.update(1)


def test_multipart_body():
    body, ctype = multipart({"chat_id": "42"}, {"photo": ("a.jpg", b"\xff\xd8jpeg", "image/jpeg")})
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}\r\n".encode()) and body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="chat_id"\r\n\r\n42\r\n' in body and b"\xff\xd8jpeg" in body


def test_webhook_posts_json():
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.handle_request, daemon=True).start()
    n = Notifier(desktop=False, webhook=f"http://127.0.0.1:{server.server_address[1]}/hook")
    n.send("Someone I don't know", b"jpg", when=1000.0).join(5)
    server.server_close()
    assert got and got[0]["message"] == "Someone I don't know" and got[0]["snapshot_jpeg_base64"] == "anBn"
