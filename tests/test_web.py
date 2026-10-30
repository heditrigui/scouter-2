import base64
import http.client
import json
import threading

import numpy as np
import pytest

from scouter.events import EventLog
from scouter.web import make_server


class Voice:
    last = "I see a cup."


class FakeApp:
    def __init__(self, log):
        self.log, self.hub, self.fps, self.voice = log, None, 12.0, Voice()
        self.known = {"Alex": 3}
        self.enroll_status = {"active": False, "result": ""}
        self.enrolled = None

    def status(self):
        return ["FACES ON"]

    def people(self):
        return dict(self.known)

    def rename(self, old, new):
        return self.known.pop(old, None) is not None and not self.known.update({new: 3})

    def forget(self, name):
        return self.known.pop(name, None) is not None

    def start_enroll(self, name, samples=15):
        self.enrolled = name

    def ask(self, q):
        return f"you asked: {q}"


@pytest.fixture
def web(tmp_path):
    log = EventLog(tmp_path / "log.db")
    log.add("arrive", "person", "Alex", np.zeros((48, 64, 3), np.uint8))
    servers = []

    def start(host="127.0.0.1", password=None):
        app = FakeApp(log)
        server = make_server(app, host, 0, password)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return app, server.server_address[1]

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()
    log.close()


def request(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, json.dumps(body) if body is not None else None, headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


POST = {"Content-Type": "application/json", "X-Scouter": "1"}


def test_page_people_and_events(web):
    app, port = web()
    status, page = request(port, "GET", "/")
    assert status == 200 and b"/stream.mjpg" in page
    assert json.loads(request(port, "GET", "/api/people")[1]) == [{"name": "Alex", "samples": 3}]
    events = json.loads(request(port, "GET", "/api/events?q=alex")[1])
    assert events[0]["text"] == "Alex arrived"
    status, jpg = request(port, "GET", "/snapshots/" + events[0]["snapshot"])
    assert status == 200 and jpg[:2] == b"\xff\xd8"
    assert request(port, "GET", "/snapshots/../log.db")[0] == 404
    assert request(port, "GET", "/snapshots/..%2Flog.db")[0] == 404


def test_people_management_and_ask(web):
    app, port = web()
    assert json.loads(request(port, "POST", "/api/people/rename", {"name": "Alex", "new": "Sam"}, POST)[1]) == {"ok": True}
    assert app.known == {"Sam": 3}
    request(port, "POST", "/api/people/enroll", {"name": "Robin"}, POST)
    assert app.enrolled == "Robin"
    assert json.loads(request(port, "POST", "/api/ask", {"question": "where is my phone"}, POST)[1]) == \
        {"answer": "you asked: where is my phone"}
    assert json.loads(request(port, "POST", "/api/people/forget", {"name": "Sam"}, POST)[1]) == {"ok": True}


def test_cross_site_posts_and_foreign_hosts_are_refused(web):
    app, port = web()
    # A form on another website can't add the custom header.
    assert request(port, "POST", "/api/people/forget", {"name": "Alex"}, {"Content-Type": "application/json"})[0] == 403
    assert app.known == {"Alex": 3}
    # DNS rebinding: a request that reaches us under someone else's host name.
    assert request(port, "GET", "/api/people", headers={"Host": "evil.example:8080"})[0] == 403


def test_lan_needs_a_password(web):
    with pytest.raises(ValueError):
        web(host="0.0.0.0")
    app, port = web(host="0.0.0.0", password="s3cret")
    assert request(port, "GET", "/api/people")[0] == 401
    bad = {"Authorization": "Basic " + base64.b64encode(b"x:wrong").decode()}
    assert request(port, "GET", "/api/people", headers=bad)[0] == 401
    good = {"Authorization": "Basic " + base64.b64encode(b"scouter:s3cret").decode()}
    assert request(port, "GET", "/api/people", headers=good)[0] == 200


def test_stream_sends_jpeg_frames(web):
    app, port = web()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/stream.mjpg")
    r = c.getresponse()
    assert r.getheader("Content-Type").startswith("multipart/x-mixed-replace")
    for _ in range(50):  # wait until the handler has registered as a client
        if app.hub.clients:
            break
        threading.Event().wait(0.02)
    assert app.hub.wanted()
    app.hub.publish(np.zeros((48, 64, 3), np.uint8))
    assert r.readline() == b"--frame\r\n"
    c.close()
