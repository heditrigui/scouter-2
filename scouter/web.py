"""A small local web dashboard: the live annotated stream, the event log, and people management.

Standard library only. Binds to 127.0.0.1 by default; anything else needs a password
(HTTP Basic auth, so use it only on a network you trust).
"""

import base64
import hmac
import ipaddress
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

from .events import parse_since


class FrameHub:
    """The newest annotated frame as JPEG, encoded only while someone is watching."""

    def __init__(self, fps: float = 10.0):
        self.interval = 1.0 / fps
        self.clients = 0
        self.jpeg = b""
        self.seq = 0
        self._last = 0.0
        self._cond = threading.Condition()

    def wanted(self) -> bool:
        return self.clients > 0 and time.monotonic() - self._last >= self.interval

    def publish(self, image) -> None:
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return
        with self._cond:
            self.jpeg, self.seq, self._last = jpg.tobytes(), self.seq + 1, time.monotonic()
            self._cond.notify_all()

    def wait(self, seq: int, timeout: float = 5.0) -> tuple[int, bytes]:
        with self._cond:
            self._cond.wait_for(lambda: self.seq != seq, timeout)
            return self.seq, self.jpeg


def is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


class Handler(BaseHTTPRequestHandler):
    app = None  # set by make_server
    password: str | None = None
    allowed_hosts: set[str] = set()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep the console for Scouter's own output
        pass

    # Security checks.

    def _authorised(self) -> bool:
        if not self.password:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
            except Exception:
                return False
            return hmac.compare_digest(given.encode(), self.password.encode())
        return False

    def _host_ok(self) -> bool:
        # Blocks DNS rebinding: a web page on another site can't reach us through a name it controls.
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return not self.allowed_hosts or host in self.allowed_hosts

    def _check(self) -> bool:
        if not self._host_ok():
            self._send(403, b"Forbidden host", "text/plain")
            return False
        if not self._authorised():
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Scouter"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        return True

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    # Routes.

    def do_GET(self):
        if not self._check():
            return
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        app = self.app
        if url.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8",
                       {"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"})
        elif url.path == "/stream.mjpg":
            self._stream()
        elif url.path == "/api/status":
            self._json({"fps": round(app.fps, 1), "status": app.status(), "last_said": app.voice.last,
                        "enroll": app.enroll_status, "log": app.log is not None})
        elif url.path == "/api/people":
            self._json([{"name": n, "samples": c} for n, c in app.people().items()])
        elif url.path == "/api/events":
            if app.log is None:
                return self._json([])
            try:
                since = parse_since(q["since"]) if q.get("since") else None
                limit = max(1, min(int(q.get("limit", 100)), 500))
            except ValueError:
                return self._json({"error": "bad since/limit"}, 400)
            events = app.log.search(q.get("q", ""), since=since, kind=q.get("kind") or None, limit=limit)
            self._json([e.as_dict() for e in events])
        elif url.path.startswith("/snapshots/"):
            p = app.log.snapshot_path(url.path[len("/snapshots/"):]) if app.log else None
            if p is None:
                return self._send(404, b"Not found", "text/plain")
            self._send(200, p.read_bytes(), "image/jpeg", {"Cache-Control": "max-age=3600"})
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if not self._check():
            return
        # A custom header can't be sent cross-site without a CORS preflight, which we never allow.
        if self.headers.get("X-Scouter") != "1":
            return self._send(403, b"Missing X-Scouter header", "text/plain")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(min(n, 10_000)) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json({"error": "bad JSON"}, 400)
        app = self.app
        path = urlparse(self.path).path
        name = str(body.get("name", "")).strip()[:60]
        if path == "/api/people/enroll":
            if not name:
                return self._json({"error": "name needed"}, 400)
            app.start_enroll(name, int(body.get("samples", 15)))
            self._json({"ok": True})
        elif path == "/api/people/rename":
            new = str(body.get("new", "")).strip()[:60]
            self._json({"ok": app.rename(name, new)})
        elif path == "/api/people/forget":
            self._json({"ok": app.forget(name)})
        elif path == "/api/ask":
            self._json({"answer": app.ask(str(body.get("question", ""))[:200])})
        else:
            self._send(404, b"Not found", "text/plain")

    def _stream(self) -> None:
        hub = self.app.hub
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        hub.clients += 1
        seq = -1
        try:
            while not self.server.stopping:
                seq, jpg = hub.wait(seq)
                if not jpg:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            hub.clients -= 1
            self.close_connection = True


def make_server(app, host: str = "127.0.0.1", port: int = 8080, password: str | None = None) -> ThreadingHTTPServer:
    if not is_loopback(host) and not password:
        raise ValueError("Serving beyond this computer needs a password (--web-password or SCOUTER_WEB_PASSWORD).")
    if app.hub is None:
        app.hub = FrameHub()
    handler = type("ScouterHandler", (Handler,), {
        "app": app, "password": password or None,
        "allowed_hosts": {"localhost", "127.0.0.1", "::1"} if is_loopback(host) else set(),
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.stopping = False
    return server


def serve(app, host: str = "127.0.0.1", port: int = 8080, password: str | None = None) -> ThreadingHTTPServer:
    server = make_server(app, host, port, password)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    shown = "localhost" if is_loopback(host) else ("this computer's address" if host in ("0.0.0.0", "::") else host)
    print(f"Dashboard: http://{shown}:{server.server_address[1]}/", flush=True)
    return server


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scouter</title>
<style>
:root{--bg:#0b0f0c;--panel:#131a15;--line:#24302a;--fg:#d8e6dc;--dim:#7f9487;--accent:#5aff78;--amber:#ffbe28;--red:#ff6b5a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:baseline;padding:12px 16px;border-bottom:1px solid var(--line)}
header b{color:var(--accent);letter-spacing:.12em}#status{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:minmax(0,3fr) minmax(280px,2fr);gap:16px;padding:16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
h2{margin:0 0 8px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
#live{width:100%;border-radius:4px;background:#000;display:block}
input,button,select{font:inherit;color:var(--fg);background:#0e1410;border:1px solid var(--line);border-radius:4px;padding:6px 8px}
button{cursor:pointer}button:hover{border-color:var(--accent)}.row{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}
.row input{flex:1;min-width:120px}#said,#answer{color:var(--accent);min-height:1.4em}
table{width:100%;border-collapse:collapse}td{padding:6px 4px;border-top:1px solid var(--line);vertical-align:middle}
td.t{color:var(--dim);white-space:nowrap;font-variant-numeric:tabular-nums}.unknown{color:var(--amber)}.alert{color:var(--red)}
img.snap{width:72px;border-radius:3px;display:block;cursor:zoom-in}#events{max-height:60vh;overflow:auto}
.p{display:flex;justify-content:space-between;align-items:center;gap:6px;padding:4px 0}.p span{flex:1}
dialog{background:var(--panel);color:var(--fg);border:1px solid var(--line);padding:8px}dialog img{max-width:90vw;max-height:80vh}
</style></head><body>
<header><b>SCOUTER</b><span id="status">connecting...</span></header>
<main>
<div>
<section><img id="live" src="/stream.mjpg" alt="Live camera with overlay"><div id="said"></div></section>
<section style="margin-top:16px"><h2>Ask</h2>
<form class="row" id="askf"><input id="q" placeholder="where is my phone? who did you see today?" autocomplete="off"><button>Ask</button></form>
<div id="answer"></div></section>
<section style="margin-top:16px"><h2>People</h2><div id="people"></div>
<form class="row" id="enf"><input id="en" placeholder="Name" maxlength="60"><button>Enroll from camera</button></form>
<div id="enst" style="color:var(--dim)"></div></section>
</div>
<section><h2>Event log</h2>
<form class="row" id="logf"><input id="lq" placeholder="search"><select id="lk"><option value="">all</option>
<option value="person">people</option><option value="unknown">unknown</option><option value="object">objects</option></select>
<select id="ls"><option value="">any time</option><option value="1h">last hour</option><option value="today">today</option><option value="7d">7 days</option></select></form>
<div id="events"><table><tbody id="ev"></tbody></table></div></section>
</main>
<dialog id="dlg"><img id="big" alt="Snapshot"></dialog>
<script>
const $=id=>document.getElementById(id);
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json','X-Scouter':'1'},body:JSON.stringify(b)}).then(r=>r.json());
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function status(){try{const s=await (await fetch('/api/status')).json();$('status').textContent=s.status.join('  ·  ');
$('said').textContent=s.last_said||'';const e=s.enroll;$('enst').textContent=e.active?`Enrolling ${e.name}: ${e.done}/${e.target} — look at the camera, one face in view`:(e.result||'');
if(!e.active&&e.result&&!status.peopleDone){status.peopleDone=true;people()}if(e.active)status.peopleDone=false}catch(x){$('status').textContent='offline'}}
async function people(){const ps=await (await fetch('/api/people')).json();$('people').innerHTML=ps.length?'':'<div style="color:var(--dim)">Nobody enrolled yet.</div>';
for(const p of ps){const d=document.createElement('div');d.className='p';d.innerHTML=`<span>${esc(p.name)} <small style="color:var(--dim)">${p.samples} samples</small></span>`;
const r=document.createElement('button');r.textContent='Rename';r.onclick=async()=>{const n=prompt('New name for '+p.name,p.name);if(n&&n!==p.name){await post('/api/people/rename',{name:p.name,new:n});people();events()}};
const f=document.createElement('button');f.textContent='Forget';f.onclick=async()=>{if(confirm('Forget '+p.name+'?')){await post('/api/people/forget',{name:p.name});people()}};
d.append(r,f);$('people').append(d)}}
async function events(){const u=new URLSearchParams({q:$('lq').value,kind:$('lk').value,since:$('ls').value,limit:150});
const es=await (await fetch('/api/events?'+u)).json();$('ev').innerHTML=es.map(e=>`<tr><td class="t">${esc(e.time.slice(5))}</td>
<td class="${e.action==='alert'?'alert':e.kind==='unknown'?'unknown':''}">${esc(e.text)}</td>
<td>${e.snapshot?`<img class="snap" loading="lazy" src="/snapshots/${encodeURI(e.snapshot)}" alt="">`:''}</td></tr>`).join('')||'<tr><td style="color:var(--dim)">No events.</td></tr>'}
$('ev').onclick=e=>{if(e.target.classList.contains('snap')){$('big').src=e.target.src;$('dlg').showModal()}};$('dlg').onclick=()=>$('dlg').close();
$('askf').onsubmit=async e=>{e.preventDefault();if(!$('q').value.trim())return;$('answer').textContent='...';$('answer').textContent=(await post('/api/ask',{question:$('q').value})).answer};
$('enf').onsubmit=async e=>{e.preventDefault();if($('en').value.trim()){await post('/api/people/enroll',{name:$('en').value});$('en').value='';status()}};
$('logf').onsubmit=e=>{e.preventDefault();events()};['lq','lk','ls'].forEach(i=>$(i).oninput=events);
status();people();events();setInterval(status,1000);setInterval(events,10000);
</script></body></html>
"""
