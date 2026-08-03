#!/usr/bin/env python3
"""Serial Monitor Web — 实时查看树莓派串口收发数据

Architecture (v3 — HTTP-only log aggregator):
  The monitor itself does NOT open any serial port.  All TX/RX data is
  posted to /log by the process that owns the arm UART (serial_protocol.py).
  This avoids the dual-process UART conflict on Pi 5.

  Bind: 127.0.0.1 — only accessible from the Pi itself (use SSH tunnel
  for remote access).
"""

import json
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT_WEB = 8081
BIND_HOST = "0.0.0.0"           # all interfaces (LAN access enabled)
MAX_LINES = 200
MAX_ENTRY_BYTES = 512            # per-entry text cap to prevent memory DoS

log = deque(maxlen=MAX_LINES)
tx_log = deque(maxlen=MAX_LINES)
rx_log = deque(maxlen=MAX_LINES)
_version = 0


def _timestamp():
    return time.strftime("%H:%M:%S")


def _add_entry(direction: str, text: str) -> dict:
    """Add a validated log entry. Returns the entry dict."""
    ts = _timestamp()
    # Truncate oversized text
    if len(text) > MAX_ENTRY_BYTES:
        text = text[:MAX_ENTRY_BYTES - 3] + "..."
    entry = {"time": ts, "dir": direction, "text": text}
    if direction == "TX":
        tx_log.append(entry)
    else:
        rx_log.append(entry)
    log.append(entry)
    global _version; _version += 1
    return entry


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._serve_page()
        elif self.path == "/data":
            self._serve_data()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/log":
            self._handle_log()
        elif self.path == "/clear":
            self._handle_clear()
        else:
            self.send_error(404)

    def _handle_clear(self):
        """Clear all log deques."""
        global _version
        log.clear()
        tx_log.clear()
        rx_log.clear()
        _version += 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _serve_page(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def _serve_data(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        data = {"rx": list(rx_log), "tx": list(tx_log), "all": list(log), "version": _version}
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _handle_log(self):
        """Accept log entries from the arm-protocol owner process.

        Validates: direction is TX or RX, text is a non-empty string,
        text does not exceed MAX_ENTRY_BYTES.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            self.send_error(400)
            return

        if length <= 0 or length > 65536:
            self.send_error(400)
            return

        try:
            body = self.rfile.read(length).decode("utf-8")
            entry = json.loads(body)
        except Exception:
            self.send_error(400)
            return

        direction = entry.get("dir", "RX")
        if direction not in ("TX", "RX"):
            direction = "RX"
        text = str(entry.get("text", ""))
        if not text:
            self.send_error(400)
            return

        _add_entry(direction, text)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        pass  # suppress access logs


HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Serial Monitor — Pi MCU</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:Consolas,monospace;font-size:13px;height:100vh;display:flex;flex-direction:column}
.header{background:#161b22;padding:10px 16px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:15px;color:#58a6ff}
.status{font-size:12px;padding:3px 10px;border-radius:12px}
.status.on{background:#238636;color:#fff}
.main{flex:1;display:flex;overflow:hidden}
.panel{flex:1;border-right:1px solid #30363d;display:flex;flex-direction:column}
.panel:last-child{border-right:none}
.panel-header{background:#161b22;padding:8px 12px;font-size:12px;font-weight:bold;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #30363d}
.rx .panel-header{color:#3fb950}
.tx .panel-header{color:#f0883e}
.log{flex:1;overflow-y:auto;padding:6px 0}
.line{padding:3px 12px;white-space:pre-wrap;word-break:break-all;display:flex;gap:8px}
.line:hover{background:#1c2129}
.time{color:#484f58;flex-shrink:0;font-size:11px;min-width:70px}
.dir{flex-shrink:0;font-size:11px;min-width:28px;font-weight:bold}
.dir.rx{color:#3fb950}
.dir.tx{color:#f0883e}
.footer{background:#161b22;padding:8px 16px;border-top:1px solid #30363d;font-size:11px;color:#484f58;text-align:center}
</style>
</head>
<body>
<div class="header">
<h1>Serial Monitor — Pi MCU</h1>
<span class="status on" id="status">LIVE</span>
<button onclick="clearLogs()" style="background:#30363d;color:#c9d1d9;border:1px solid #484f58;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px">Clear</button>
</div>
<div class="main">
<div class="panel rx">
<div class="panel-header">RX (Arm -> Pi)</div>
<div class="log" id="rx-log"></div>
</div>
<div class="panel tx">
<div class="panel-header">TX (Pi -> Arm)</div>
<div class="log" id="tx-log"></div>
</div>
</div>
<div class="footer">
  Read-only log aggregator — serial port owned by the main app process
</div>
<script>
let lastVer = -1;
async function poll() {
  try {
    let r = await fetch("/data");
    let d = await r.json();
    if (d.version !== lastVer) {
      renderLog("rx-log", d.rx, "rx");
      renderLog("tx-log", d.tx, "tx");
      lastVer = d.version;
    }
  } catch(e) {}
}
function renderLog(id, entries, dir) {
  let el = document.getElementById(id), h = "";
  for (let e of entries) {
    h += '<div class="line"><span class="time">' + esc(e.time) + '</span>' +
         '<span class="dir ' + dir + '">' + dir.toUpperCase() + '</span>' +
         '<span class="text">' + esc(e.text) + '</span></div>';
  }
  el.innerHTML = h;
  el.scrollTop = el.scrollHeight;
}
function esc(s) {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
async function clearLogs() {
  await fetch("/clear", {method:"POST"});
}
setInterval(poll, 200);
poll();
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"=== Serial Monitor (HTTP-only) ===", flush=True)
    print(f"Web:  http://{BIND_HOST}:{PORT_WEB}", flush=True)
    print(f"Log:  POST http://{BIND_HOST}:{PORT_WEB}/log", flush=True)
    print(f"Data: GET  http://{BIND_HOST}:{PORT_WEB}/data", flush=True)

    server = ThreadingHTTPServer((BIND_HOST, PORT_WEB), Handler)
    print(f"Ready!", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
