#!/usr/bin/env python3
"""Serial Monitor Web — 实时查看树莓派串口收发数据"""
import json
import time
import serial
import threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT_WEB = 8081
SERIAL_PORT = "/dev/serial0"
BAUDRATE = 115200
MAX_LINES = 200

log = deque(maxlen=MAX_LINES)
tx_log = deque(maxlen=MAX_LINES)
rx_log = deque(maxlen=MAX_LINES)
ser = None
ser_lock = threading.Lock()
running = True

def timestamp():
    return time.strftime("%H:%M:%S")

def serial_reader():
    global ser
    while running:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.3)
            print(f"[MON] Opened {SERIAL_PORT} @ {BAUDRATE}", flush=True)
            while running:
                try:
                    line = ser.readline()
                    if line:
                        text = line.decode("ascii", errors="replace").strip()
                        if text:
                            ts = timestamp()
                            entry = {"time": ts, "dir": "RX", "text": text}
                            rx_log.append(entry)
                            log.append(entry)
                            print(f"[RX {ts}] {text}", flush=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"[MON] Serial error: {e}, retry in 2s...", flush=True)
        finally:
            if ser and ser.is_open:
                ser.close()
        time.sleep(2)

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            data = {"rx": list(rx_log), "tx": list(tx_log), "all": list(log)}
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/send":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            text = body.strip()
            if text:
                with ser_lock:
                    if ser and ser.is_open:
                        try:
                            ser.write((text + "\r\n").encode("ascii"))
                            ser.flush()
                            ts = timestamp()
                            entry = {"time": ts, "dir": "TX", "text": text}
                            tx_log.append(entry)
                            log.append(entry)
                            print(f"[TX {ts}] {text}", flush=True)
                        except Exception as e:
                            print(f"[TX ERROR] {e}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        elif self.path == "/log":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                entry = json.loads(body)
                ts = timestamp()
                entry["time"] = ts
                direction = entry.get("dir", "RX")
                if direction == "TX":
                    tx_log.append(entry)
                else:
                    rx_log.append(entry)
                log.append(entry)
            except Exception as e:
                print(f"[MON] Bad /log payload: {e}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Serial Monitor — Pi MCU</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:Consolas,monospace;font-size:13px;height:100vh;display:flex;flex-direction:column}
.header{background:#161b22;padding:10px 16px;border-bottom:1px solid #30363d;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.header h1{font-size:15px;color:#58a6ff}
.status{font-size:12px;padding:3px 10px;border-radius:12px}
.status.on{background:#238636;color:#fff}
.status.off{background:#da3633;color:#fff}
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
.send-bar{background:#161b22;padding:10px 16px;border-top:1px solid #30363d;display:flex;gap:8px}
.send-bar input{flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px 12px;font-family:inherit;font-size:13px;border-radius:6px;outline:none}
.send-bar input:focus{border-color:#58a6ff}
.send-bar button{background:#238636;color:#fff;border:none;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:bold}
.send-bar button:hover{background:#2ea043}
.presets{display:flex;gap:6px}
.presets button{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;font-family:inherit}
.presets button:hover{background:#30363d}
</style>
</head>
<body>
<div class="header">
<h1>Serial Monitor — Pi MCU</h1>
<div><span class="presets">
<button onclick="send('$DONE')">$DONE</button>
<button onclick="send('#HOME')">#HOME</button>
<button onclick="send('#POSD,160,0,60,160,0,60,0')">#POSD test</button>
</span>
<span class="status on" id="status">LIVE</span></div>
</div>
<div class="main">
<div class="panel rx">
<div class="panel-header">RX (MCU -> Pi)</div>
<div class="log" id="rx-log"></div>
</div>
<div class="panel tx">
<div class="panel-header">TX (Pi -> MCU)</div>
<div class="log" id="tx-log"></div>
</div>
</div>
<div class="send-bar">
<input id="send-input" placeholder="..." onkeydown="if(event.key==='Enter')sendCurrent()">
<button onclick="sendCurrent()">Send</button>
</div>
<script>
let lastLen=0;
async function poll(){
try{let r=await fetch("/data");let d=await r.json();let t=d.all.length;if(t>lastLen){renderLog("rx-log",d.rx,"rx");renderLog("tx-log",d.tx,"tx");lastLen=t}}catch(e){}
}
function renderLog(id,entries,dir){
let el=document.getElementById(id),h="";
for(let e of entries)h+=`<div class="line"><span class="time">${e.time}</span><span class="dir ${dir}">${dir.toUpperCase()}</span><span class="text">${esc(e.text)}</span></div>`;
el.innerHTML=h;el.scrollTop=el.scrollHeight
}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function sendCurrent(){let v=document.getElementById("send-input").value;send(v);document.getElementById("send-input").value=""}
async function send(t){if(!t)return;await fetch("/send",{method:"POST",body:t})}
setInterval(poll,200);poll()
</script>
</body>
</html>"""

if __name__ == "__main__":
    print(f"=== Serial Monitor ===", flush=True)
    print(f"Web:    http://0.0.0.0:{PORT_WEB}", flush=True)
    print(f"Serial: {SERIAL_PORT} @ {BAUDRATE}", flush=True)
    print(f"Log API: http://127.0.0.1:{PORT_WEB}/log (accepts serial_protocol reports)", flush=True)

    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT_WEB), Handler)
    print(f"Ready! http://192.168.31.93:{PORT_WEB}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        running = False
        server.shutdown()