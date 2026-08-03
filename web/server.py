#!/usr/bin/env python3
"""HTTP server for the puzzle robot web interface. Dependencies injected via setup()."""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np

_ctx = {}

def setup(*, shared_state, handle_action, order_points, log_print,
          calib_file, jpeg_quality, stream_fps):
    _ctx["SharedState"] = shared_state
    _ctx["handle_action"] = handle_action
    _ctx["order_points"] = order_points
    _ctx["log_print"] = log_print
    _ctx["CALIBRATION_FILE"] = calib_file
    _ctx["JPEG_QUALITY"] = jpeg_quality
    _ctx["STREAM_FPS"] = stream_fps


HTML = """<!DOCTYPE html><html><head><title>Puzzle Recognition</title>

<meta charset=utf-8>

<style>

*{margin:0;padding:0;box-sizing:border-box}

body{background:#111;color:#fff;font:14px/1.5 monospace}

.top{display:flex;height:calc(100vh - 140px)}

.video{flex:1;display:flex;align-items:center;justify-content:center;position:relative;cursor:crosshair}

.video img{max-width:100%;max-height:100%;display:block}

.video canvas{position:absolute;top:0;left:0;pointer-events:none;z-index:10}

.crosshair-tip{position:absolute;pointer-events:none;z-index:20;color:#0f0;font:bold 12px monospace;background:rgba(0,0,0,0.8);padding:3px 7px;border-radius:4px;border:1px solid #0f0;white-space:nowrap;display:none}

.ctrl{position:fixed;bottom:0;left:0;right:0;background:rgba(0,0,0,0.95);padding:10px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center;z-index:100;border-top:2px solid #333}

.ctrl button{padding:10px 14px;font-size:13px;font-weight:bold;border:2px solid #555;border-radius:6px;cursor:pointer;color:#fff;transition:all 0.15s;min-width:55px}

.ctrl button:hover{opacity:0.85;transform:scale(1.03)}

.ctrl button:active{transform:scale(0.96)}

.ctrl .sep{width:2px;background:#444;margin:0 4px}

.btn-num{background:#607D8B;border-color:#90A4AE}

.btn-t{background:#009688;border-color:#4DB6AC}

.btn-d{background:#795548;border-color:#A1887F}

.btn-a{background:#3F51B5;border-color:#7986CB}

.btn-s{background:#FF9800;border-color:#FFB74D}

.btn-r-on{background:#4CAF50;border-color:#81C784}

.btn-r-off{background:#F44336;border-color:#E57373}

.btn-f-on{background:#E91E63;border-color:#F06292;animation:pulse 1.2s infinite}

.btn-f-off{background:#607D8B;border-color:#90A4AE}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

.mode-bar{position:fixed;bottom:88px;left:50%;transform:translateX(-50%);color:#ff0;font:bold 15px monospace;background:rgba(0,0,0,0.85);padding:6px 20px;border-radius:6px 6px 0 0;z-index:50}

.info{position:fixed;top:10px;left:10px;color:#0f0;font:14px monospace;background:rgba(0,0,0,0.75);padding:8px 14px;border-radius:6px;z-index:50;pointer-events:none}

</style></head><body>

<div class=info id=info>Loading...</div>

<div class=top><div class=video id=video_container><img id=stream src=/stream><canvas id=crosshair></canvas><div class=crosshair-tip id=coord_tip></div></div></div>

<div class=mode-bar id=mode_bar>Mode: AUTO</div>

<div class=ctrl>

<button class=btn-num onclick=act('1')>1</button>

<button class=btn-num onclick=act('2')>2</button>

<button class=btn-num onclick=act('3')>3</button>

<span class=sep></span>

<button class=btn-t onclick=act('T')>T</button>

<button class=btn-d onclick=act('D')>D</button>

<button class=btn-a onclick=act('A')>A</button>

<button class=btn-s onclick=act('S')>S</button>

<span class=sep></span>

<button class=btn-r-off id=btn_r onclick=act('R')>R:OFF</button>

<span class=sep></span>

<button class=btn-f-off id=btn_f onclick=act('F')>F:OFF</button>

</div>

<script>

function act(cmd){fetch('/action?cmd='+cmd).then(r=>r.text()).then(t=>{

 if(cmd=='M'||cmd=='0'||cmd=='1'||cmd=='2'||cmd=='3') document.getElementById('mode_bar').innerHTML=('Mode: '+t.split('->')[1])||t;

 if(cmd=='R'){var b=document.getElementById('btn_r');var on=t.includes('ON');b.className=on?'btn-r-on':'btn-r-off';b.textContent=on?'R:ON':'R:OFF';}

 if(cmd=='F'){var b=document.getElementById('btn_f');var on=t.includes('FROZEN');b.className=on?'btn-f-on':'btn-f-off';b.textContent=on?'F:FROZEN':'F:OFF';}

})}

// ---- Crosshair: mouse hover shows ARM-mm coords (A4 area only) ----
(function(){
 var img=document.getElementById('stream'),
     canvas=document.getElementById('crosshair'),
     tip=document.getElementById('coord_tip'),
     container=document.getElementById('video_container'),
     ctx=canvas.getContext('2d');

 var calib=null; // {has_calib, corners, c2w, ppm, origin_wx, origin_wy}

 // Fetch calibration data every 3s
 function fetchCalib(){
   fetch('/calib_data').then(function(r){return r.json()}).then(function(d){
     calib=d.has_calib?d:null;
   }).catch(function(){calib=null;});
 }
 fetchCalib();
 setInterval(fetchCalib,3000);

 function syncCanvas(){
   var r=img.getBoundingClientRect(),
       cr=container.getBoundingClientRect();
   var ox=r.left-cr.left, oy=r.top-cr.top;
   canvas.style.left=ox+'px';
   canvas.style.top=oy+'px';
   canvas.width=r.width;
   canvas.height=r.height;
 }
 img.addEventListener('load',syncCanvas);
 window.addEventListener('resize',syncCanvas);
 setInterval(syncCanvas,2000);

 function hideCrosshair(){
   ctx.clearRect(0,0,canvas.width,canvas.height);
   tip.style.display='none';
 }

 // Perspective transform: camera-img-px -> warp-px
 function transformPoint(x,y,matrix){
   var u=matrix[0][0]*x+matrix[0][1]*y+matrix[0][2];
   var v=matrix[1][0]*x+matrix[1][1]*y+matrix[1][2];
   var w=matrix[2][0]*x+matrix[2][1]*y+matrix[2][2];
   return [u/w, v/w];
 }

 // Warp-px -> arm-mm (matching coords.py image_to_arm)
 function warpToArm(wx,wy,ppm,ox,oy){
   return [(ox-wx)/ppm, (wy-oy)/ppm];
 }

 // Point in polygon (ray-casting)
 function pointInPolygon(px,py,poly){
   var inside=false;
   for(var i=0,j=poly.length-1;i<poly.length;j=i++){
     var xi=poly[i][0], yi=poly[i][1];
     var xj=poly[j][0], yj=poly[j][1];
     if((yi>py)!==(yj>py) && px<(xj-xi)*(py-yi)/(yj-yi)+xi) inside=!inside;
   }
   return inside;
 }

 container.addEventListener('mousemove',function(e){
   var r=img.getBoundingClientRect();
   var dx=e.clientX-r.left, dy=e.clientY-r.top;
   if(dx<0||dy<0||dx>r.width||dy>r.height){hideCrosshair();return;}

   var scaleX=img.naturalWidth/r.width, scaleY=img.naturalHeight/r.height;
   var px=dx*scaleX, py=dy*scaleY;

   // Check calibration + A4 boundary
   var insideA4=false, armX=0, armY=0;
   if(calib && calib.corners){
     insideA4=pointInPolygon(px,py,calib.corners);
     if(insideA4){
       var wp=transformPoint(px,py,calib.c2w);
       var arm=warpToArm(wp[0],wp[1],calib.ppm,calib.origin_wx,calib.origin_wy);
       armX=arm[0]; armY=arm[1];
     }
   }

   if(!insideA4){hideCrosshair();return;}

   syncCanvas();
   ctx.clearRect(0,0,canvas.width,canvas.height);

   // Dashed crosshair lines (green = valid ARM coords)
   ctx.strokeStyle='rgba(0,255,0,0.7)';
   ctx.lineWidth=1;
   ctx.setLineDash([5,5]);

   ctx.beginPath();
   ctx.moveTo(0,dy);
   ctx.lineTo(canvas.width,dy);
   ctx.stroke();

   ctx.beginPath();
   ctx.moveTo(dx,0);
   ctx.lineTo(dx,canvas.height);
   ctx.stroke();

   ctx.setLineDash([]);

   // Solid center dot
   ctx.fillStyle='#0f0';
   ctx.beginPath();
   ctx.arc(dx,dy,4,0,Math.PI*2);
   ctx.fill();

   // Coordinate tip in arm mm
   tip.textContent='('+armX.toFixed(1)+', '+armY.toFixed(1)+') mm';
   var cr=container.getBoundingClientRect();
   var tx=e.clientX-cr.left+16, ty=e.clientY-cr.top+16;
   if(tx+130>cr.width) tx=e.clientX-cr.left-140;
   if(ty+26>cr.height) ty=e.clientY-cr.top-30;
   tip.style.left=tx+'px';
   tip.style.top=ty+'px';
   tip.style.display='block';
 });

 container.addEventListener('mouseleave',hideCrosshair);
})();

setInterval(function(){fetch('/status').then(r=>r.json()).then(d=>{

 document.getElementById('info').innerHTML=(d.frozen?'[FROZEN] ':'')+'Pieces: '+d.pieces+' | Mode: '+d.mode+' | FPS: '+d.fps+' | '+d.last_action;

 document.getElementById('mode_bar').innerHTML='Mode: '+d.selected_mode + (d.recognition ? ' | REC:ON' : ' | REC:OFF');

 var b=document.getElementById('btn_r');if(d.recognition){b.className='btn-r-on';b.textContent='R:ON';}else{b.className='btn-r-off';b.textContent='R:OFF';}

 var fb=document.getElementById('btn_f');if(d.frozen){fb.className='btn-f-on';fb.textContent='F:FROZEN';}else{fb.className='btn-f-off';fb.textContent='F:OFF';}

})},1000)

</script></body></html>"""





CALIB_HTML = """<!DOCTYPE html><html><head><title>Calibrate</title>

<meta charset=utf-8>

<style>body{margin:0;background:#000;text-align:center;font:14px monospace;color:#fff}

img{max-width:100vw;max-height:85vh;cursor:crosshair}

.info{color:#0f0;padding:10px}

button{padding:10px 20px;margin:5px;font-size:16px;cursor:pointer;background:#333;color:#fff;border:2px solid #555;border-radius:4px}

#coords{color:#ff0;margin:10px}

</style></head><body><div class=info>Click 4 corners of A4 paper (auto-sorted)</div>

<div id=coords></div>

<img id=calib_img src=/raw_frame>

<div><button onclick=save()>Save</button> <button onclick=reset()>Reset</button></div>

<script>

var pts=[];

document.getElementById('calib_img').onclick=function(e){

 var r=this.getBoundingClientRect();

 var x=e.clientX-r.left, y=e.clientY-r.top;

 var sx=this.naturalWidth/r.width, sy=this.naturalHeight/r.height;

 var px=Math.round(x*sx), py=Math.round(y*sy);

 if(pts.length<4){pts.push([px,py]);

  document.getElementById('coords').innerHTML='Pt'+pts.length+':('+px+','+py+')|'+JSON.stringify(pts);}

 if(pts.length==4) document.getElementById('coords').innerHTML='Ready: '+JSON.stringify(pts);

};

function save(){

 if(pts.length!=4){alert('Need 4 corners');return;}

 fetch('/save_calib',{method:'POST',body:JSON.stringify(pts)}).then(r=>r.text()).then(t=>alert(t));

}

function reset(){pts=[];document.getElementById('coords').innerHTML='';}

</script></body></html>"""





class Handler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/":

            self.send_response(200)

            self.send_header("Content-type", "text/html; charset=utf-8")

            self.end_headers()

            self.wfile.write(HTML.encode())

        elif self.path.startswith("/action"):

            cmd = "auto"

            if "?" in self.path:

                qs = self.path.split("?", 1)[1]

                for kv in qs.split("&"):

                    if "=" in kv and kv.split("=")[0] == "cmd":

                        cmd = kv.split("=")[1]

            self.send_response(200)

            self.send_header("Content-type", "text/plain; charset=utf-8")

            self.end_headers()

            self.wfile.write(_ctx['handle_action'](cmd).encode())

        elif self.path == "/calibrate":

            self.send_response(200)

            self.send_header("Content-type", "text/html; charset=utf-8")

            self.end_headers()

            self.wfile.write(CALIB_HTML.encode())

        elif self.path == "/raw_frame":

            self.send_response(200)

            self.send_header("Content-type", "image/jpeg")

            self.end_headers()

            with _ctx['SharedState'].lock:

                f = _ctx['SharedState'].raw_frame.copy() if _ctx['SharedState'].raw_frame is not None else None

            if f is not None:

                _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, _ctx['JPEG_QUALITY']])

                self.wfile.write(buf.tobytes())

        elif self.path == "/stream":

            self.send_response(200)

            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")

            self.send_header("Cache-Control", "no-cache")

            self.end_headers()

            while True:

                with _ctx['SharedState'].lock:

                    f = _ctx['SharedState'].frame.copy() if _ctx['SharedState'].frame is not None else None

                if f is not None:

                    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, _ctx['JPEG_QUALITY']])

                    if ok:

                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

                time.sleep(1.0 / _ctx['STREAM_FPS'])

        elif self.path == "/freeze_data":

            self.send_response(200)

            self.send_header("Content-type", "application/json")

            self.send_header("Access-Control-Allow-Origin", "*")

            self.end_headers()

            with _ctx['SharedState'].lock:

                fd = _ctx['SharedState'].freeze_data if _ctx['SharedState'].freeze_data is not None else {"frozen": False, "pieces": [], "message": "No freeze data available"}

            self.wfile.write(json.dumps(fd, default=str).encode())

        elif self.path == "/calib_data":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with _ctx['SharedState'].lock:
                cd = _ctx['SharedState'].calib_data if _ctx['SharedState'].calib_data is not None else {"has_calib": False}
            self.wfile.write(json.dumps(cd, default=str).encode())

        elif self.path.startswith("/restart"):

            self.send_response(200)

            self.send_header("Content-type", "text/plain; charset=utf-8")

            self.end_headers()

            self.wfile.write(b"RESTARTING...")

            import os, sys, time

            def _do_restart():

                time.sleep(0.5)

                os.execv(sys.executable, [sys.executable] + sys.argv)

            import threading

            threading.Thread(target=_do_restart, daemon=True).start()

        elif self.path == "/status":

            self.send_response(200)

            self.send_header("Content-type", "application/json")

            self.end_headers()

            with _ctx['SharedState'].lock:

                s = json.dumps(_ctx['SharedState'].status)

            self.wfile.write(s.encode())


    def do_POST(self):

        if self.path == "/save_calib":

            length = int(self.headers.get('Content-Length', 0))

            body = self.rfile.read(length)

            try:

                pts_raw = json.loads(body)

                if len(pts_raw) != 4:

                    self.send_error(400, "Need 4 corners")

                    return

                pts_sorted = _ctx['order_points'](np.array(pts_raw, dtype=np.float32)).tolist()

                with open(str(_ctx['CALIBRATION_FILE']), 'w') as f:

                    json.dump({"corners": pts_sorted}, f, indent=2)

                _ctx['log_print'](f"Calibration saved (sorted)")

                self.send_response(200)

                self.end_headers()

                self.wfile.write(b"OK")

            except Exception as e:

                self.send_error(400, str(e))




class TS(ThreadingMixIn, HTTPServer):

    allow_reuse_address = True

    daemon_threads = True


