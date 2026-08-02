#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版拼图 Web 界面 — 无视频流，用 Canvas 线条/点绘制
======================================================
数据来源: 串口 /dev/ttyAMA2 (GPIO4/5) 接收 + HTTP POST
坐标: arm-mm 坐标系 (原点 A4右下+75mm, X←, Y↓)
端口: 8080 (默认, 可通过 --port 修改)

用法:
  python3 simple_puzzle_web.py               # 默认端口 8080
  python3 simple_puzzle_web.py --port 8090   # 指定端口
"""

import json
import math
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from typing import Any, Optional

# ── 串口 (可选) ──
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── 常量 (与 pi_stream_puzzle_v2.py 一致) ──
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
WARP_WIDTH = 840
WARP_HEIGHT = 1188
PIXELS_PER_MM = 4.0
ORIGIN_WX = 839.0
ORIGIN_WY = 300.0

SERIAL_PORT = "/dev/ttyAMA2"
SERIAL_BAUD = 9600
FRAME_END = b"\xff\xff\xff"

# ── 颜色调色板 ──
PIECE_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8C42", "#98D8C8", "#F7DC6F", "#BB8FCE",
]
TARGET_COLORS = [
    "#00FF88", "#00D4FF", "#FFB800", "#FF6BFF",
    "#00FFCC", "#66B2FF", "#FFDD44", "#CC88FF",
]

# ── 共享状态 ──
STATE_LOCK = threading.Lock()
puzzle_state: dict[str, Any] = {
    "pieces": [],
    "plan": [],
    "pickup_order": [],
    "assembly_order": [],
    "solve_info": {},
    "source": "等待数据...",
    "updated": 0.0,
    "serial_rx": [],
}


# ═══════════════════════════════════════════════════════════
# 串口监听
# ═══════════════════════════════════════════════════════════

def serial_listener():
    """后台线程: 持续监听串口接收数据, 更新 puzzle_state."""
    if not HAS_SERIAL:
        print("[SERIAL] pyserial 未安装, 跳过串口监听")
        return

    print(f"[SERIAL] 启动监听 {SERIAL_PORT} @ {SERIAL_BAUD}")
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.5)
            print(f"[SERIAL] 已连接 {SERIAL_PORT}")
            buf = b""

            while True:
                try:
                    chunk = ser.read(ser.in_waiting or 1)
                    if not chunk:
                        time.sleep(0.05)
                        continue
                    buf += chunk

                    while True:
                        idx = buf.find(b"\xff\xff\xff")
                        if idx < 0:
                            idx = buf.find(b"\r\n")
                        if idx < 0:
                            break
                        sep_len = 3 if buf[idx:idx+3] == b"\xff\xff\xff" else 2
                        line_raw = buf[:idx]
                        buf = buf[idx + sep_len:]
                        try:
                            text = line_raw.decode("gbk", errors="replace").strip()
                        except Exception:
                            text = line_raw.decode("ascii", errors="replace").strip()
                        if text:
                            _handle_serial_line(text)
                except (OSError, serial.SerialException):
                    break
                except Exception:
                    time.sleep(0.1)

            ser.close()
        except Exception as e:
            print(f"[SERIAL] 错误: {e}, 2秒后重连...")
        time.sleep(2)


def _handle_serial_line(text: str):
    """解析串口收到的数据行。"""
    now = time.time()
    with STATE_LOCK:
        puzzle_state["serial_rx"].append({
            "time": time.strftime("%H:%M:%S"),
            "text": text,
        })
        if len(puzzle_state["serial_rx"]) > 50:
            puzzle_state["serial_rx"] = puzzle_state["serial_rx"][-50:]

    print(f"[SERIAL RX] {text}")

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "pieces" in data:
            _update_state_from_dict(data, f"串口 {time.strftime('%H:%M:%S')}")
    except (json.JSONDecodeError, ValueError):
        pass


def _update_state_from_dict(data: dict, source: str):
    """用字典数据更新共享状态。"""
    with STATE_LOCK:
        puzzle_state["pieces"] = data.get("pieces", [])
        puzzle_state["plan"] = data.get("plan", [])
        puzzle_state["pickup_order"] = data.get("pickup_order", [])
        puzzle_state["assembly_order"] = data.get("assembly_order", [])
        puzzle_state["solve_info"] = data.get("solve_info", {})
        puzzle_state["source"] = source
        puzzle_state["updated"] = time.time()
    print(f"[STATE] 更新来自 {source}: {len(puzzle_state['pieces'])} 块, {len(puzzle_state['plan'])} 个方案")


# ═══════════════════════════════════════════════════════════
# HTTP 服务器
# ═══════════════════════════════════════════════════════════

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>拼图状态板</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font:13px/1.5 'Consolas','Microsoft YaHei',monospace;
  display:flex;flex-direction:column;height:100vh;overflow:hidden}
.header{background:#161b22;padding:8px 16px;border-bottom:1px solid #30363d;
  display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:15px;color:#58a6ff}
.status{font-size:12px;display:flex;gap:16px;align-items:center}
.status .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.status .live{background:#3fb950}
.status .stale{background:#f0883e}
.status .dead{background:#da3633}
.main{flex:1;display:flex;overflow:hidden}
.canvas-wrap{flex:1;position:relative;background:#0d1117;
  display:flex;align-items:center;justify-content:center}
.canvas-wrap canvas{display:block}
.coord-tip{position:absolute;pointer-events:none;z-index:20;
  color:#0f0;font:bold 12px 'Consolas',monospace;
  background:rgba(0,0,0,0.85);padding:3px 8px;
  border-radius:4px;border:1px solid #0f0;display:none}
.panel{width:320px;background:#161b22;border-left:1px solid #30363d;
  display:flex;flex-direction:column;overflow-y:auto}
.panel-section{border-bottom:1px solid #30363d;padding:10px 12px}
.panel-section h3{font-size:12px;color:#8b949e;text-transform:uppercase;
  letter-spacing:1px;margin-bottom:6px}
.piece-row{display:flex;gap:8px;padding:4px 0;align-items:center;font-size:12px}
.piece-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}
.piece-info{flex:1}
.coord{color:#8b949e;font-size:11px}
.rx-log{max-height:200px;overflow-y:auto;font-size:11px}
.rx-line{padding:2px 0;border-bottom:1px solid #21262d;display:flex;gap:8px}
.rx-time{color:#484f58;flex-shrink:0}
.rx-text{color:#c9d1d9;word-break:break-all}
.empty{color:#484f58;text-align:center;padding:20px;font-style:italic}
.footer{background:#161b22;padding:8px 16px;border-top:1px solid #30363d;
  display:flex;gap:8px;align-items:center;font-size:12px}
.footer input{flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;
  padding:6px 10px;font:inherit;border-radius:4px}
.footer input:focus{border-color:#58a6ff;outline:none}
.footer button{background:#238636;color:#fff;border:none;padding:6px 14px;
  border-radius:4px;cursor:pointer;font:inherit;font-weight:bold}
.footer button:hover{opacity:0.85}
.footer button.ac{background:#1f6feb}
.footer button.danger{background:#da3633}
</style>
</head>
<body>

<div class="header">
  <h1>拼图状态板 — Arm-mm 坐标系</h1>
  <div class="status">
    <span>更新: <span id="src">--</span></span>
    <span class="dot" id="dot"></span>
    <span id="age">--</span>
  </div>
</div>

<div class="main">
  <div class="canvas-wrap" id="wrap">
    <canvas id="cvs"></canvas>
    <div class="coord-tip" id="tip"></div>
  </div>

  <div class="panel">
    <div class="panel-section">
      <h3>拼图块 Pick 坐标</h3>
      <div id="piece-list"><span class="empty">无数据</span></div>
    </div>
    <div class="panel-section">
      <h3>组装方案 Place 坐标</h3>
      <div id="plan-list"><span class="empty">无数据</span></div>
    </div>
    <div class="panel-section">
      <h3>串口日志</h3>
      <div class="rx-log" id="rx-log"><span class="empty">无数据</span></div>
    </div>
  </div>
</div>

<div class="footer">
  <span>手动输入 JSON:</span>
  <input id="json-input" placeholder='{"pieces":[{"id":"p1","pick_mm":[10,20],"place_mm":[100,150],...}]}'>
  <button class="ac" onclick="sendJSON()">发送</button>
  <button onclick="loadDemo()">演示数据</button>
  <button class="danger" onclick="clearAll()">清除</button>
  <span style="color:#8b949e;margin-left:auto" id="poll-status"></span>
</div>

<script>
// ═══ 颜色常量 ═══
var PIECE_COLORS = ["#FF6B6B","#4ECDC4","#FFE66D","#A8E6CF","#FF8C42","#98D8C8","#F7DC6F","#BB8FCE"];
var TARGET_COLORS = ["#00FF88","#00D4FF","#FFB800","#FF6BFF","#00FFCC","#66B2FF","#FFDD44","#CC88FF"];

// ═══ 坐标转换 ═══
var SCALE = 3.2;
var OX = 10;
var OY, CANVAS_W, CANVAS_H;

function armToCanvas(ax, ay) {
  return {
    x: CANVAS_W - OX - ax * SCALE,
    y: OY + ay * SCALE
  };
}

var state = {pieces:[], plan:[], pickup_order:[], assembly_order:[], solve_info:{}, serial_rx:[], source:'', updated:0};

async function fetchState() {
  try {
    var r = await fetch('/state');
    var d = await r.json();
    state = d;
    draw();
    updatePanel();
    document.getElementById('poll-status').textContent = '✓';
  } catch(e) {
    document.getElementById('poll-status').textContent = '✗';
  }
}

var canvas, ctx;
function initCanvas() {
  canvas = document.getElementById('cvs');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  setupCrosshair();
}

function resizeCanvas() {
  var wrap = document.getElementById('wrap');
  canvas.width = wrap.clientWidth - 4;
  canvas.height = wrap.clientHeight - 4;
  CANVAS_W = canvas.width;
  CANVAS_H = canvas.height;
  OY = CANVAS_H - 240;
  draw();
}

function draw() {
  if (!ctx) return;
  var W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  var o = armToCanvas(0, 0);
  var tl = armToCanvas(210, -75);
  var tr = armToCanvas(0, -75);
  var bl = armToCanvas(210, 222);
  var br = armToCanvas(0, 222);

  // A4 纸区域
  ctx.fillStyle = 'rgba(22,27,34,0.8)';
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(tl.x, tl.y); ctx.lineTo(tr.x, tr.y);
  ctx.lineTo(br.x, br.y); ctx.lineTo(bl.x, bl.y);
  ctx.closePath();
  ctx.fill(); ctx.stroke();

  // 装配目标区
  var zoneY = 73.5;
  var zoneTop = armToCanvas(0, zoneY);
  var zoneBot = armToCanvas(210, 222);
  ctx.fillStyle = 'rgba(200,150,50,0.12)';
  ctx.strokeStyle = 'rgba(255,200,0,0.5)';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.rect(zoneTop.x, zoneTop.y, zoneBot.x - zoneTop.x, zoneBot.y - zoneTop.y);
  ctx.fill(); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = 'rgba(255,200,0,0.7)';
  ctx.font = '11px Consolas,monospace';
  ctx.textAlign = 'center';
  ctx.fillText('ASSEMBLY ZONE', (zoneTop.x+zoneBot.x)/2, (zoneTop.y+zoneBot.y)/2);

  // X 轴
  var xEnd = armToCanvas(210, 0);
  ctx.strokeStyle = '#F44336'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(xEnd.x, xEnd.y); ctx.stroke();
  var ax1 = armToCanvas(204, -5), ax2 = armToCanvas(204, 5);
  ctx.beginPath(); ctx.moveTo(xEnd.x, xEnd.y); ctx.lineTo(ax1.x, ax1.y);
  ctx.moveTo(xEnd.x, xEnd.y); ctx.lineTo(ax2.x, ax2.y); ctx.stroke();
  ctx.fillStyle = '#F44336'; ctx.font = 'bold 14px Consolas,monospace'; ctx.textAlign = 'left';
  ctx.fillText('X', xEnd.x+4, xEnd.y-4);

  // Y 轴
  var yEnd = armToCanvas(0, 222);
  ctx.strokeStyle = '#4CAF50';
  ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(yEnd.x, yEnd.y); ctx.stroke();
  var ay1 = armToCanvas(-5, 216), ay2 = armToCanvas(5, 216);
  ctx.beginPath(); ctx.moveTo(yEnd.x, yEnd.y); ctx.lineTo(ay1.x, ay1.y);
  ctx.moveTo(yEnd.x, yEnd.y); ctx.lineTo(ay2.x, ay2.y); ctx.stroke();
  ctx.fillStyle = '#4CAF50';
  ctx.fillText('Y', yEnd.x+4, yEnd.y+4);

  // 刻度
  ctx.textAlign = 'center';
  for (var x = 20; x <= 200; x += 20) {
    var p = armToCanvas(x, 0), pu = armToCanvas(x, -3), pd = armToCanvas(x, 3);
    ctx.strokeStyle = '#F44336'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pu.x, pu.y); ctx.lineTo(pd.x, pd.y); ctx.stroke();
    var lbl = armToCanvas(x, -10);
    ctx.fillStyle = '#F44336'; ctx.font = '9px Consolas,monospace';
    ctx.fillText(x, lbl.x, lbl.y);
  }
  for (var y = 0; y <= 220; y += 20) {
    var p = armToCanvas(0, y), pl = armToCanvas(-3, y), pr = armToCanvas(3, y);
    ctx.strokeStyle = '#4CAF50'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pl.x, pl.y); ctx.lineTo(pr.x, pr.y); ctx.stroke();
    var lbl = armToCanvas(-8, y);
    ctx.fillStyle = '#4CAF50'; ctx.font = '9px Consolas,monospace'; ctx.textAlign = 'right';
    ctx.fillText(y, lbl.x, lbl.y);
  }

  // 原点标记
  ctx.fillStyle = '#FFEB3B';
  ctx.beginPath(); ctx.arc(o.x, o.y, 6, 0, Math.PI*2); ctx.fill();
  ctx.strokeStyle = '#FFEB3B'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(o.x, o.y, 10, 0, Math.PI*2); ctx.stroke();
  ctx.fillStyle = '#FFEB3B'; ctx.font = '11px Consolas,monospace'; ctx.textAlign = 'left';
  ctx.fillText('O(0,0)', o.x+14, o.y-10);

  // 网格
  ctx.strokeStyle = 'rgba(48,54,61,0.4)'; ctx.lineWidth = 1;
  for (var x = 20; x <= 200; x += 20) {
    var g1 = armToCanvas(x, -75), g2 = armToCanvas(x, 222);
    ctx.beginPath(); ctx.moveTo(g1.x, g1.y); ctx.lineTo(g2.x, g2.y); ctx.stroke();
  }
  for (var y = -60; y <= 220; y += 20) {
    var g1 = armToCanvas(0, y), g2 = armToCanvas(210, y);
    ctx.beginPath(); ctx.moveTo(g1.x, g1.y); ctx.lineTo(g2.x, g2.y); ctx.stroke();
  }

  // ── 目标多边形 + Place 点 ──
  for (var i = 0; i < state.plan.length; i++) {
    var item = state.plan[i];
    var color = TARGET_COLORS[i % TARGET_COLORS.length];
    var poly = item.target_polygon_mm || item.measured_target_polygon_mm;
    if (poly && poly.length >= 2) {
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.setLineDash([]);
      ctx.beginPath();
      var first = armToCanvas(poly[0][0], poly[0][1]);
      ctx.moveTo(first.x, first.y);
      for (var j = 1; j < poly.length; j++) {
        var pp = armToCanvas(poly[j][0], poly[j][1]);
        ctx.lineTo(pp.x, pp.y);
      }
      ctx.closePath(); ctx.stroke();
      ctx.fillStyle = color + '26';
      if (color.charAt(0) === '#') ctx.fillStyle = color + '26';
      ctx.fill();
    }

    // Place 点
    var place = item.place_mm;
    if (place) {
      var pt = armToCanvas(place[0], place[1]);
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 8, 0, Math.PI*2); ctx.stroke();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 5, 0, Math.PI*2); ctx.fill();

      var pid = item.piece_id || ('#'+(i+1));
      var rot = item.rotate_deg || 0;
      ctx.fillStyle = color; ctx.font = 'bold 11px Consolas,monospace'; ctx.textAlign = 'left';
      ctx.fillText(pid + ' ' + rot.toFixed(0) + '°', pt.x+12, pt.y-8);
      ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.font = '9px Consolas,monospace';
      ctx.fillText('('+place[0].toFixed(1)+','+place[1].toFixed(1)+')', pt.x+12, pt.y+8);
    }
  }

  // ── Pick 点 + 连接线 ──
  var pickMap = {};
  for (var k = 0; k < state.pieces.length; k++) {
    pickMap[state.pieces[k].id] = state.pieces[k].pick_mm;
  }
  for (var i = 0; i < state.plan.length; i++) {
    var item = state.plan[i];
    var color = TARGET_COLORS[i % TARGET_COLORS.length];
    var place = item.place_mm;
    var pick = pickMap[item.piece_id];
    if (!place || !pick) continue;
    var from = armToCanvas(pick[0], pick[1]);
    var to = armToCanvas(place[0], place[1]);

    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([4, 6]);
    ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(from.x, from.y, 7, 0, Math.PI*2); ctx.stroke();
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.beginPath(); ctx.arc(from.x, from.y, 4, 0, Math.PI*2); ctx.fill();
  }

  // 单独 pieces (无 plan)
  if (state.plan.length === 0 && state.pieces.length > 0) {
    for (var i = 0; i < state.pieces.length; i++) {
      var p = state.pieces[i];
      var color = PIECE_COLORS[i % PIECE_COLORS.length];
      var pick = p.pick_mm;
      if (!pick) continue;
      var pt = armToCanvas(pick[0], pick[1]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 7, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(pt.x, pt.y, 9, 0, Math.PI*2); ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 11px Consolas,monospace'; ctx.textAlign = 'left';
      ctx.fillText(p.id || ('P'+(i+1)), pt.x+14, pt.y-6);
      ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '9px Consolas,monospace';
      ctx.fillText('('+pick[0].toFixed(1)+','+pick[1].toFixed(1)+')', pt.x+14, pt.y+10);
    }
  }

  // 目标矩形框
  var info = state.solve_info || {};
  if (info.target_origin_mm && info.target_size_mm) {
    var tox = info.target_origin_mm[0], toy = info.target_origin_mm[1];
    var tw = info.target_size_mm[0], th = info.target_size_mm[1];
    var r1 = armToCanvas(tox, toy), r2 = armToCanvas(tox-tw, toy);
    var r3 = armToCanvas(tox-tw, toy+th), r4 = armToCanvas(tox, toy+th);
    ctx.strokeStyle = '#2196F3'; ctx.lineWidth = 2.5; ctx.setLineDash([8, 3]);
    ctx.beginPath();
    ctx.moveTo(r1.x, r1.y); ctx.lineTo(r2.x, r2.y);
    ctx.lineTo(r3.x, r3.y); ctx.lineTo(r4.x, r4.y);
    ctx.closePath(); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#2196F3'; ctx.font = 'bold 10px Consolas,monospace'; ctx.textAlign = 'center';
    ctx.fillText(tw.toFixed(0)+'×'+th.toFixed(0)+'mm', (r1.x+r3.x)/2, (r1.y+r3.y)/2);
  }
}

// ═══ 右侧面板 ═══
function updatePanel() {
  var ph = document.getElementById('piece-list');
  if (!state.pieces.length) {
    ph.innerHTML = '<span class="empty">无拼图块数据</span>';
  } else {
    ph.innerHTML = '';
    for (var i = 0; i < state.pieces.length; i++) {
      var p = state.pieces[i];
      var color = PIECE_COLORS[i % PIECE_COLORS.length];
      var pick = p.pick_mm || [0,0];
      var area = p.area_mm2 ? ' ' + p.area_mm2.toFixed(0) + 'mm²' : '';
      var div = document.createElement('div');
      div.className = 'piece-row';
      div.innerHTML = '<span class="piece-dot" style="background:'+color+'"></span>' +
        '<span class="piece-info"><b>'+escHtml(p.id)+'</b>'+area+'<br>' +
        '<span class="coord">pick: ('+pick[0].toFixed(1)+','+pick[1].toFixed(1)+')mm</span></span>';
      ph.appendChild(div);
    }
  }

  var pl = document.getElementById('plan-list');
  if (!state.plan.length) {
    pl.innerHTML = '<span class="empty">无组装方案</span>';
  } else {
    pl.innerHTML = '';
    for (var i = 0; i < state.plan.length; i++) {
      var item = state.plan[i];
      var color = TARGET_COLORS[i % TARGET_COLORS.length];
      var place = item.place_mm || [0,0];
      var rot = item.rotate_deg || 0;
      var div = document.createElement('div');
      div.className = 'piece-row';
      div.innerHTML = '<span class="piece-dot" style="background:'+color+'"></span>' +
        '<span class="piece-info"><b>'+escHtml(item.piece_id)+'</b> ↻'+rot.toFixed(0)+'°<br>' +
        '<span class="coord">place: ('+place[0].toFixed(1)+','+place[1].toFixed(1)+')mm</span></span>';
      pl.appendChild(div);
    }
  }

  var rx = document.getElementById('rx-log');
  var lines = state.serial_rx || [];
  if (!lines.length) {
    rx.innerHTML = '<span class="empty">无串口数据</span>';
  } else {
    rx.innerHTML = '';
    var recent = lines.slice(-20);
    for (var i = 0; i < recent.length; i++) {
      var div = document.createElement('div');
      div.className = 'rx-line';
      div.innerHTML = '<span class="rx-time">'+recent[i].time+'</span><span class="rx-text">'+escHtml(recent[i].text)+'</span>';
      rx.appendChild(div);
    }
    rx.scrollTop = rx.scrollHeight;
  }

  document.getElementById('src').textContent = state.source || '--';
  var age = (Date.now()/1000 - (state.updated||0));
  var dot = document.getElementById('dot');
  if (age < 5) dot.className = 'dot live';
  else if (age < 30) dot.className = 'dot stale';
  else dot.className = 'dot dead';
  document.getElementById('age').textContent = age < 120 ? age.toFixed(0)+'秒前' : '超时';
}

function escHtml(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ═══ 十字光标 ═══
function setupCrosshair() {
  var tip = document.getElementById('tip');
  canvas.addEventListener('mousemove', function(e) {
    var r = canvas.getBoundingClientRect();
    var mx = e.clientX - r.left, my = e.clientY - r.top;
    var ax = (CANVAS_W - OX - mx) / SCALE;
    var ay = (my - OY) / SCALE;
    if (ax >= -5 && ax <= 215 && ay >= -80 && ay <= 230) {
      tip.style.display = 'block';
      tip.style.left = (mx+16)+'px'; tip.style.top = (my+16)+'px';
      tip.textContent = '('+ax.toFixed(1)+','+ay.toFixed(1)+') mm';
      tip.style.color = (ax>210||ay<-75||ay>222) ? '#888' : '#0f0';
    } else { tip.style.display = 'none'; }
  });
  canvas.addEventListener('mouseleave', function() { tip.style.display = 'none'; });
}

// ═══ 按钮 ═══
async function sendJSON() {
  var text = document.getElementById('json-input').value.trim();
  if (!text) return;
  try {
    var data = JSON.parse(text);
    await fetch('/push', {method:'POST', body:JSON.stringify(data)});
    document.getElementById('json-input').value = '';
    fetchState();
  } catch(e) { alert('JSON 解析错误: '+e.message); }
}

async function loadDemo() {
  var demo = {
    pieces: [
      {id:"piece_1",pick_mm:[38.25,80.75],area_mm2:581.4},
      {id:"piece_2",pick_mm:[23.00,152.50],area_mm2:864.8},
      {id:"piece_3",pick_mm:[-21.25,59.75],area_mm2:1466.8},
    ],
    plan: [
      {piece_id:"piece_3",pick_mm:[-21.25,59.75],place_mm:[175.93,103.71],rotate_deg:35.98,target_polygon_mm:[[160,95],[170,95],[175,110],[165,115]]},
      {piece_id:"piece_1",pick_mm:[38.25,80.75],place_mm:[115.64,123.17],rotate_deg:62.58,target_polygon_mm:[[100,115],[115,110],[120,130],[105,135]]},
      {piece_id:"piece_2",pick_mm:[23.00,152.50],place_mm:[144.19,125.60],rotate_deg:-78.28,target_polygon_mm:[[130,115],[150,120],[145,135],[125,130]]},
    ],
    pickup_order:["piece_3","piece_1","piece_2"],
    assembly_order:["piece_1","piece_2","piece_3"],
    solve_info:{mode:"unknown-pattern",fill_ratio:0.9679,target_size_mm:[100,60],target_origin_mm:[200,80]}
  };
  await fetch('/push', {method:'POST', body:JSON.stringify(demo)});
  fetchState();
}

async function clearAll() {
  await fetch('/push', {method:'POST', body:'{"pieces":[],"plan":[],"pickup_order":[],"assembly_order":[],"solve_info":{}}'});
  fetchState();
}

// ═══ 启动 ═══
initCanvas();
fetchState();
setInterval(fetchState, 500);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html(HTML)
        elif self.path == "/state":
            self._serve_json(puzzle_state)
        elif self.path == "/status":
            with STATE_LOCK:
                info = {
                    "pieces": len(puzzle_state["pieces"]),
                    "plan": len(puzzle_state["plan"]),
                    "source": puzzle_state["source"],
                    "updated": puzzle_state["updated"],
                }
            self._serve_json(info)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/push":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                _update_state_from_dict(data, f"HTTP POST {time.strftime('%H:%M:%S')}")
                self._serve_json({"ok": True})
            except json.JSONDecodeError as e:
                self.send_error(400, f"Invalid JSON: {e}")
        else:
            self.send_error(404)

    def _serve_html(self, content: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with STATE_LOCK:
            payload = json.dumps(data, ensure_ascii=False)
        self.wfile.write(payload.encode("utf-8"))

    def log_message(self, format, *args):
        pass


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = 8080
    args = sys.argv[1:]
    if "--port" in args:
        idx = args.index("--port")
        if idx + 1 < len(args):
            port = int(args[idx + 1])

    print("=" * 60)
    print("  简化版拼图 Web 界面 (无视频流)")
    print(f"  端口: {port}")
    print(f"  串口: {SERIAL_PORT} @ {SERIAL_BAUD}")
    print(f"  坐标: arm-mm (原点 A4右下+75mm, X←, Y↓)")
    print("=" * 60)

    # 加载 freeze.json
    for path in ["/home/man/puzzle_app/freeze.json", "freeze.json"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "pieces" in data:
                _update_state_from_dict(data, f"freeze.json ({path})")
                print(f"[INIT] 已加载 {path}")
                break
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    if HAS_SERIAL:
        threading.Thread(target=serial_listener, daemon=True, name="serial-rx").start()
    else:
        print("[WARN] pyserial 未安装, 仅支持 HTTP POST 数据输入")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"\n已启动: http://192.168.1.101:{port}")
    print(f"发送数据: curl -X POST http://localhost:{port}/push -d @freeze.json")
    print(f"按 Ctrl+C 退出\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
        server.shutdown()
