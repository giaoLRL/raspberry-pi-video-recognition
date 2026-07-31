"""Debug the solver with current detection data."""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.31.93", username="man", password="giao666666", timeout=10)

# Stop stream to release camera
stdin, stdout, stderr = client.exec_command("sudo pkill -f pi_stream_puzzle.py 2>/dev/null; sleep 1; echo stopped")
print("Stream stopped:", stdout.read().decode().strip())

script = """
import sys, json, numpy as np
sys.path.insert(0, "/home/man/puzzle_app")
from puzzle_vision.config import load_config
from puzzle_vision.geometry import edge_lengths, normalize_winding, polygon_area
from pathlib import Path

# Read calibration and config
calib = json.loads(Path("/home/man/puzzle_app/a4_corners.json").read_text())
cfg = load_config(str(Path("/home/man/puzzle_app/config.json")))

PPM = (840/210.0 + 1188/297.0) / 2.0
print(f"Pixels per mm: {PPM}")
print(f"Corners: {calib['corners']}")

# Simulate what detect_pieces produces for the current frame
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
ret, frame = cap.read()
cap.release()

if not ret:
    print("No frame")
    exit()

# Build matrices
src_pts = np.array(calib["corners"], dtype=np.float32)
dst_pts = np.array([[0,0],[839,0],[839,1187],[0,1187]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(frame, M, (840, 1188))

# Simple segmentation - find white pieces on orange
hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
h, w = warped.shape[:2]

# Sample corners
corners_px = 30
samples = []
for (y,x) in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]:
    patch = hsv[max(0,y-corners_px):min(h,y+corners_px), max(0,x-corners_px):min(w,x+corners_px)]
    if patch.size > 0:
        samples.append(patch.reshape(-1,3))
all_s = np.vstack(samples)
bg_h = float(np.median(all_s[:,0]))
bg_s = float(np.median(all_s[:,1]))
bg_v = float(np.median(all_s[:,2]))
print(f"BG HSV: H={bg_h:.0f} S={bg_s:.0f} V={bg_v:.0f}")

dh = np.abs(hsv[:,:,0].astype(np.float32)-bg_h)
dh = np.minimum(dh, 180-dh)
ds = np.abs(hsv[:,:,1].astype(np.float32)-bg_s)
dv = np.abs(hsv[:,:,2].astype(np.float32)-bg_v)
dist = np.sqrt(dh*dh*0.5 + ds*ds*0.3 + dv*dv*0.2)
mask = (dist > 60).astype(np.uint8)*255

cv2.imwrite("/tmp/debug_mask.jpg", mask)
print(f"Mask saved. Non-zero pixels: {np.count_nonzero(mask)}")

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contours: {len(contours)}")

pieces_data = []
for ic, c in enumerate(contours):
    area = cv2.contourArea(c)
    area_mm2 = area / (PPM**2)
    hull = cv2.convexHull(c)
    poly = cv2.approxPolyDP(hull, 0.015 * cv2.arcLength(hull, True), True)
    if len(poly) < 3:
        continue
    edges_mm = [float(np.linalg.norm(poly[i][0]-poly[(i+1)%len(poly)][0])/PPM) for i in range(len(poly))]
    print(f"Piece {ic}: area={area_mm2:.0f}mm2 sides={len(poly)} edges_mm={[round(e,1) for e in edges_mm]}")
    pieces_data.append({"area_mm2": area_mm2, "sides": len(poly), "edges_mm": edges_mm})

# Save debug
import json
with open("/tmp/pieces_debug.json", "w") as f:
    json.dump({"bg_hsv": [bg_h, bg_s, bg_v], "ppm": PPM, "pieces": pieces_data}, f, indent=2)
print("Saved pieces_debug.json")
"""

sftp = client.open_sftp()
sftp.putfo(__import__('io').BytesIO(script.encode()), "/tmp/debug_solve_now.py")
sftp.close()

stdin, stdout, stderr = client.exec_command("python3 /tmp/debug_solve_now.py 2>&1")
print(stdout.read().decode().strip())

# Download debug files
sftp = client.open_sftp()
for f in ["debug_mask.jpg", "pieces_debug.json"]:
    try:
        sftp.get("/tmp/"+f, r"e:\树莓派碎片\\"+f)
        import os
        print("Downloaded", f, os.path.getsize(r"e:\树莓派碎片\\"+f), "bytes")
    except Exception as e:
        print("Skip", f, e)
sftp.close()
client.close()

# Restart stream
client2 = paramiko.SSHClient()
client2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client2.connect("192.168.31.93", username="man", password="giao666666", timeout=10)
client2.exec_command("cd /home/man/puzzle_app && nohup python3 /home/man/pi_stream_puzzle.py > /tmp/puzzle_stream.log 2>&1 &")
client2.close()
print("Stream restarted")
