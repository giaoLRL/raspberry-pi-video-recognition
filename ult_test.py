import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

# 先在远程注入终极测试代码
cmd = """python3 -c "
c = open('/home/man/pi_stream_puzzle.py','r').read()

# 去掉旧的白色线代码块（整个 obs_by_id 到 cv2.polylines 的部分）
old_block = '''# 白色薄线：源轮廓经刚性变换到目标位置（应该完全重合绿线）
             obs = obs_by_id.get(item.get(\"piece_id\", \"\"))
             if obs is not None:
                 src_mm = np.asarray(obs.polygon_mm, dtype=np.float64)
                 pick = np.asarray(item.get(\"pick_mm\", [0,0]), dtype=np.float64)
                 place = np.asarray(item.get(\"place_mm\", [0,0]), dtype=np.float64)
                 angle = math.radians(float(item.get(\"rotate_deg\", 0.0)))
                 R = rotation_matrix_row(-angle)
                 transformed_mm = transform_points(src_mm - pick, R, np.zeros(2)) + place
                 # 白线误差诊断
                 max_e = np.max(np.abs(transformed_mm - poly_mm))
                 if max_e > 0.5:
                     import sys
                     print(f'[WHITE-MISMATCH] {item.get(chr(34)+\\'piece_id\\'+chr(34))}: white_vs_green max_diff={max_e:.4f}mm', file=sys.stderr, flush=True)
                 trans_warp = transformed_mm * PIXELS_PER_MM
                 trans_cam = np.round(warp_to_camera(trans_warp, w2c)).astype(np.int32)
                 cv2.polylines(out, [trans_cam], True, (255, 255, 0), 2, cv2.LINE_AA)'''

# 替换为：直接用绿线数据画一条红色2px线（完全同源的终极对比）
new_block = '''# 终极测试：用同一个poly_mm画红色细线（如果红绿不重合，就是OpenCV渲染bug）
             poly_warp2 = poly_mm * PIXELS_PER_MM
             pcam2 = np.round(warp_to_camera(poly_warp2, w2c)).astype(np.int32)
             cv2.polylines(out, [pcam2], True, (0, 255, 255), 2, cv2.LINE_AA)'''

if old_block in c:
    c = c.replace(old_block, new_block)
    open('/home/man/pi_stream_puzzle.py','w').write(c)
    print('REPLACED - ultimate test active')
else:
    print('BLOCK NOT FOUND - trying partial match')
    # fallback
    if 'cv2.polylines(out, [trans_cam], True, (255, 255, 0), 2, cv2.LINE_AA)' in c:
        c = c.replace(
            'cv2.polylines(out, [trans_cam], True, (255, 255, 0), 2, cv2.LINE_AA)',
            '# no-transform-overlay'
        )
        open('/home/man/pi_stream_puzzle.py','w').write(c)
        print('FALLBACK: commented out transform overlay')
"
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode().strip())
err = stderr.read().decode()
if err: print('ERR:', err[:500])

# 重启
ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(12)

stdin, stdout, stderr = ssh.exec_command('grep -c "Solve OK" /tmp/puzzle.log; tail -3 /tmp/puzzle.log')
print(stdout.read().decode().strip())

ssh.close()
print('\nDone. NOW: green=3px, 红色线=2px, BOTH from SAME poly_mm data.')
print('If red and green DO NOT overlap → OpenCV LINE_AA rendering bug (thickness difference)')
print('If red and green DO overlap → the transform overlay I wrote earlier was the bug')
