import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 在 draw_overlay 白线绘制后加入误差打印
cmd = """python3 -c "
content = open('/home/man/pi_stream_puzzle.py','r').read()
old = '# 白色薄线'
new = '''# 白线误差诊断
                 max_e = np.max(np.abs(transformed_mm - poly_mm))
                 if max_e > 0.5:
                     import sys
                     print(f'[WHITE-MISMATCH] {item.get(chr(34)+'piece_id'+chr(34))}: white_vs_green max_diff={max_e:.4f}mm', file=sys.stderr, flush=True)
# 白色薄线'''
content = content.replace(old, new)
open('/home/man/pi_stream_puzzle.py','w').write(content)
print('OK')
" 2>&1
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode().strip())
err_out = stderr.read().decode().strip()
if err_out: print('ERR:', err_out[:500])

# 重启
ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)
ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(15)

# 读取诊断输出
stdin, stdout, stderr = ssh.exec_command('grep WHITE-MISMATCH /tmp/puzzle.log | tail -20')
print('\n=== WHITE-MISMATCH diagnosis ===')
print(stdout.read().decode().strip() or 'No mismatch detected (white == green)')

ssh.close()
