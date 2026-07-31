import paramiko, hashlib, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

# 上传正确的 config.json（从原项目）
sftp = ssh.open_sftp()
sftp.put(r'D:\nnn\pick_rdk_solver_complete\config.json', '/home/man/puzzle_app/config.json')
sftp.close()

# 验证
stdin, stdout, stderr = ssh.exec_command('python3 -c "import json; c=json.load(open(\"/home/man/puzzle_app/config.json\")); u=c[\"unknown\"]; print(\"min_w:\",u[\"min_width_mm\"],\"max_w:\",u[\"max_width_mm\"],\"min_h:\",u[\"min_height_mm\"],\"max_h:\",u[\"max_height_mm\"],\"fill:\",u[\"minimum_accepted_fill_ratio\"],\"geom:\",u[\"maximum_accepted_geometry_score\"])"')
print('Config restored:', stdout.read().decode().strip())

# 重启
ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(15)

# 检查
for check in ['grep -c "Solve OK" /tmp/puzzle.log', 'grep "BlueBox" /tmp/puzzle.log | tail -2']:
    stdin, stdout, stderr = ssh.exec_command(check)
    print(stdout.read().decode().strip())

ssh.close()
print('\nDone! Config restored to original values.')
print('Now the solver will only accept rectangles:')
print('   width 90-120mm, height 50-90mm, fill>=90%, geom<=16')
print('http://192.168.31.93:8080/')
