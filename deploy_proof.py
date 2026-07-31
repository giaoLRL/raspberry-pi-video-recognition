import paramiko, hashlib, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()

lm = hashlib.md5(open(r'e:\树莓派碎片\pi_stream_puzzle.py','rb').read()).hexdigest()
stdin, stdout, stderr = ssh.exec_command('md5sum /home/man/pi_stream_puzzle.py')
rm = stdout.read().decode().strip().split()[0]
print(f'MD5 OK: {lm == rm}')

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(15)

stdin, stdout, stderr = ssh.exec_command('tail -5 /tmp/puzzle.log')
print(stdout.read().decode().strip())

ssh.close()
print()
print('=== 终极测试 ===')
print('绿色 3px + 红色 1px + 橙色半透明 4px')
print('三组数据来自 EXACT SAME pcam 数组!')
print('如果绿/红/橙不重合 → OpenCV cv2.LINE_AA thickness渲染bug')
print('如果绿/红/橙完全重合 → 之前白线变换有误')
print('http://192.168.31.93:8080/')
