import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 上传更新的代码
sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()
print('代码已上传')

# 停掉旧进程
ssh.exec_command('pkill -f pi_stream_puzzle.py')
time.sleep(2)

# 启动新进程
stdin, stdout, stderr = ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 '
    'WAYLAND_DISPLAY=wayland-0 '
    'QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(3)

# 检查状态
stdin, stdout, stderr = ssh.exec_command('ps aux | grep pi_stream_puzzle | grep -v grep')
print('进程:', stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command('ss -tlnp | grep 8080')
print('8080端口:', stdout.read().decode().strip())

# 等一会看日志
time.sleep(5)
stdin, stdout, stderr = ssh.exec_command('grep DETECT /tmp/puzzle.log | tail -10')
print('\n检测日志:')
print(stdout.read().decode().strip())

ssh.close()
