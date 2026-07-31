import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()
print('已上传')

ssh.exec_command('pkill -f pi_stream_puzzle.py')
time.sleep(2)

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 '
    'WAYLAND_DISPLAY=wayland-0 '
    'QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
print('等待15秒后查看日志...')
time.sleep(15)

stdin, stdout, stderr = ssh.exec_command('grep DEBUG-SOLVE /tmp/puzzle.log')
print(stdout.read().decode().strip())

ssh.close()
