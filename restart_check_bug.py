import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -f pi_stream_puzzle.py')
time.sleep(2)

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 '
    'WAYLAND_DISPLAY=wayland-0 '
    'QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(20)

stdin, stdout, stderr = ssh.exec_command('grep -E "SOLVER-BUG|Solve OK" /tmp/puzzle.log | head -20')
print(stdout.read().decode().strip())

ssh.close()
