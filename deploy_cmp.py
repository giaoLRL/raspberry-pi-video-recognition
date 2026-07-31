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
print(f'MD5 match: {lm == rm}')

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(12)

stdin, stdout, stderr = ssh.exec_command('ps aux | grep pi_stream | grep -v grep | wc -l')
print(f'Process: {stdout.read().decode().strip()}')

stdin, stdout, stderr = ssh.exec_command('grep -c "Solve OK" /tmp/puzzle.log')
print(f'Solve OK: {stdout.read().decode().strip()}')

stdin, stdout, stderr = ssh.exec_command('tail -3 /tmp/puzzle.log')
print(f'Last log: {stdout.read().decode().strip()}')

ssh.close()
print('Done - http://192.168.31.93:8080/')
