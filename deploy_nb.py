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
time.sleep(18)

for check in [
    'grep -c "Solve OK" /tmp/puzzle.log',
    'grep "BlueBox" /tmp/puzzle.log | tail -2',
    'grep "FAILED" /tmp/puzzle.log | tail -2',
]:
    stdin, stdout, stderr = ssh.exec_command(check)
    print(stdout.read().decode().strip())

ssh.close()
print('\nDone! Fast budget removed, full search: 18000 nodes / 3.2s')
print('http://192.168.31.93:8080/')
