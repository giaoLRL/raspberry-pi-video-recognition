import paramiko, hashlib, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Force kill
ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(3)

# Upload
sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()
print('Uploaded')

# Verify
local_md5 = hashlib.md5(open(r'e:\树莓派碎片\pi_stream_puzzle.py','rb').read()).hexdigest()
stdin, stdout, stderr = ssh.exec_command('md5sum /home/man/pi_stream_puzzle.py')
remote_md5 = stdout.read().decode().strip().split()[0]
print(f'Local MD5:  {local_md5}')
print(f'Remote MD5: {remote_md5}')
print(f'Match: {local_md5 == remote_md5}')

# Start
ssh.exec_command('cd /home/man && XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &')
time.sleep(15)

# Check results
for cmd in ['grep "Solve OK" /tmp/puzzle.log | tail -2', 'grep "Solve FAILED" /tmp/puzzle.log | tail -2']:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    if out:
        print(cmd.split('|')[0].strip(), ':', out)

ssh.close()
