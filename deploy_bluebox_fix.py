import paramiko, hashlib, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 1. Stop old process
print("[1/5] Stopping old process...")
ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

# 2. Upload fixed file
print("[2/5] Uploading fixed pi_stream_puzzle.py...")
sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()

# 3. Verify upload (MD5)
print("[3/5] Verifying upload...")
local_md5 = hashlib.md5(open(r'e:\树莓派碎片\pi_stream_puzzle.py','rb').read()).hexdigest()
stdin, stdout, stderr = ssh.exec_command('md5sum /home/man/pi_stream_puzzle.py')
remote_md5 = stdout.read().decode().strip().split()[0]
print(f"  Local  MD5: {local_md5}")
print(f"  Remote MD5: {remote_md5}")
print(f"  Match: {local_md5 == remote_md5}")

# 4. Start new process
print("[4/5] Starting...")
ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 '
    'WAYLAND_DISPLAY=wayland-0 '
    'QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(12)

# 5. Check status
print("[5/5] Checking status...")
stdin, stdout, stderr = ssh.exec_command('ps aux | grep pi_stream_puzzle | grep -v grep')
proc = stdout.read().decode().strip()
print(f"  Process: {'OK' if proc else 'NOT RUNNING'}")

stdin, stdout, stderr = ssh.exec_command('ss -tlnp | grep 8080')
port = stdout.read().decode().strip()
print(f"  Port 8080: {'OK' if port else 'NOT LISTENING'}")

stdin, stdout, stderr = ssh.exec_command('grep -E "Solve OK|BlueBox|DETECT" /tmp/puzzle.log | tail -10')
log = stdout.read().decode().strip()
print(f"\n  Log:\n{log}")

ssh.close()
print("\nDone! Visit http://192.168.31.93:8080/")
