import paramiko, json
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Read current config
sftp = ssh.open_sftp()
with sftp.open('/home/man/puzzle_app/config.json', 'r') as f:
    config = json.load(f)

# Unlock size constraints
config['unknown']['min_width_mm'] = 10.0
config['unknown']['max_width_mm'] = 210.0
config['unknown']['min_height_mm'] = 10.0
config['unknown']['max_height_mm'] = 210.0
config['unknown']['boundary_snap_tolerance_mm'] = 0.0

# Write back
with sftp.open('/home/man/puzzle_app/config.json', 'w') as f:
    json.dump(config, f, indent=2)

sftp.close()
print('Config updated')

# Verify
stdin, stdout, stderr = ssh.exec_command('grep -A1 "min_width\|max_width\|min_height\|max_height\|boundary_snap" /home/man/puzzle_app/config.json | grep -v "^--$"')
print(stdout.read().decode().strip())

# Restart
ssh.exec_command('pkill -f pi_stream_puzzle.py')
import time
time.sleep(2)
ssh.exec_command('cd /home/man && XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &')
time.sleep(15)

stdin, stdout, stderr = ssh.exec_command('grep "Solve OK" /tmp/puzzle.log | tail -2')
print('Solves:', stdout.read().decode().strip())

ssh.close()
