import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Check taught_layout
stdin, stdout, stderr = ssh.exec_command('cat /home/man/puzzle_app/taught_layout.json')
print('=== taught_layout.json ===')
print(stdout.read().decode().strip())

# Check config use_taught_layout
stdin, stdout, stderr = ssh.exec_command('grep -A2 "use_taught_layout" /home/man/puzzle_app/config.json')
print('\n=== use_taught_layout ===')
print(stdout.read().decode().strip())

# Check latest solve details
stdin, stdout, stderr = ssh.exec_command('grep -E "solver_path|taught" /tmp/puzzle.log | tail -5')
print('\n=== solver path ===')
print(stdout.read().decode().strip())

ssh.close()
