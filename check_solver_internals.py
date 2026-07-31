import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Add debug in UnknownPuzzleSolver.solve() at the plan building stage
stdin, stdout, stderr = ssh.exec_command("grep -n 'target_polygon = transform_points' /home/man/puzzle_app/puzzle_vision/solver.py")
print('target lines:', stdout.read().decode().strip())

# Check what's around line 2420
stdin, stdout, stderr = ssh.exec_command("sed -n '2415,2445p' /home/man/puzzle_app/puzzle_vision/solver.py")
print('\nContext:')
print(stdout.read().decode())

ssh.close()
