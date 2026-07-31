import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Check for errors
stdin, stdout, stderr = ssh.exec_command('grep -i "error\\|traceback\\|fail\\|exception" /tmp/puzzle.log | head -10')
print('Errors:')
print(stdout.read().decode().strip())

# Check if process is still running
stdin, stdout, stderr = ssh.exec_command('ps aux | grep pi_stream | grep -v grep')
print('\nProcess:')
print(stdout.read().decode().strip())

# Check solver.py syntax
stdin, stdout, stderr = ssh.exec_command('python3 -c "import py_compile; py_compile.compile(\'/home/man/puzzle_app/puzzle_vision/solver.py\', doraise=True); print(\'OK\')" 2>&1')
print('\nSyntax check:')
print(stdout.read().decode().strip())
print(stderr.read().decode().strip()[:500])

ssh.close()
