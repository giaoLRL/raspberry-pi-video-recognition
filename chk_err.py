import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 看最新的错误详情
stdin, stdout, stderr = ssh.exec_command('grep "Solve FAILED" /tmp/puzzle.log | tail -5')
print('Latest failures:')
print(stdout.read().decode().strip())

# 看求解器内部错误
stdin, stdout, stderr = ssh.exec_command('grep -E "Traceback|Error|error" /tmp/puzzle.log | tail -10')
print('\nTracebacks:')
print(stdout.read().decode().strip())

ssh.close()
