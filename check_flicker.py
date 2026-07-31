import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)
stdin, stdout, stderr = ssh.exec_command('grep -E "Solve OK|Solve FAILED|DETECT" /tmp/puzzle.log | tail -20')
print(stdout.read().decode().strip())
ssh.close()
