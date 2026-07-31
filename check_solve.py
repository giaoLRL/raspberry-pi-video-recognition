import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)
stdin, stdout, stderr = ssh.exec_command('grep -E "Solve (OK|FAILED)" /tmp/puzzle.log | tail -5')
print('OUT:', stdout.read().decode().strip())
print('ERR:', stderr.read().decode().strip())
ssh.close()
