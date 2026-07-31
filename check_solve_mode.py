import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 查看完整求解日志
stdin, stdout, stderr = ssh.exec_command('grep -E "Solve (OK|FAILED)|DEBUG-SOLVE" /tmp/puzzle.log | tail -20')
print(stdout.read().decode().strip())

ssh.close()
