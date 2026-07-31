import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='gao666666', timeout=10)
cmd = "grep -E 'OUT|DIAG|Solve OK|BlueBox|auto:' /tmp/puzzle.log | tail -25"
_, o, _ = ssh.exec_command(cmd)
print(o.read().decode())
ssh.close()
