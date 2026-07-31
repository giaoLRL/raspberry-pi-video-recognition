import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)
stdin, stdout, stderr = ssh.exec_command('grep -n "fill=.*:.3f\|_fmt" /home/man/pi_stream_puzzle.py')
print(stdout.read().decode().strip())
ssh.close()
