import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Check file state
stdin, stdout, stderr = ssh.exec_command('ls -la /home/man/pi_stream_puzzle.py; md5sum /home/man/pi_stream_puzzle.py')
print('Pi file:', stdout.read().decode().strip())

# Also check running process
stdin, stdout, stderr = ssh.exec_command('ps aux | grep pi_stream | grep -v grep')
print('Process:', stdout.read().decode().strip())

ssh.close()
