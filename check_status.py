import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

stdin, stdout, stderr = ssh.exec_command('grep -i error /tmp/puzzle.log | tail -5')
print('Errors:', stdout.read().decode().strip() or 'none')

stdin, stdout, stderr = ssh.exec_command('grep -c "Solve OK" /tmp/puzzle.log')
print('Solve OK count:', stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command('grep DETECT /tmp/puzzle.log | tail -3')
print('Detect:', stdout.read().decode().strip())

ssh.close()
