import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Wait 30 more seconds for solver to run full budget
time.sleep(30)

cmds = [
    'grep -c "Solve OK" /tmp/puzzle.log',
    'grep -c "Solve FAILED" /tmp/puzzle.log',
    'grep "fill=" /tmp/puzzle.log | tail -3',
    'grep "BlueBox" /tmp/puzzle.log | tail -1',
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print(f'{stdout.read().decode().strip()}')

ssh.close()
