import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

cmds = [
    "cat /etc/wayvnc/config",
    "ss -tlnp | grep wayvnc",
    "ss -tlnp | grep -E '590|vnc'",
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'--- {cmd} ---')
    if out:
        print(out)
    if err:
        print('ERR:', err)
    print()

ssh.close()
