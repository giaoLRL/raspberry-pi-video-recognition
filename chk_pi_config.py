import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

cmds = [
    'ls -la /home/man/puzzle_app/config.json 2>&1',
    'head -5 /home/man/puzzle_app/config.json 2>&1',
    'python3 -c "import json; c=json.load(open(\"/home/man/puzzle_app/config.json\")); u=c[\"unknown\"]; print(\"fill_ratio:\",u[\"minimum_accepted_fill_ratio\"], \"geom:\",u[\"maximum_accepted_geometry_score\"])" 2>&1',
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'CMD: {cmd}')
    print(f'OUT: {out}')
    if err: print(f'ERR: {err[:300]}')
    print()

ssh.close()
