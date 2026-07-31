import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 等一会让求解器多跑几轮
time.sleep(20)

# 验证config + 检查最新求解结果
cmds = [
    'python3 -c "import json; c=json.load(open(\"/home/man/puzzle_app/config.json\")); u=c[\"unknown\"]; print(\"min_w:\",u[\"min_width_mm\"],\"max_w:\",u[\"max_width_mm\"],\"h:\",u[\"min_height_mm\"],\"-\",u[\"max_height_mm\"],\"fill:\",u[\"minimum_accepted_fill_ratio\"])"',
    'grep -c "Solve OK" /tmp/puzzle.log',
    'grep -c "FAILED" /tmp/puzzle.log',
    'tail -10 /tmp/puzzle.log',
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'--- {cmd[:70]} ---')
    print(out)
    if err: print('ERR:', err[:200])
    print()

ssh.close()
