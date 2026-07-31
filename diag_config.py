import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 查看实际config和最近求解参数
cmds = [
    'cat /home/man/puzzle_app/config.json | python3 -c "import sys,json; c=json.load(sys.stdin); u=c.get(\"unknown\",{}); print(\"fill:\",u.get(\"minimum_accepted_fill_ratio\"), \"geom:\",u.get(\"maximum_accepted_geometry_score\"), \"w:\",u.get(\"min_width_mm\"),\"-\",u.get(\"max_width_mm\"), \"h:\",u.get(\"min_height_mm\"),\"-\",u.get(\"max_height_mm\"))"',
    'grep "Solve OK" /tmp/puzzle.log | tail -5',
    'grep "BlueBox" /tmp/puzzle.log | tail -3',
    'grep "rejected\|accepted\|solution_accepted" /tmp/puzzle.log | tail -5',
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'--- {cmd[:80]}... ---')
    print(out if out else '(empty)')
    if err: print('ERR:', err[:200])
    print()

ssh.close()
