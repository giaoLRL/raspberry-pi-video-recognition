import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

cmds = [
    "ps aux | grep -E '(weston|sway|labwc|wayfire|mutter)' | grep -v grep",
    "ls /run/user/*/wayland-* 2>/dev/null || echo 'no wayland socket'",
    "echo XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR",
    "echo WAYLAND_DISPLAY=$WAYLAND_DISPLAY",
    "who",
]
for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(f'[{cmd}]: {out}')
    if err:
        print(f'[{cmd}] ERR: {err}')

ssh.close()
