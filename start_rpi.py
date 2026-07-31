import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 用Wayland环境变量启动
startup_cmd = (
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 '
    'WAYLAND_DISPLAY=wayland-0 '
    'QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
stdin, stdout, stderr = ssh.exec_command(startup_cmd)
print('启动命令已发送')
err = stderr.read().decode().strip()
if err:
    print('stderr:', err)
out = stdout.read().decode().strip()
if out:
    print('stdout:', out)

# 等待一下
import time
time.sleep(3)

# 检查状态
for cmd in [
    'ps aux | grep pi_stream_puzzle | grep -v grep',
    'ss -tlnp | grep 8080',
]:
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode().strip()
    print(f'[{cmd}]: {out if out else "无结果"}')

# 检查日志
stdin, stdout, stderr = ssh.exec_command('tail -10 /tmp/puzzle.log')
print('日志:')
print(stdout.read().decode().strip())

ssh.close()
