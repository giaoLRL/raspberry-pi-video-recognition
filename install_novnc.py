import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 安装 noVNC
stdin, stdout, stderr = ssh.exec_command('sudo apt install -y novnc 2>&1')
print('安装中...')
out = stdout.read().decode()
err = stderr.read().decode()
print(out[-500:] if len(out) > 500 else out)
if err:
    print('stderr:', err[-300:])

print('\n=== 检查安装 ===')
stdin, stdout, stderr = ssh.exec_command('which websockify; ls /usr/share/novnc/ 2>/dev/null | head -5')
print(stdout.read().decode().strip())

ssh.close()
