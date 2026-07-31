import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 先停掉旧的（如果有）
ssh.exec_command('pkill -f websockify 2>/dev/null')
time.sleep(1)

# 启动 websockify：6080端口浏览器访问，转发到本地5900的VNC
cmd = 'nohup websockify --web=/usr/share/novnc 6080 localhost:5900 > /tmp/novnc.log 2>&1 &'
stdin, stdout, stderr = ssh.exec_command(cmd)
print('启动命令已发送')
time.sleep(2)

# 检查状态
stdin, stdout, stderr = ssh.exec_command('ps aux | grep websockify | grep -v grep')
print('进程:', stdout.read().decode().strip())

stdin, stdout, stderr = ssh.exec_command('ss -tlnp | grep 6080')
print('6080端口:', stdout.read().decode().strip())

ssh.close()
