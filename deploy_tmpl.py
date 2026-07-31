import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(15)

for check in [
    'grep -c "Solve OK" /tmp/puzzle.log',
    'grep "BlueBox" /tmp/puzzle.log | tail -1',
]:
    stdin, stdout, stderr = ssh.exec_command(check)
    print(stdout.read().decode().strip())

ssh.close()
print()
print('现在画面有：')
print('  绿色=观测碎片实际位置  红色=同数据验证  紫色虚线+[T]=模板理想形状')
print('  蓝色框=100x60mm 目标矩形')
print('如果紫色模板线完美贴合蓝框 → 求解100%正确')
print('http://192.168.31.93:8080/')
