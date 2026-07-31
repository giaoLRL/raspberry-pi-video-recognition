import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 查看WARP尺寸
stdin, stdout, stderr = ssh.exec_command("grep -n 'WARP_WIDTH\\|WARP_HEIGHT' /home/man/pi_stream_puzzle.py | head -10")
print('WARP尺寸:')
print(stdout.read().decode())

# 查看完整的检测过滤逻辑（250-270行）
stdin, stdout, stderr = ssh.exec_command("sed -n '240,300p' /home/man/pi_stream_puzzle.py")
print('\n=== 检测过滤逻辑(240-300) ===')
print(stdout.read().decode())

ssh.close()
