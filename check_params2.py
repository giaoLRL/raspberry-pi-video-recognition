import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 查看PIXELS_PER_MM和A4标定相关
stdin, stdout, stderr = ssh.exec_command("grep -n -A5 'PIXELS_PER_MM\\|warp_px_to_cm\\|SCALE_PX_PER_CM\\|scale_x\\|scale_y\\|A4_W\\|A4_H' /home/man/pi_stream_puzzle.py | head -40")
print(stdout.read().decode())

print("\n=== 标定相关 ===")
stdin, stdout, stderr = ssh.exec_command("grep -n 'calibration\\|corners\\|scale' /home/man/pi_stream_puzzle.py | head -20")
print(stdout.read().decode())

ssh.close()
