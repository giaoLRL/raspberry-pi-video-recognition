import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 查看关键参数和过滤逻辑
stdin, stdout, stderr = ssh.exec_command("grep -n -A2 'PIXELS_PER_CM\\|MIN_PIECE_AREA\\|MAX_PIECE_AREA\\|MIN_PIECE_SHORT\\|area.*cm2\\|area.*8800\\|area.*169000\\|filter.*area\\|短边' /home/man/pi_stream_puzzle.py | head -60")
print(stdout.read().decode())

ssh.close()
