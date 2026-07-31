import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

ssh.exec_command('pkill -9 -f pi_stream_puzzle.py 2>/dev/null')
time.sleep(2)

sftp = ssh.open_sftp()
sftp.put(r'e:\树莓派碎片\pi_stream_puzzle.py', '/home/man/pi_stream_puzzle.py')
sftp.close()

# 远程也在白线处做醒目的修改
cmd = """python3 -c "
c = open('/home/man/pi_stream_puzzle.py','r').read()
# 1px白改2px青，更醒目
c = c.replace('(255, 255, 255), 1, cv2.LINE_AA', '(255, 255, 0), 2, cv2.LINE_AA')
open('/home/man/pi_stream_puzzle.py','w').write(c)
print('done')
"
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode().strip())

ssh.exec_command(
    'cd /home/man && '
    'XDG_RUNTIME_DIR=/run/user/6 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland '
    'nohup python3 pi_stream_puzzle.py > /tmp/puzzle.log 2>&1 &'
)
time.sleep(12)

stdin, stdout, stderr = ssh.exec_command('grep -c "Solve OK" /tmp/puzzle.log; grep WHITE-MISMATCH /tmp/puzzle.log | tail -5')
out = stdout.read().decode().strip()
err = stderr.read().decode().strip() 
print(out if out else 'No mismatch')
if err: print('stderr:', err[:200])

ssh.close()
print('\nDone. White=cyan 2px now. Visit http://192.168.31.93:8080/')
print('If cyan line overlaps green → solver correct, perspective is the cause.')
print('If cyan line is offset from green → there IS a solver bug.')
