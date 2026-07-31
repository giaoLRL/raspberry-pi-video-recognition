import paramiko, time
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 抓取一帧MJPEG图像保存到本地
import urllib.request
try:
    req = urllib.request.urlopen('http://192.168.31.93:8080/', timeout=5)
    # 读取MJPEG边界后的第一帧JPEG
    buf = b''
    while True:
        chunk = req.read(4096)
        if not chunk:
            break
        buf += chunk
        # 查找JPEG起始和结束
        start = buf.find(b'\xff\xd8')
        end = buf.find(b'\xff\xd9')
        if start != -1 and end != -1 and end > start:
            jpg = buf[start:end+2]
            with open(r'e:\树莓派碎片\frame.jpg', 'wb') as f:
                f.write(jpg)
            print(f'Saved frame.jpg, size={len(jpg)} bytes')
            break
    else:
        print('No JPEG frame found, buf len=', len(buf))
except Exception as e:
    print('capture error:', e)

# 同时取一下日志中所有 BlueBox 行和 target_size
_, o, _ = ssh.exec_command("grep -E 'BlueBox|target_size|target_origin' /tmp/puzzle.log | tail -10")
print('--- LOG ---')
print(o.read().decode())

# 取config.json中的目标矩形约束
_, o, _ = ssh.exec_command("cat /home/man/puzzle_app/config.json")
print('--- CONFIG ---')
print(o.read().decode())

ssh.close()
