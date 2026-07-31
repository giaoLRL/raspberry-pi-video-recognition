import paramiko, time

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.31.93", username="man", password="giao666666", timeout=15)
sftp = client.open_sftp()

# Stop
client.exec_command("sudo pkill -f pi_stream_puzzle.py 2>/dev/null")
time.sleep(1)

# Remove old calibration
client.exec_command("rm -f /home/man/puzzle_app/a4_corners.json")

# Upload latest
sftp.put(r"e:\树莓派碎片\pi_stream_puzzle.py", "/home/man/pi_stream_puzzle.py")
print("Uploaded pi_stream_puzzle.py")

# Restart
client.exec_command("cd /home/man/puzzle_app && nohup python3 /home/man/pi_stream_puzzle.py > /tmp/puzzle_stream.log 2>&1 &")
time.sleep(4)

# Verify
stdin, stdout, stderr = client.exec_command("ss -tlnp | grep 8080")
print("Port 8080:", stdout.read().decode().strip())

stdin, stdout, stderr = client.exec_command("curl -s http://localhost:8080/status")
print("Status:", stdout.read().decode().strip())

# Check log for A4 detection
stdin, stdout, stderr = client.exec_command("head -30 /tmp/puzzle_stream.log")
log = stdout.read().decode().strip()
for line in log.split("\n"):
    line = line.strip()
    if line and "GET" not in line:
        print("Log:", line)

sftp.close()
client.close()
print("\nDone - visit http://192.168.31.93:8080/")
