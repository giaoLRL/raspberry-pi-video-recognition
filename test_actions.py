import paramiko
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("192.168.31.93", username="man", password="giao666666", timeout=10)

# Test actions
for cmd in ["toggle_overlay", "cycle_area", "print_debug", "save", "auto"]:
    stdin, stdout, stderr = client.exec_command("curl -s http://localhost:8080/action?cmd=" + cmd)
    print(cmd + ": " + stdout.read().decode().strip())

# Check status
stdin, stdout, stderr = client.exec_command("curl -s http://localhost:8080/status")
print("\nStatus: " + stdout.read().decode().strip())

client.close()
