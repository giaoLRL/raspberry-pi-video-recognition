import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Check key config values + what the solver actually uses
check_script = """
import json, sys
sys.path.insert(0, '/home/man/puzzle_app')
from puzzle_vision.config import load_config, DEFAULT_CONFIG

# Check if config.json used
import os
cfg_path = '/home/man/puzzle_app/config.json'
print('config.json exists:', os.path.exists(cfg_path))
print('config.json size:', os.path.getsize(cfg_path) if os.path.exists(cfg_path) else 0)

# Load
config = load_config(cfg_path if os.path.exists(cfg_path) else None)
u = config.get('unknown', {})
print()
print('min_width_mm:', u.get('min_width_mm'))
print('max_width_mm:', u.get('max_width_mm'))
print('min_height_mm:', u.get('min_height_mm'))
print('max_height_mm:', u.get('max_height_mm'))
print('minimum_accepted_fill_ratio:', u.get('minimum_accepted_fill_ratio'))
print('maximum_accepted_geometry_score:', u.get('maximum_accepted_geometry_score'))
print('max_search_seconds:', u.get('max_search_seconds'))
print('max_search_nodes:', u.get('max_search_nodes'))
"""
stdin, stdout, stderr = ssh.exec_command(f'python3 -c "{check_script}"')
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(out)
if err: print('ERR:', err[:500])

ssh.close()
