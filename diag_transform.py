import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 临时注入诊断代码到 draw_overlay
cmd = r"""python3 -c "
import numpy as np, math

# Last solve result
import json, sys
sys.path.insert(0, '/home/man/puzzle_app')
from puzzle_vision.geometry import rotation_matrix_row, transform_points

# Read from last solve log the IN/OUT coords
# Actually let's just check if our transform matches by doing a quick test
# piece_1 data from earlier log:
src = np.asarray([(174.8,138.8), (160.5,215.8), (129.5,145.8)], dtype=np.float64)
dst = np.asarray([(82.5,40.5),   (160.8,40.5),   (97.6,83.7)], dtype=np.float64)
pick = np.asarray([154.93, 166.8], dtype=np.float64)   # centroid
place = np.asarray([113.63, 54.9], dtype=np.float64)   # centroid

# Approach 1: the solver's direct transform
src_c = src - src.mean(axis=0); dst_c = dst - dst.mean(axis=0)
U,s,Vt = np.linalg.svd(src_c.T @ dst_c)
R_true = U @ Vt
t_true = dst.mean(axis=0) - src.mean(axis=0) @ R_true
result1 = src @ R_true + t_true
print('Solver direct:')
print(f'  R={np.round(R_true,4)}')
print(f'  t={np.round(t_true,2)}')
print(f'  error={np.max(np.abs(result1-dst)):.6f}')

# Approach 2: pick/place/angle reconstruction
angle = math.degrees(math.atan2(R_true[0,1], R_true[0,0]))
rot_solver = rotation_matrix_row(math.radians(angle))
t_solver = place - pick @ rot_solver
result2 = src @ rot_solver + t_solver
print('')
print(f'Approach 2 (pick->place, angle):')
print(f'  angle={angle:.1f} deg')
print(f'  rot={np.round(rot_solver,4)}')
print(f'  t={np.round(t_solver,2)}')
print(f'  error={np.max(np.abs(result2-dst)):.6f}')

# Approach 3: my current code in draw_overlay
# rotate_deg = wrap_angle_deg(-angle)
rotate_deg = (angle + 180.0) % 360.0 - 180.0  # NEGATE then wrap
print(f'')
print(f'  rotate_deg (as reported by solver) = {rotate_deg:.1f}')
R = rotation_matrix_row(-math.radians(rotate_deg))
result3 = (src - pick) @ R + place
print(f'Approach 3 (my code):')
print(f'  R={np.round(R,4)}')
print(f'  error={np.max(np.abs(result3-dst)):.6f}')

# Check if R matches R_true
print(f'')
print(f'R_true vs R: diff={np.max(np.abs(R_true-R)):.6f}')
print(f'result1 vs result3: diff={np.max(np.abs(result1-result3)):.6f}')
" 2>&1
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print('OUT:', out[:2000] if out else '(empty)')
print('ERR:', err[:1000] if err else '(empty)')

ssh.close()
