import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 在 draw_overlay 里插入调试代码，打印实际的 item 数据
cmd = """python3 -c "
import numpy as np, math, sys
sys.path.insert(0,'/home/man/puzzle_app')
from puzzle_vision.geometry import rotation_matrix_row, transform_points

# 模拟一段 plan item 数据（从实际日志取）
src_mm = np.asarray([(174.8,138.8), (160.5,215.8), (129.5,145.8)], dtype=np.float64)
pick = np.asarray([154.93, 166.80], dtype=np.float64)  # 源中心
place = np.asarray([113.63, 54.90], dtype=np.float64)   # 目标中心
target_mm = np.asarray([(82.5,40.5), (160.8,40.5), (97.6,83.7)], dtype=np.float64)

# === 重建求解器的正确变换 ===
src_c = src_mm - src_mm.mean(axis=0)
dst_c = target_mm - target_mm.mean(axis=0)
U,s,Vt = np.linalg.svd(src_c.T @ dst_c)
R = U @ Vt
t = target_mm.mean(axis=0) - src_mm.mean(axis=0) @ R
print(f'求解器 R={np.round(R,4)}')
print(f'求解器 t={np.round(t,2)}')

# rotate_deg 怎么算的？
theta = math.degrees(math.atan2(R[0,1], R[0,0]))
rotate_deg = ((theta + 180.0) % 360.0 - 180.0)  # wrap_angle_deg
rotate_deg_reported = (( -theta + 180.0) % 360.0 - 180.0)  # wrap_angle_deg(-theta)
print(f'')
print(f'theta={theta:.1f} deg')
print(f'rotate_deg(如果我直接用theta)={rotate_deg:.1f}')
print(f'rotate_deg(plan里实际报告的)={rotate_deg_reported:.1f}')

# 用报告中实际的 rotate_deg 重建
for rd_use in [rotate_deg, rotate_deg_reported, 100.5, -100.5]:
    R2 = rotation_matrix_row(-math.radians(rd_use))
    result = (src_mm - pick) @ R2 + place
    err = np.max(np.abs(result - target_mm))
    marker = ' <--' if err < 1.0 else ''
    print(f'  用 rotate_deg={rd_use:6.1f}: R={np.round(R2[0],3)}  error={err:.2f}mm{marker}')

print(f'')
print(f'结论: rotate_deg 的正确值是 {rotate_deg_reported:.1f}°')
print(f'       rotation_matrix_row(-rotate_deg) = rotation_matrix_row({-rotate_deg_reported:.1f}°)')
" 2>&1
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(out[:3000] if out else '(empty)')
if err: print('ERR:', err[:500])

ssh.close()
