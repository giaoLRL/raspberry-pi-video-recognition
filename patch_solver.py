import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# Add debug at line 2420 in solver.py
# We need to insert: print edge lengths of observation.polygon_mm before transform
# and target_polygon after transform
cmd = r"""python3 -c "
import re
with open('/home/man/puzzle_app/puzzle_vision/solver.py', 'r') as f:
    content = f.read()

old = '''            target_polygon = transform_points(
                observation.polygon_mm, target_r, target_t
            )'''

new = '''            target_polygon = transform_points(
                observation.polygon_mm, target_r, target_t
            )
            # DEBUG
            import numpy as np
            obs_edges = [np.linalg.norm(observation.polygon_mm[i] - observation.polygon_mm[(i+1)%len(observation.polygon_mm)]) for i in range(len(observation.polygon_mm))]
            tgt_edges = [np.linalg.norm(target_polygon[i] - target_polygon[(i+1)%len(target_polygon)]) for i in range(len(target_polygon))]
            obs_str = ', '.join(f'{e:.3f}' for e in obs_edges)
            tgt_str = ', '.join(f'{e:.3f}' for e in tgt_edges)
            if abs(sum(obs_edges)-sum(tgt_edges)) > 0.01:
                import sys
                print(f'[SOLVER-BUG] {observation.id}: IN edges=[{obs_str}] OUT edges=[{tgt_str}] perim_in={sum(obs_edges):.3f} perim_out={sum(tgt_edges):.3f}', file=sys.stderr, flush=True)
                print(f'[SOLVER-BUG]   IN coords=' + str(np.round(observation.polygon_mm, 3).tolist()), file=sys.stderr, flush=True)
                print(f'[SOLVER-BUG]   OUT coords=' + str(np.round(target_polygon, 3).tolist()), file=sys.stderr, flush=True)
                print(f'[SOLVER-BUG]   target_r=' + str(np.round(target_r, 6).tolist()), file=sys.stderr, flush=True)
                print(f'[SOLVER-BUG]   target_t=' + str(np.round(target_t, 6).tolist()), file=sys.stderr, flush=True)'''

content = content.replace(old, new)
with open('/home/man/puzzle_app/puzzle_vision/solver.py', 'w') as f:
    f.write(content)
print('DONE')
"
"""
stdin, stdout, stderr = ssh.exec_command(cmd)
out = stdout.read().decode()
err = stderr.read().decode()
print(out)
if err:
    print('ERR:', err[:500])

ssh.close()
