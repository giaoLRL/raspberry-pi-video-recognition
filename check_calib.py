import paramiko, json, numpy as np, cv2
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('192.168.31.93', username='man', password='giao666666', timeout=10)

# 取标定角点
_, o, _ = ssh.exec_command('cat /home/man/puzzle_app/a4_corners.json')
corners_data = o.read().decode()
print('--- A4 CORNERS ---')
print(corners_data)

# 取一帧图像尺寸
_, o, _ = ssh.exec_command('python3 -c "import cv2; c=cv2.VideoCapture(0); r,f=c.read(); print(f.shape if r else None); c.release()"')
print('--- FRAME SHAPE ---')
print(o.read().decode())

ssh.close()

# 分析
data = json.loads(corners_data)
corners = np.array(data['corners'], dtype=np.float32)
print('\n--- ANALYSIS ---')
print('corners shape:', corners.shape)
print('corners:\n', corners)

# A4纸在warp坐标系是 840x1188 (210mm x 297mm, 4px/mm)
# corners是相机图像中A4纸的4个角点
# 计算A4纸在相机图像中的像素宽高
# order: TL, TR, BR, BL (after order_points)
s = corners.sum(axis=1)
d = np.diff(corners, axis=1).reshape(-1)
ordered = np.zeros((4,2), dtype=np.float32)
ordered[0] = corners[np.argmin(s)]  # TL
ordered[1] = corners[np.argmin(d)]  # TR
ordered[2] = corners[np.argmax(s)]  # BR
ordered[3] = corners[np.argmax(d)]  # BL
print('ordered (TL,TR,BR,BL):\n', ordered)

w_top = np.linalg.norm(ordered[1]-ordered[0])    # 顶边像素
w_bot = np.linalg.norm(ordered[2]-ordered[3])    # 底边像素
h_left = np.linalg.norm(ordered[3]-ordered[0])   # 左边像素
h_right = np.linalg.norm(ordered[2]-ordered[1])  # 右边像素
print(f'A4 top width:    {w_top:.1f}px = {w_top/4:.1f}mm (should be 210mm)')
print(f'A4 bot width:    {w_bot:.1f}px = {w_bot/4:.1f}mm (should be 210mm)')
print(f'A4 left height:  {h_left:.1f}px = {h_left/4:.1f}mm (should be 297mm)')
print(f'A4 right height: {h_right:.1f}px = {h_right/4:.1f}mm (should be 297mm)')
print(f'A4 aspect in image (w/h): {w_top/h_left:.3f}  (physical 210/297={210/297:.3f})')

# 蓝框四角 (相机坐标)
box = np.array([[630,592],[1214,620],[1206,796],[622,768]], dtype=np.float32)
# 用 camera_to_warp 反变换回 warp 坐标
dst = np.array([[0,0],[839,0],[839,1187],[0,1187]], dtype=np.float32)
c2w = cv2.getPerspectiveTransform(ordered, dst)
box_warp = cv2.perspectiveTransform(box.reshape(1,-1,2), c2w).reshape(-1,2)
print('\n--- BLUE BOX IN WARP COORDS ---')
print('box_warp:', box_warp)
print('box_mm:', box_warp / 4.0)
# 计算蓝框在warp坐标系中的宽高
bw = np.linalg.norm(box_warp[1]-box_warp[0])
bh = np.linalg.norm(box_warp[3]-box_warp[0])
print(f'BlueBox warp width:  {bw:.1f}px = {bw/4:.1f}mm')
print(f'BlueBox warp height: {bh:.1f}px = {bh/4:.1f}mm')
print(f'BlueBox aspect (w/h): {bw/bh:.3f}  (expected 100/60={100/60:.3f})')
