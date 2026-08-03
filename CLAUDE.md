# Puzzle Robot Project

树莓派拼图机器人视觉识别系统。

## 环境架构

```
Windows PC (开发机)
  │
  ├─ R: 盘 = SMB 挂载 → \\man.local\man （树莓派 home 目录）
  │   改 R: 盘文件 = 直接改树莓派文件，无需 scp
  │
  └─ SSH → man@man.local （或 man@192.168.1.101）
      密钥: ~/.ssh/id_rsa （已加入 Pi 的 authorized_keys）
```

## 操作原则

**改文件 → R 盘直写。重启服务 → SSH。**

R: 盘就是树莓派的 `/home/man/`，直接读写，速度快、不会断。
只在杀进程/重启服务时才用 SSH。

```
# 读文件
Read /r/main.py

# 改文件
Edit /r/main.py

# 重启服务
ssh man@man.local "pkill -f main.py; sleep 1; cd /home/man/puzzle_robot_project && setsid python3 main.py >> /tmp/puzzle_main.log 2>&1 & disown"
```

## Shell 稳定性

**绝对不要在 R: 盘卡死时反复尝试命令。**

如果 `echo hello` 返回 `exit code 1` 无输出，说明 R: 盘 SMB 连接断了。立即执行：

```bash
cmd.exe /c "net use R: /delete /y && net use R: \\man.local\man /persistent:yes"
```

然后再试 `echo hello` 确认恢复。

## 树莓派服务管理

服务名: `main.py` (端口 8080) + `serial_monitor.py` (端口 8081)

重启流程：
```bash
# 1. 杀旧进程
ssh man@man.local "pkill -f main.py; sleep 1"

# 2. 等 SSH 恢复（kill 可能导致短暂断连）
sleep 5

# 3. 启动新进程（注意环境变量！）
ssh man@man.local "cd /home/man && QT_QPA_PLATFORM=offscreen nohup python3 /home/man/puzzle_robot_project/main.py > /tmp/puzzle_stream.log 2>&1 & disown"

# 4. 验证
ssh man@man.local "pgrep -a main && curl -s http://127.0.0.1:8080/status"
```

## 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| Bash `exit code 1` 无输出 | R: 盘 SMB 卡死 | `cmd.exe /c "net use R: ..."` |
| SSH `Permission denied` | 非交互无 TTY | 确保密钥已配，或 `python -c "import paramiko..."` |
| Python 进程 `Aborted` | 缺 `QT_QPA_PLATFORM=offscreen` | 启动命令必须带此环境变量 |
| Edit 匹配失败 | 文件每行代码后有空行 | Read 确认实际格式后再 Edit |
| 找不到树莓派 | IP 变了 | `ping man.local` 获取当前 IP |


## 坐标系

- **solver-mm**: TL 原点, X 右, Y 下
- **arm-mm**: A4 右边缘+75mm 原点, X 左, Y 下（传给机械臂的坐标）
- 转换: `_plan_to_arm()` → `solver_to_arm()`
