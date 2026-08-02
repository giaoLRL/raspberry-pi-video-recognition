#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口屏测试脚本 — TJC 淘晶驰 USART HMI
=====================================
连接: GPIO4=TXD2, GPIO5=RXD2 → /dev/ttyAMA2
协议: 控件.属性=值 + 帧尾 \xff\xff\xff
"""

import sys
import time
import struct
import threading
from typing import Any

# ---- 配置 ----
SERIAL_PORT = "/dev/ttyAMA2"
BAUD_RATES = [115200, 9600, 57600, 38400, 19200]  # 常用波特率
FRAME_END = b"\xff\xff\xff"  # TJC 指令帧尾


def find_serial():
    """导入 pyserial，失败则提示安装。"""
    try:
        import serial
        return serial
    except ImportError:
        print("[FAIL] pyserial 未安装，请在 Pi 上执行:")
        print("       pip install pyserial")
        sys.exit(1)


def open_port(serial, port: str, baud: int) -> Any:
    """打开串口。"""
    try:
        ser = serial.Serial(port, baud, timeout=0.5, write_timeout=1.0)
        print(f"  ✓ {port} @ {baud} 已打开")
        return ser
    except Exception as exc:
        print(f"  ✗ {port} @ {baud} 打开失败: {exc}")
        return None


def send_command(ser, cmd: str, end: bytes = FRAME_END, encoding: str = "gbk") -> bytes:
    """发送指令并读取应答。

    TJC 屏文本部分使用 GBK 编码。ASCII 是 GBK 的子集，
    所以纯英文指令用 gbk 编码也不会有问题。
    """
    raw = cmd.encode(encoding) + end
    print(f"\n[TX] {cmd}")
    ser.reset_input_buffer()
    ser.write(raw)
    ser.flush()
    time.sleep(0.15)
    reply = b""
    while ser.in_waiting:
        chunk = ser.read(ser.in_waiting)
        reply += chunk
        time.sleep(0.05)
    if reply:
        # TJC 回传格式: 0x01/0x02/0x03 + payload + 0xff 0xff 0xff
        # 尝试提取有效文本
        try:
            clean = reply.rstrip(b"\xff")
            if clean.startswith(b"\x01"):
                print(f"[RX] (成功确认) {clean[1:].decode(encoding, errors='replace').strip() or 'OK'}")
            elif clean.startswith(b"\x02"):
                print(f"[RX] (失败) {clean[1:].decode(encoding, errors='replace').strip()}")
            elif clean.startswith(b"\x03"):
                print(f"[RX] (数据) {clean[1:].decode(encoding, errors='replace').strip()}")
            else:
                text = clean.decode(encoding, errors="replace").strip()
                print(f"[RX] {text}")
        except Exception:
            print(f"[RX] (binary) {reply.hex()}")
    else:
        print("[RX] (无应答)")
    return reply


def test_connection(ser) -> bool:
    """发送版本查询，检验是否为 TJC 串口屏。"""
    print("\n--- 连接检测 ---")

    # 先开回传模式（回传所有指令状态），然后查版本
    send_command(ser, "bkcmd=2")   # 0=不回传 1=失败才回 2=全回 3=原始
    time.sleep(0.1)
    send_command(ser, "bkcmd=3")   # 尝试原始模式
    time.sleep(0.1)
    reply = send_command(ser, "get version")
    if reply:
        return True

    # 部分固件用不同命令
    reply = send_command(ser, "prints \"HELLO_TJC\",0")
    if reply:
        return True

    # 再试: 获取设备型号
    reply = send_command(ser, "get model")
    if reply:
        return True

    return False


def draw_test_pattern(ser):
    """在屏幕上绘制测试图案。"""
    print("\n--- 绘制测试图案 ---")

    # 切换到页面 0（默认页）
    send_command(ser, "page 0")
    time.sleep(0.2)

    # 清屏为白色
    send_command(ser, "cls 65535")
    time.sleep(0.3)

    # 在不同位置画彩色矩形
    print("\n[画矩形]")
    colors_rect = [
        (0, 0, 100, 100, 63488),    # 红色 (左上)
        (100, 0, 100, 100, 2016),   # 绿色 (上中)
        (200, 0, 100, 100, 31),     # 蓝色 (右上)
        (0, 100, 100, 100, 65504),  # 黄色 (左下)
        (100, 100, 100, 100, 31),   # 青色
        (200, 100, 100, 100, 63519),# 品红 (右下)
    ]
    for x, y, w, h, color in colors_rect:
        send_command(ser, f"fill {x},{y},{w},{h},{color}")
        time.sleep(0.08)

    time.sleep(0.3)

    # 画线条
    print("\n[画线]")
    send_command(ser, "line 0,0,300,200,0")      # 黑色对角线
    time.sleep(0.1)
    send_command(ser, "line 0,200,300,0,65535")  # 白色对角线
    time.sleep(0.1)
    send_command(ser, "line 150,0,150,200,0")     # 垂直中线
    time.sleep(0.1)
    send_command(ser, "line 0,100,300,100,0")     # 水平中线

    time.sleep(0.3)

    # 画圆
    print("\n[画圆]")
    send_command(ser, "cir 150,100,60,63488")     # 红色大圆
    time.sleep(0.1)
    send_command(ser, "cir 150,100,30,31")         # 蓝色小圆
    time.sleep(0.1)

    # 画文字
    print("\n[写文字]")
    # xstr x,y,w,h,font,bgc,fgc,xalign,yalign,text
    send_command(ser, 'xstr 0,0,300,40,0,65535,0,1,1,"串口屏测试 OK"')
    time.sleep(0.1)
    send_command(ser, 'xstr 0,200,300,40,0,65535,0,1,1,"TJC Screen Test"')

    print("\n✓ 测试图案绘制完成")


def detect_baud(serial) -> tuple:
    """自动探测正确的波特率。"""
    print("\n===== 波特率自动探测 =====")
    for baud in BAUD_RATES:
        print(f"\n尝试波特率: {baud}")
        ser = open_port(serial, SERIAL_PORT, baud)
        if ser is None:
            continue
        try:
            if test_connection(ser):
                print(f"\n★ 检测成功! 波特率 = {baud}")
                return ser, baud
            ser.close()
        except Exception as exc:
            print(f"  错误: {exc}")
            try:
                ser.close()
            except Exception:
                pass
    return None, 0


def interactive(ser):
    """交互模式：手动输入指令测试。"""
    print("\n===== 交互模式 =====")
    print("输入 TJC 指令（不含帧尾），帧尾自动添加。")
    print("输入 'quit' 退出, 'help' 查看示例。")

    examples = [
        ("page 0",          "切换到第0页"),
        ("cls 65535",       "清屏为白色 (65535=白色, 0=黑色)"),
        ("cls 63488",       "清屏为红色"),
        ("fill 10,10,100,80,63488", "画矩形 x,y,w,h,color"),
        ("line 0,0,320,240,0",      "画线 x1,y1,x2,y2,color"),
        ("cir 160,120,50,31",       "画圆 x,y,r,color"),
        ('xstr 0,0,320,30,0,65535,0,1,1,"Hello"', "写文字"),
        ("bkcmd=2",         "开启全部回传"),
        ("bkcmd=0",         "关闭回传(仅生产用)"),
        ("get version",      "查询固件版本"),
    ]

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not cmd:
            continue
        if cmd.lower() == "quit":
            break
        if cmd.lower() == "help":
            print("\n常用指令示例:")
            for ex_cmd, desc in examples:
                print(f"  {ex_cmd}")
                print(f"    → {desc}")
            continue

        send_command(ser, cmd)


# ================================================================
# 主入口
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  串口屏测试 — TJC USART HMI")
    print(f"  端口: {SERIAL_PORT}  (GPIO4=TXD2, GPIO5=RXD2)")
    print("=" * 60)

    serial = find_serial()

    # 先尝试用户指定的波特率
    if len(sys.argv) > 1:
        baud = int(sys.argv[1])
        print(f"\n使用指定波特率: {baud}")
        ser = open_port(serial, SERIAL_PORT, baud)
        if ser is None:
            sys.exit(1)
        if not test_connection(ser):
            print("[WARN] 未检测到 TJC 设备回应，尝试继续...")
    else:
        ser, baud = detect_baud(serial)
        if ser is None:
            print("\n[FAIL] 所有波特率均未检测到设备。")
            print("请检查:")
            print("  1. 接线: GPIO4→RX, GPIO5→TX (交叉)")
            print("  2. 供电: 串口屏需独立 5V 供电")
            print("  3. 屏幕是否已烧录 TJC 工程")
            print("  4. 手动指定波特率: python test_serial_screen.py 115200")
            sys.exit(1)

    try:
        draw_test_pattern(ser)
        interactive(ser)
    finally:
        ser.close()
        print("\n串口已关闭")
