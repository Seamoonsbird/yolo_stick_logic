#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 转 TTL 串口测试工具（在 Docker 容器内运行）。

用来验证「容器 → USB 转 TTL → 单片机」这条链路是否通，分两步：

  1) 回环自测（不用单片机）：把 USB 转 TTL 的 TX 和 RX 用杜邦线短接，
     发什么就应该原样收到什么。能通过说明驱动、权限、波特率都没问题。
  2) 真实收发：接上单片机，发命令帧、看回包。

命令帧默认沿用本项目已有的格式（logic_test.py 里的协议）：
    0xAA  0x02  <cmd>  0x55

用法（容器内）：
    pip install pyserial

    python test/serial_test.py --list                    # 看容器里有没有串口
    python test/serial_test.py --loopback                # 回环自测（TX-RX 短接）
    python test/serial_test.py --loopback --hex "01 02"  # 回环自测，自定义数据
    python test/serial_test.py --cmd 0x10 --listen 1     # 发一帧 AA 02 10 55，听 1 秒回包
    python test/serial_test.py --text "hello" --baud 9600 --repeat 5
    python test/serial_test.py --repeat 0 --listen 10    # 只监听，不发

常见坑：
  - 容器里 `ls /dev/ttyUSB*` 什么都没有
        → 设备没直通。本项目 run_yolo_docker.sh 用了 --privileged，
          正常能看到；看不到见 README 排查段落。
  - Permission denied
        → 容器内一般已是 root；宿主机则需把用户加进 dialout 组。
  - 一插上设备节点就消失
        → Ubuntu 的 brltty 会把 ttyUSB 当成盲文点显器抢走：
          sudo systemctl stop brltty-udev.service brltty.service
  - 回环收不到数据
        → 检查 TX/RX 是否真的短接、杜邦线是否接触不良、是不是把
          3.3V/5V 和 GND 也接错位置。
"""

import argparse
import glob
import sys
import time
from typing import Optional, Tuple

# 与 logic_test.py 保持一致的单片机帧格式
FRAME_HEAD = 0xAA
FRAME_TAIL = 0x55
DEFAULT_CMD = 0x10


# ============================================================
# pyserial（延迟导入，缺库时给出清晰提示）
# ============================================================

def _pyserial():
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        sys.exit(
            "未安装 pyserial。请在容器内执行：\n"
            "    pip install pyserial\n"
            "注意：容器重建后要重装，想永久保留请写进 Dockerfile。"
        )
    return serial, list_ports


def hexs(data: bytes) -> str:
    """字节串 → 'AA 02 10 55' 形式，便于肉眼看。"""
    return " ".join(f"{b:02X}" for b in data) if data else "(空)"


def ascii_of(data: bytes) -> str:
    """可打印字符原样显示，其余用 . 占位。"""
    return "".join(chr(b) if 32 <= b < 127 else "." for b in data)


# ============================================================
# 设备发现
# ============================================================

def print_dev_nodes():
    """直接看 /dev 下的串口节点（pyserial 列不出来时的兜底排查）。"""
    print("\n/dev 下的串口节点：")
    nodes = sorted(
        glob.glob("/dev/ttyUSB*")
        + glob.glob("/dev/ttyACM*")
        + glob.glob("/dev/ttyTHS*")
    )
    if not nodes:
        print("  (无)")
    for n in nodes:
        print(f"  {n}")

    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        print("\n/dev/serial/by-id/（推荐用它，插拔后名字不变）：")
        for n in by_id:
            print(f"  {n}")


def list_ports() -> list:
    """列出 pyserial 能识别的所有串口。"""
    _, list_ports_mod = _pyserial()
    ports = sorted(list_ports_mod.comports(), key=lambda p: p.device)

    if not ports:
        print("pyserial 没有发现任何串口设备。")
        print_dev_nodes()
        return []

    print(f"发现 {len(ports)} 个串口：")
    for p in ports:
        vid = f"{p.vid:04X}" if p.vid is not None else "----"
        pid = f"{p.pid:04X}" if p.pid is not None else "----"
        print(f"  {p.device:<24} VID:PID={vid}:{pid}  {p.description}")
        if p.hwid:
            print(f"    {p.hwid}")
    print_dev_nodes()
    return ports


def pick_port(explicit: Optional[str]) -> str:
    """选端口：用户指定 > by-id > 第一个 ttyUSB/ttyACM。"""
    if explicit:
        return explicit

    # by-id 带芯片序列号，ttyUSB0/1 互换时也不会选错
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    if by_id:
        return by_id[0]

    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]

    sys.exit(
        "找不到串口设备。请先确认：\n"
        "  1. 容器内执行 ls -l /dev/ttyUSB* /dev/ttyACM*\n"
        "  2. 没有的话重新跑 sh/run_yolo_docker.sh（--privileged 已能直通）\n"
        "  3. USB 线是数据线，不是只供电的充电线\n"
        "  4. 检查 brltty 是否抢占了设备"
    )


# ============================================================
# 数据构造
# ============================================================

def parse_hex(text: str) -> bytes:
    """'AA 02 10 55' / 'aa,02,10,55' / '0xAA 0x02' → bytes。"""
    cleaned = text.replace(",", " ").replace("0x", " ").replace("0X", " ")
    parts = cleaned.split()
    try:
        return bytes(int(p, 16) for p in parts)
    except ValueError as exc:
        sys.exit(f"无法解析十六进制 \"{text}\"：{exc}")


def build_payload(args) -> bytes:
    if args.cmd is not None:
        # AA 02 <cmd> 55：与本项目 logic_test.py 的帧格式一致
        return bytes([FRAME_HEAD, 0x02, args.cmd & 0xFF, FRAME_TAIL])
    if args.hex_data:
        return parse_hex(args.hex_data)
    if args.text:
        return args.text.encode("utf-8")
    return bytes([FRAME_HEAD, 0x02, DEFAULT_CMD, FRAME_TAIL])


# ============================================================
# 收发
# ============================================================

def open_port(device: str, baud: int, timeout: float):
    serial, _ = _pyserial()
    try:
        ser = serial.Serial(device, baud, timeout=timeout)
    except (serial.SerialException, OSError) as exc:
        sys.exit(
            f"打开 {device} 失败：{exc}\n"
            "排查方向：\n"
            "  - Permission denied → 用 root 或在宿主机加入 dialout 组\n"
            "  - Device or resource busy → 有别的程序占用（如 brltty、minicom）\n"
            "  - 设备已拔出 → ls /dev/ttyUSB* 确认节点还在"
        )
    return ser


def read_for(ser, seconds: float) -> bytes:
    """在 seconds 秒内尽量多读，返回收到的全部字节。"""
    buf = bytearray()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        if waiting:
            buf += ser.read(waiting)
        else:
            time.sleep(0.02)
    return bytes(buf)


def loopback_test(ser, payload: bytes, repeat: int, interval: float) -> Tuple[int, int]:
    """短接 TX-RX 后，发什么就该收到什么。返回 (成功次数, 失败次数)。"""
    ok = fail = 0
    for i in range(repeat):
        ser.reset_input_buffer()          # 清掉上一轮回显，避免错位
        t0 = time.perf_counter()
        ser.write(payload)
        ser.flush()

        echoed = ser.read(len(payload))
        dt_ms = (time.perf_counter() - t0) * 1000

        if echoed == payload:
            ok += 1
            print(f"  [{i + 1}/{repeat}] 回环 OK   {dt_ms:6.1f}ms   {hexs(echoed)}")
        else:
            fail += 1
            print(
                f"  [{i + 1}/{repeat}] 回环 FAIL {dt_ms:6.1f}ms   "
                f"期望 {hexs(payload)} / 实收 {hexs(echoed)}"
            )
            if not echoed:
                print("          → 一个字节都没回来：TX/RX 没短接？还是线插错口了？")

        if interval > 0 and i + 1 < repeat:
            time.sleep(interval)
    return ok, fail


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="USB 转 TTL 串口测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s --list\n"
            "  %(prog)s --loopback\n"
            "  %(prog)s --cmd 0x10 --listen 1\n"
            "  %(prog)s --repeat 0 --listen 10"
        ),
    )
    parser.add_argument("--list", action="store_true", help="只列出串口，不打开")
    parser.add_argument("--port", help="串口设备，默认自动选第一个")
    parser.add_argument("--baud", type=int, default=115200, help="波特率（默认 115200）")
    parser.add_argument("--timeout", type=float, default=0.5, help="单次读超时秒数（默认 0.5）")
    parser.add_argument("--repeat", type=int, default=1, help="发送次数，0 表示只监听")
    parser.add_argument("--interval", type=float, default=0.3, help="两次发送间隔秒数")
    parser.add_argument("--listen", type=float, default=0.0, help="每次发送后监听回包的秒数")
    parser.add_argument("--loopback", action="store_true", help="回环自测（需短接 TX-RX）")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cmd", type=lambda s: int(s, 0),
                       help="命令字节，自动包成 AA 02 <cmd> 55，如 0x10")
    group.add_argument("--hex", dest="hex_data", help="直接发送十六进制字节，如 \"AA 02 10 55\"")
    group.add_argument("--text", help="直接发送字符串，如 \"hello\"")

    args = parser.parse_args()

    # 不带任何参数时，默认就是列出串口，减少误触
    if len(sys.argv) == 1:
        args.list = True

    if args.list:
        list_ports()
        return 0

    serial, _ = _pyserial()

    device = pick_port(args.port)
    ser = open_port(device, args.baud, args.timeout)
    print(f"串口已打开: {device} @ {args.baud} 8N1\n")

    try:
        # ---- 只监听模式 ----
        if args.repeat == 0:
            seconds = args.listen or 10.0
            print(f"只监听，不发数据（{seconds:.0f}s）...")
            data = read_for(ser, seconds)
            print(f"\n共收到 {len(data)} 字节：")
            print(f"  HEX: {hexs(data)}")
            print(f"  ASCII: {ascii_of(data) or '(空)'}")
            return 0

        payload = build_payload(args)
        print(f"待发送 {len(payload)} 字节: {hexs(payload)}")

        # ---- 回环自测 ----
        if args.loopback:
            print("回环模式：TX-RX 必须已短接\n")
            ok, fail = loopback_test(ser, payload, args.repeat, args.interval)
            print(f"\n结果: 成功 {ok} / 失败 {fail}")
            if ok and not fail:
                print("✅ 驱动、权限、波特率都正常，可以接单片机了")
            elif fail:
                print("❌ 回环不通，先别接单片机，按上面提示查线")
                return 1
            return 0

        # ---- 正常收发 ----
        ok = 0
        for i in range(args.repeat):
            ser.write(payload)
            ser.flush()
            print(f"  [{i + 1}/{args.repeat}] 已发送 {hexs(payload)}")

            if args.listen > 0:
                data = read_for(ser, args.listen)
                if data:
                    ok += 1
                    print(f"       收到 {len(data)} 字节: {hexs(data)}")
                    print(f"       ASCII: {ascii_of(data)}")
                else:
                    print(f"       {args.listen:.1f}s 内无回包")

            if args.interval > 0 and i + 1 < args.repeat:
                time.sleep(args.interval)

        if args.listen > 0:
            print(f"\n结果: {ok}/{args.repeat} 次收到回包")
            if ok == 0:
                print("提示: 单片机没回、TX/RX 没交叉、或波特率不对都会导致无回包")
        return 0

    except KeyboardInterrupt:
        print("\n已中断")
        return 0
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    sys.exit(main())
