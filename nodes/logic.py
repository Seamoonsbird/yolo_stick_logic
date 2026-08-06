"""
盲杖辅助逻辑节点 — 帧驱动架构。

功能:
  1. 盲道检测   — 提醒走盲道正中间（偏移超阈值时语音纠正）
  2. 楼梯检测   — 上/下楼梯提醒（近距离告警）
  3. 水坑/井盖  — 前方障碍物提醒（高优先级）
  4. 草地/斑马线 — 仅正前方提醒（侧前方忽略，避免误报）

输出:
  - 控制台语音文案（可接入 TTS）
  - state_q → 录制画面叠加状态信息
"""

from __future__ import annotations

import time
import queue
import datetime
from typing import Optional, List
from config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)

# ==========================================
# 几何常量
# ==========================================

FRAME_WIDTH = CAMERA_WIDTH
FRAME_HEIGHT = CAMERA_HEIGHT
CENTER_X = FRAME_WIDTH // 2       # 320
CENTER_Y = FRAME_HEIGHT // 2      # 240

# ---- 空间分区 ----

# 正前方区域：画面中央竖条（水平方向中间 50%）
AHEAD_ZONE_WIDTH = FRAME_WIDTH // 2                     # 320px
AHEAD_ZONE_X1 = CENTER_X - AHEAD_ZONE_WIDTH // 2        # 160
AHEAD_ZONE_X2 = CENTER_X + AHEAD_ZONE_WIDTH // 2        # 480

# 侧前方区域：正前方以外的两侧（草地/斑马线在这些区域不提醒）
SIDE_LEFT_X2 = AHEAD_ZONE_X1                             # x < 160 = 左侧
SIDE_RIGHT_X1 = AHEAD_ZONE_X2                            # x > 480 = 右侧

# 近距离区域：画面下方 1/3（y 越大越近）
NEAR_ZONE_Y = FRAME_HEIGHT * 2 // 3                      # y > 320 = 近距离

# 中距离区域
MID_ZONE_Y = FRAME_HEIGHT // 3                           # y > 160 = 中距离以上

# ==========================================
# ⚙️ 盲杖参数
# ==========================================

# 盲道偏移容忍度（px）。盲道中心偏离画面中线超过此值 → 提醒纠正
BLIND_PATH_OFFSET_TOLERANCE = 50

# 盲道丢失容忍帧数（连续多少帧没检测到盲道才认为丢失）
BLIND_PATH_LOSS_FRAMES = 15

# 目标最小置信度（低于此值的检测忽略）
MIN_CONFIDENCE = 0.5

# 近距离判断：目标底部 y2 超过此值视为「即将到达」
DANGER_ZONE_Y = FRAME_HEIGHT * 4 // 5                    # y2 > 384 = 危险距离

# ---- 告警冷却时间（秒），防止频繁重复提醒 ----
ALERT_COOLDOWN = {
    "blind_path":   3.0,    # 盲道偏移提醒
    "upstairs":     3.0,    # 上楼梯
    "downstairs":   3.0,    # 下楼梯
    "water":        2.0,    # 水坑（高频危险）
    "well_lid":     2.0,    # 井盖（高频危险）
    "grass":        5.0,    # 草地（场景变化）
    "crosswalk":    5.0,    # 斑马线（场景变化）
    "wood":         5.0,    # 木板（特殊地形）
    "general":      1.5,    # 通用最短间隔
}

# ==========================================
# 类别名映射（需与模型训练时的类别名一致）
# 模型类别: {0:'blind_path',1:'well_lid',2:'crosswalk',3:'grass',4:'upstairs',5:'downstairs',6:'wood',7:'water'}
# ==========================================

# 盲道
CLASS_BLIND_PATH = "blind_path"

# 楼梯
CLASS_UPSTAIRS = "upstairs"
CLASS_DOWNSTAIRS = "downstairs"

# 障碍物
CLASS_WATER = "water"
CLASS_WELL_LID = "well_lid"

# 地形（仅正前方提醒，侧方忽略）
CLASS_GRASS = "grass"
CLASS_CROSSWALK = "crosswalk"
CLASS_WOOD = "wood"

# ---- 按类别分组，方便批量处理 ----
HAZARD_CLASSES = {
    CLASS_UPSTAIRS:   "上楼梯",
    CLASS_DOWNSTAIRS: "下楼梯",
    CLASS_WATER:      "水坑",
    CLASS_WELL_LID:   "井盖",
}

TERRAIN_CLASSES = {
    CLASS_GRASS:     "草地",
    CLASS_CROSSWALK: "斑马线",
    CLASS_WOOD:      "木板",
}


# ==========================================
# 几何工具
# ==========================================

def box_center(box):
    """包围盒中心点。"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_bottom_center(box):
    """包围盒底部中点（物体与地面接触点）。"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def box_area(box):
    """包围盒面积。"""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def is_in_ahead_zone(box):
    """
    判断目标是否在正前方区域。
    规则：包围盒中心 X 落在 [AHEAD_ZONE_X1, AHEAD_ZONE_X2]。
    """
    cx, _ = box_center(box)
    return AHEAD_ZONE_X1 <= cx <= AHEAD_ZONE_X2


def is_in_side_zone(box):
    """判断目标是否在侧前方区域。"""
    return not is_in_ahead_zone(box)


def is_nearby(box):
    """判断目标是否在近距离（画面下方）。"""
    _, y2 = box_bottom_center(box)
    return y2 >= NEAR_ZONE_Y


def is_danger_close(box):
    """判断目标是否在危险距离（非常近）。"""
    _, y2 = box_bottom_center(box)
    return y2 >= DANGER_ZONE_Y


# ==========================================
# 检测分类 & 筛选
# ==========================================

def classify_detections(det_results):
    """
    按类别名分组。
    返回:
      {
        "blind_path": [det, ...],
        "stairs_up":  [det, ...],
        "stairs_down":[det, ...],
        "puddle":     [det, ...],
        "manhole":    [det, ...],
        "grass":      [det, ...],
        "crosswalk":  [det, ...],
      }
    """
    groups = {k: [] for k in (
        CLASS_BLIND_PATH, CLASS_UPSTAIRS, CLASS_DOWNSTAIRS,
        CLASS_WATER, CLASS_WELL_LID, CLASS_GRASS, CLASS_CROSSWALK, CLASS_WOOD,
    )}

    for det in det_results:
        name = det.get("name", "")
        conf = det.get("conf", 0)
        if conf < MIN_CONFIDENCE:
            continue
        if name in groups:
            groups[name].append(det)

    return groups


def filter_ahead_only(detections):
    """只保留正前方的检测（用于草地/斑马线）。"""
    return [d for d in detections if is_in_ahead_zone(d["box"])]


def filter_nearby(detections):
    """只保留近距离的检测。"""
    return [d for d in detections if is_nearby(d["box"])]


def select_largest(detections):
    """选面积最大的检测结果（通常是最近的）。"""
    if not detections:
        return None
    return max(detections, key=lambda d: box_area(d["box"]))


# ==========================================
# 告警管理
# ==========================================

class AlertManager:
    """
    告警冷却管理器。
    同一类别在冷却时间内不会重复触发，避免刷屏。
    """

    def __init__(self):
        self._last_alert_time: dict[str, float] = {}

    def can_alert(self, category: str) -> bool:
        """检查该类告警是否已过冷却期。"""
        now = time.time()
        cooldown = ALERT_COOLDOWN.get(category, ALERT_COOLDOWN["general"])
        last = self._last_alert_time.get(category, 0)
        return (now - last) >= cooldown

    def mark_alert(self, category: str):
        """记录告警时间。"""
        self._last_alert_time[category] = time.time()


# ==========================================
# TTS 语音输出
# ==========================================

import subprocess
import threading

# TTS 引擎选择: "espeak" | "pyttsx3" | "print"
# espeak 最可靠（命令行直接调 ALSA），适合 Docker + Jetson
TTS_ENGINE = "print"

# espeak-ng 中文语音参数
ESPEAK_VOICE = "zh"          # 普通话
ESPEAK_SPEED = 160           # 语速 (词/分钟)，默认 175，稍慢更清晰
ESPEAK_VOLUME = 100          # 音量 0-200

# pyttsx3 引擎实例（延迟初始化，避免没装库时崩溃）
_pyttsx_engine = None
_pyttsx_lock = threading.Lock()

# espeak 子进程锁（避免并发调用互相覆盖）
_espeak_lock = threading.Lock()


def _tts_espeak(text: str) -> bool:
    """
    使用 espeak-ng 命令行 TTS。
    返回 True 表示播放成功。
    """
    try:
        cmd = [
            "espeak-ng",
            "-v", ESPEAK_VOICE,
            "-s", str(ESPEAK_SPEED),
            "-a", str(ESPEAK_VOLUME),
            "--",
            text,
        ]
        # 不等待 TTS 播放完毕，后台异步播放避免阻塞推理循环
        with _espeak_lock:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        return True
    except FileNotFoundError:
        return False
    except Exception:
        return False


def _tts_pyttsx3(text: str) -> bool:
    """
    使用 pyttsx3 库进行 TTS。
    返回 True 表示播放成功。
    """
    global _pyttsx_engine
    try:
        with _pyttsx_lock:
            if _pyttsx_engine is None:
                import pyttsx3
                _pyttsx_engine = pyttsx3.init()
                # 尝试设置中文语音
                voices = _pyttsx_engine.getProperty("voices")
                for v in voices:
                    if "zh" in v.id.lower() or "chinese" in v.name.lower() or "mandarin" in v.name.lower():
                        _pyttsx_engine.setProperty("voice", v.id)
                        break
                _pyttsx_engine.setProperty("rate", 160)
                _pyttsx_engine.setProperty("volume", 1.0)
            _pyttsx_engine.say(text)
            _pyttsx_engine.runAndWait()
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _speak_text(text: str):
    """
    根据 TTS_ENGINE 配置选择 TTS 方式，异步执行避免阻塞主循环。
    """
    def _run():
        if TTS_ENGINE == "espeak":
            if _tts_espeak(text):
                return
        elif TTS_ENGINE == "pyttsx3":
            if _tts_pyttsx3(text):
                return
        # 所有 TTS 都失败 → fallback 到 espeak（可能是 pyttsx3 失败）
        if TTS_ENGINE != "espeak" and _tts_espeak(text):
            return
        # 最终兜底：无法发声，仅打印到控制台
        print(f"[盲杖] 🔇 语音引擎不可用，尝试过 espeak-ng / pyttsx3 均失败。请安装: apt install espeak-ng")

    # 在后台线程中运行 TTS，不阻塞检测循环
    t = threading.Thread(target=_run, daemon=True)
    t.start()


# 最近告警记录（模块级，供 speak 写入、主循环读取塞进 state_q）
_recent_alerts: list[str] = []


def get_recent_alerts() -> list[str]:
    """获取最近告警列表（供主循环写入 status 字典）。"""
    return _recent_alerts.copy()


# Speak 的对外接口（受告警冷却管理，底层走 TTS 引擎）
def speak(text: str, category: str = "general", alert_mgr: Optional[AlertManager] = None):
    """
    语音播报（受冷却管理控制）。
    - 冷却期内自动跳过
    - 后台线程异步 TTS，不阻塞推理
    - 告警文案自动记录到 _recent_alerts（供视频叠加显示）
    """
    if alert_mgr is not None and not alert_mgr.can_alert(category):
        return

    print(f"[盲杖] 🗣 {text}")

    # 记录告警到模块级列表（视频叠加用）
    now = datetime.datetime.now().strftime("%H:%M:%S")
    _recent_alerts.append(f"[{now}] {text}")
    if len(_recent_alerts) > 10:
        _recent_alerts.pop(0)

    _speak_text(text)

    if alert_mgr is not None:
        alert_mgr.mark_alert(category)


# ==========================================
# 各类检测逻辑
# ==========================================

def handle_blind_path(groups, alert_mgr, status: dict):
    """
    盲道检测：判断是否走在盲道正中间。
    - 检测到盲道 → 计算盲道区域中心 vs 画面中心的偏移
    - 偏移超阈值 → 提醒纠正方向
    - 未检测到盲道 → 提醒寻找盲道
    """
    blind_paths = groups.get(CLASS_BLIND_PATH, [])

    if not blind_paths:
        status["盲道"] = "未检测到"
        return

    # 取面积最大的盲道区域（主盲道）
    main_path = select_largest(blind_paths)
    if main_path is None:
        status["盲道"] = "未检测到"
        return

    cx, _ = box_center(main_path["box"])
    offset = cx - CENTER_X

    # 盲道区域宽度
    x1, _, x2, _ = main_path["box"]
    path_width = x2 - x1

    if abs(offset) <= BLIND_PATH_OFFSET_TOLERANCE:
        status["盲道"] = f"居中 (偏移{offset:+.0f}px)"
        return

    direction = "右" if offset < 0 else "左"
    status["盲道"] = f"偏{direction} (偏移{offset:+.0f}px)"

    speak(f"盲道偏{direction}，请向{direction}调整", "blind_path", alert_mgr)


def handle_stairs(groups, alert_mgr, status: dict):
    """
    楼梯检测：上下楼梯提醒。
    - 近距离楼梯 → 高危提醒
    - 中距离楼梯 → 提前预警
    """
    for cls_name, label in HAZARD_CLASSES.items():
        if cls_name not in (CLASS_UPSTAIRS, CLASS_DOWNSTAIRS):
            continue

        detections = groups.get(cls_name, [])
        if not detections:
            continue

        # 只看正前方的楼梯
        ahead = filter_ahead_only(detections)
        if not ahead:
            continue

        nearest = select_largest(ahead)
        if nearest is None:
            continue

        if is_danger_close(nearest["box"]):
            status[label] = "⚠️危险距离"
            speak(f"前方{label}，请小心慢行", cls_name, alert_mgr)
        elif is_nearby(nearest["box"]):
            status[label] = "🔶近距离"
            speak(f"前方{label}，请注意", cls_name, alert_mgr)
        else:
            status[label] = "远距离"


def handle_obstacles(groups, alert_mgr, status: dict):
    """
    水坑/井盖检测：前方障碍物提醒。
    - 正前方 + 近距离 → 告警
    - 侧前方 → 仅记录不告警
    """
    for cls_name, label in [
        (CLASS_WATER, "水坑"),
        (CLASS_WELL_LID, "井盖"),
    ]:
        detections = groups.get(cls_name, [])
        if not detections:
            continue

        ahead = filter_ahead_only(detections)
        nearby_ahead = filter_nearby(ahead)

        if nearby_ahead:
            status[label] = "⚠️前方危险"
            speak(f"前方有{label}，请绕行", cls_name, alert_mgr)
        elif ahead:
            status[label] = "前方远处"
            speak(f"前方远处有{label}，请注意", cls_name, alert_mgr)
        else:
            # 侧方有障碍，记录但不语音提醒
            status[label] = "侧方"


def handle_terrain(groups, alert_mgr, status: dict):
    """
    地形检测（草地/斑马线/木板）：仅正前方提醒，侧前方忽略。
    - 正前方 + 近距离 → 提醒地面变化
    - 侧前方 → 不提醒（用户不会走到侧面）
    """
    terrain_items = [
        (CLASS_GRASS,     "草地"),
        (CLASS_CROSSWALK, "斑马线"),
        (CLASS_WOOD,      "木板"),
    ]

    for cls_name, label in terrain_items:
        detections = groups.get(cls_name, [])
        if not detections:
            continue

        # 关键逻辑：只看正前方，侧前方忽略
        ahead = filter_ahead_only(detections)
        side_count = len([d for d in detections if is_in_side_zone(d["box"])])

        if not ahead:
            if side_count > 0:
                status[label] = f"侧方有{side_count}处(忽略)"
            continue

        nearest = select_largest(ahead)
        if nearest is None:
            continue

        if is_danger_close(nearest["box"]):
            status[label] = "⚠️正前方近距离"
            if cls_name == CLASS_CROSSWALK:
                speak(f"前方斑马线，请注意来往车辆", cls_name, alert_mgr)
            elif cls_name == CLASS_WOOD:
                speak(f"前方木板路，请注意脚下", cls_name, alert_mgr)
            else:
                speak(f"前方草地，请注意脚下", cls_name, alert_mgr)
        elif is_nearby(nearest["box"]):
            status[label] = "🔶正前方"
            speak(f"前方{label}，请注意", cls_name, alert_mgr)
        else:
            # 远处不提醒，避免频繁干扰
            status[label] = "前方远处"


# ==========================================
# 主循环
# ==========================================

def logic_worker(result_q, recorder_q, stop_event, state_q=None):
    """
    盲杖逻辑主循环。

    参数:
      result_q     — 推理结果队列，元素为 [{"box":[x1,y1,x2,y2], "cls":int, "conf":float, "name":str}, ...]
      recorder_q   — 录制队列（本模块不使用，保留接口兼容）
      stop_event   — 多进程停止事件
      state_q      — 状态队列 → 录制节点叠加显示
    """

    print("[盲杖] 盲杖辅助系统启动")
    print(f"[盲杖] 画面尺寸: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[盲杖] 正前方区域: x∈[{AHEAD_ZONE_X1}, {AHEAD_ZONE_X2}]")
    print(f"[盲杖] 近距离阈值: y>{NEAR_ZONE_Y}, 危险距离: y>{DANGER_ZONE_Y}")
    print(f"[盲杖] 盲道偏移容忍: ±{BLIND_PATH_OFFSET_TOLERANCE}px")
    print(f"[盲杖] 侧前方草地/斑马线 → 不提醒")
    print()

    alert_mgr = AlertManager()
    frame_count = 0
    loss_count = 0          # 盲道连续丢失帧计数
    status: dict[str, str] = {}

    try:
        while not stop_event.is_set():
            # ======================================
            # 取帧（逐帧取，保持最新）
            # ======================================
            if result_q.empty():
                time.sleep(0.01)
                continue

            # 排空队列，只保留最新一批检测结果
            latest_detections = []
            while not result_q.empty():
                try:
                    latest_detections = result_q.get_nowait()
                except queue.Empty:
                    break

            frame_count += 1

            # ======================================
            # 检测分类
            # ======================================
            groups = classify_detections(latest_detections)

            # ---- 盲道丢失计数 ----
            if groups.get(CLASS_BLIND_PATH):
                loss_count = 0
            else:
                loss_count += 1

            # ---- 重置状态 ----
            status = {
                "帧": str(frame_count),
                "最近提醒": get_recent_alerts(),
            }

            # ======================================
            # 各模块检测 & 提醒
            # ======================================

            # 1. 盲道居中检测（最高频）
            handle_blind_path(groups, alert_mgr, status)

            # 盲道长时间丢失提醒
            if loss_count >= BLIND_PATH_LOSS_FRAMES and loss_count == BLIND_PATH_LOSS_FRAMES:
                speak("未检测到盲道，请用盲杖探寻", "blind_path", alert_mgr)

            # 2. 楼梯检测（上下楼梯）
            handle_stairs(groups, alert_mgr, status)

            # 3. 水坑/井盖检测（障碍物）
            handle_obstacles(groups, alert_mgr, status)

            # 4. 草地/斑马线检测（仅正前方）
            handle_terrain(groups, alert_mgr, status)

            # ======================================
            # 诊断日志（每 30 帧汇总）
            # ======================================
            if frame_count % 30 == 0:
                total_det = len(latest_detections)
                parts = [f"{k}={len(v)}" for k, v in groups.items() if v]
                detail = ", ".join(parts) if parts else "无"
                print(f"[盲杖] 帧#{frame_count} | 检测总数={total_det} | {detail}")

            # ======================================
            # 状态推送到录制节点（非阻塞）
            # ======================================
            if state_q is not None:
                if state_q.full():
                    try:
                        state_q.get_nowait()
                    except queue.Empty:
                        pass
                state_q.put(status)

    except Exception as e:
        print(f"[盲杖] 崩溃: {e}")
        import traceback
        traceback.print_exc()

    print("[盲杖] 已停止")
