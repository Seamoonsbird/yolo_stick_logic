# 音频资源目录（assets/sounds）

本目录存放语音播报相关的音频文件。

## 当前状态

- `fallback_alert.wav`：**占位提示音**（短蜂鸣），仅作为 espeak-ng 不可用时的兜底，
  保证系统至少有声音反馈。它不是真正的中文播报内容。

## 语音方案说明

项目采用 **espeak-ng 实时合成** 路线（`nodes/audio.py`）：
告警逻辑产生的中文文案（如“前方有井盖，请绕行”）由 espeak-ng 合成 WAV，
再经 `paplay`（PulseAudio）/ `aplay`（ALSA）播放。**默认不需要任何预录文件**。

## 如需改用预录中文语音（可选）

若希望音质更好、口音更自然，可把提前录好的人声 WAV 放到本目录，
并在 `nodes/logic.py` 的 `speak()` 中优先按 `category` 选择对应文件播放。
建议命名规范：

| 文件名 | 对应提示 |
| --- | --- |
| `blind_path.wav` | 盲道偏移纠正 |
| `upstairs.wav` / `downstairs.wav` | 上下楼梯 |
| `water.wav` / `well_lid.wav` | 水坑 / 井盖 |
| `grass.wav` / `crosswalk.wav` / `wood.wav` | 草地 / 斑马线 / 木板 |

要求：16bit PCM 的 WAV，单声道/双声道均可，paplay 可直接播放。

## 自检

进入容器后执行：`bash sh/audio_check.sh`
