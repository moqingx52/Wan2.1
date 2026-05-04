"""Default prompts and numeric constants for the prediction pipeline."""

from __future__ import annotations

# Output geometry for submission video / start frame resize.
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# Test clip provides 16 frames; use last observability frame as I2V condition (0-based index).
START_FRAME_INDEX = 15

# Wan I2V uses 4n+1 frame counts; 81 -> take frames[1:51] => 50 submission frames.
WAN_FRAME_NUM = 81

# Lower than Wan default (40 for I2V) to save VRAM/time on 14B 720P multi-GPU runs.
WAN_SAMPLE_STEPS_DEFAULT = 30

# README: 51 predicted action/joint rows after the 16-frame prefix.
TARGET_ACTION_ROWS = 51

# First row index for prediction segment when test ends at frame index 15 (README convention).
PRED_INDEX_START = 16

DEFAULT_NEGATIVE_PROMPT = (
    "模糊，闪烁，抖动，变形，机械臂断裂，夹爪畸形，物体突然消失，"
    "物体突然出现，背景变化，镜头运动，视角切换，字幕，水印，文字，"
    "卡通，低清晰度，过曝，欠曝，多余机械臂，多余手指，多余物体"
)


def wrap_instruction_prompt(instruction: str) -> str:
    instruction = (instruction or "").strip()
    return (
        "固定单目相机视角，真实机器人操作场景，保持桌面、背景、光照、"
        "机械臂外观、夹爪外观和目标物体外观与输入帧一致。"
        f"任务指令：{instruction}。"
        "机器人机械臂应根据任务指令继续执行自然、平滑、物理合理的操作，"
        "运动幅度适中，物体交互过程连续，不能改变场景布局，不能新增物体，"
        "不能切换镜头，不能出现文字、水印或字幕。"
    )
