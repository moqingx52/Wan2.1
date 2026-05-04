"""Default prompts and numeric constants for the prediction pipeline."""

from __future__ import annotations


def parse_size_spec(spec: str) -> tuple[int, int]:
    """Parse 'W*H' or 'WxH' (e.g. 832*480, 1280x720) into (width, height)."""
    s = (
        spec.strip()
        .replace("×", "*")
        .replace("x", "*")
        .replace("X", "*")
    )
    if "*" not in s:
        raise ValueError(
            f"Invalid size {spec!r}: expected WxH like 832*480 or 1280*720"
        )
    a, b = s.split("*", 1)
    return int(a.strip()), int(b.strip())


# Submission video resolution (README / evaluation).
SUBMIT_WIDTH = 1280
SUBMIT_HEIGHT = 720
OUTPUT_WIDTH = SUBMIT_WIDTH
OUTPUT_HEIGHT = SUBMIT_HEIGHT

# Wan generation: prefer 480P checkpoint + 832*480, then upscale at pack (saves VRAM).
WAN_SIZE_DEFAULT = "832*480"
# 4n+1; 53 frames -> raw[1:51] gives 50 submission frames (same slice as 81-frame path).
WAN_FRAME_NUM_DEFAULT = 53

# Test clip provides 16 frames; use last observability frame as I2V condition (0-based index).
START_FRAME_INDEX = 15

# Lower than Wan default (40 for I2V) to save VRAM/time on 14B multi-GPU runs.
WAN_SAMPLE_STEPS_DEFAULT = 25

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


def assert_frame_num_for_pack(frame_num: int) -> None:
    """Need at least 51 latent frames so raw[1:51] yields 50 submission frames."""
    if frame_num < 51:
        raise ValueError(
            f"frame_num must be >= 51 to take 50 frames after conditioning (got {frame_num})"
        )
    if (frame_num - 1) % 4 != 0:
        raise ValueError(
            f"frame_num must be 4n+1 for Wan (got {frame_num})"
        )
