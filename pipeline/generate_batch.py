"""Invoke Wan official generate.py per case (single- or multi-GPU)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pipeline.defaults import (
    WAN_FRAME_NUM_DEFAULT,
    WAN_SAMPLE_STEPS_DEFAULT,
    WAN_SIZE_DEFAULT,
    assert_frame_num_for_pack,
)
from pipeline.paths import PipelinePaths


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _memory_cli_args(
    offload_model: bool,
    t5_cpu: bool,
    sample_steps: int,
) -> list[str]:
    """Extra flags passed to generate.py (str2bool expects strings from CLI)."""
    parts = [
        "--offload_model",
        "true" if offload_model else "false",
        "--sample_steps",
        str(sample_steps),
    ]
    if t5_cpu:
        parts.append("--t5_cpu")
    return parts


def generate_batch(
    paths: PipelinePaths,
    ckpt_dir: Path,
    case_filter: str | None = None,
    nproc: int = 1,
    base_seed: int = 2026,
    skip_done: bool = True,
    sample_guide_scale: float = 5.0,
    offload_model: bool = True,
    t5_cpu: bool = True,
    sample_steps: int = WAN_SAMPLE_STEPS_DEFAULT,
    t5_fsdp: bool = False,
    wan_size: str = WAN_SIZE_DEFAULT,
    frame_num: int = WAN_FRAME_NUM_DEFAULT,
) -> None:
    assert_frame_num_for_pack(frame_num)

    manifest = paths.manifest_path()
    if not manifest.exists():
        raise FileNotFoundError(f"Run `manifest` first: missing {manifest}")

    df = pd.read_csv(manifest)
    if case_filter is not None:
        df = df[df["case"] == case_filter]

    wan_repo = paths.wan_repo()
    generate_py = wan_repo / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(generate_py)

    prompts_dir = paths.prompts_dir()
    start_dir = paths.start_frames_dir()

    mem_suffix = _memory_cli_args(offload_model, t5_cpu, sample_steps)

    for run_idx, (_, row) in enumerate(df.iterrows()):
        case = row["case"]
        case_raw = paths.raw_dir(case)
        case_raw.mkdir(parents=True, exist_ok=True)
        save_file = case_raw / "raw.mp4"
        done_flag = case_raw / "DONE"

        if skip_done and done_flag.exists():
            print(f"[SKIP] {case}")
            continue

        start_img = start_dir / f"{case}.png"
        prompt_file = prompts_dir / f"{case}.prompt.txt"
        if not start_img.exists():
            raise FileNotFoundError(start_img)
        if not prompt_file.exists():
            raise FileNotFoundError(prompt_file)

        prompt = _read_text(prompt_file)
        neg_file = prompts_dir / f"{case}.negative.txt"
        neg_args: list[str] = []
        if neg_file.is_file():
            neg_args = ["--sample_neg_prompt", _read_text(neg_file)]

        log_path = case_raw / "generate.log"

        cmd: list[str]
        if nproc <= 1:
            cmd = [
                sys.executable,
                str(generate_py),
                "--task",
                "i2v-14B",
                "--size",
                wan_size,
                "--ckpt_dir",
                str(ckpt_dir),
                "--image",
                str(start_img),
                "--prompt",
                prompt,
                "--frame_num",
                str(frame_num),
                "--save_file",
                str(save_file),
                "--base_seed",
                str(base_seed + run_idx),
                "--sample_guide_scale",
                str(sample_guide_scale),
            ]
            cmd.extend(mem_suffix)
            cmd.extend(neg_args)
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={nproc}",
                str(generate_py),
                "--task",
                "i2v-14B",
                "--size",
                wan_size,
                "--ckpt_dir",
                str(ckpt_dir),
                "--image",
                str(start_img),
                "--prompt",
                prompt,
                "--frame_num",
                str(frame_num),
                "--save_file",
                str(save_file),
                "--dit_fsdp",
            ]
            if t5_fsdp:
                cmd.append("--t5_fsdp")
            cmd.extend(
                [
                    "--ulysses_size",
                    str(nproc),
                    "--base_seed",
                    str(base_seed + run_idx),
                    "--sample_guide_scale",
                    str(sample_guide_scale),
                ]
            )
            cmd.extend(mem_suffix)
            cmd.extend(neg_args)

        print("[RUN]", case)
        print(shlex.join(cmd))

        env = os.environ.copy()
        with log_path.open("w", encoding="utf-8") as log_f:
            ret = subprocess.run(cmd, cwd=str(wan_repo), env=env, stdout=log_f, stderr=subprocess.STDOUT)

        if ret.returncode != 0:
            print(f"[ERROR] {case} (log: {log_path})")
            continue

        if not save_file.exists():
            # generate.py may still name output differently in edge cases; try any mp4
            mp4s = list(case_raw.glob("*.mp4"))
            if not mp4s:
                print(f"[ERROR] {case}: no output mp4 in {case_raw}")
                continue
            latest = max(mp4s, key=lambda p: p.stat().st_mtime)
            if latest != save_file:
                latest.replace(save_file)

        done_flag.write_text("ok", encoding="utf-8")
        print(f"[DONE] {case}")
