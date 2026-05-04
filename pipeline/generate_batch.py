"""Generate-phase launchers (legacy per-case and resident worker)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.defaults import (
    WAN_FRAME_NUM_DEFAULT,
    WAN_SAMPLE_STEPS_DEFAULT,
    WAN_SIZE_DEFAULT,
    assert_frame_num_for_pack,
)
from pipeline.paths import PipelinePaths


@dataclass(frozen=True)
class CaseInputs:
    case: str
    run_idx: int
    start_img: Path
    prompt: str
    neg_prompt: str | None
    raw_dir: Path
    save_file: Path
    done_flag: Path
    log_path: Path


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


def iter_case_inputs(
    paths: PipelinePaths,
    case_filter: str | None = None,
) -> list[CaseInputs]:
    manifest = paths.manifest_path()
    if not manifest.exists():
        raise FileNotFoundError(f"Run `manifest` first: missing {manifest}")

    df = pd.read_csv(manifest)
    if case_filter is not None:
        df = df[df["case"] == case_filter]

    prompts_dir = paths.prompts_dir()
    start_dir = paths.start_frames_dir()

    out: list[CaseInputs] = []
    for run_idx, (_, row) in enumerate(df.iterrows()):
        case = row["case"]
        raw_dir = paths.raw_dir(case)
        raw_dir.mkdir(parents=True, exist_ok=True)

        start_img = start_dir / f"{case}.png"
        prompt_file = prompts_dir / f"{case}.prompt.txt"
        if not start_img.exists():
            raise FileNotFoundError(start_img)
        if not prompt_file.exists():
            raise FileNotFoundError(prompt_file)

        neg_file = prompts_dir / f"{case}.negative.txt"
        out.append(
            CaseInputs(
                case=case,
                run_idx=run_idx,
                start_img=start_img,
                prompt=_read_text(prompt_file),
                neg_prompt=_read_text(neg_file) if neg_file.is_file() else None,
                raw_dir=raw_dir,
                save_file=raw_dir / "raw.mp4",
                done_flag=raw_dir / "DONE",
                log_path=raw_dir / "generate.log",
            )
        )
    return out


def launch_generate_worker(
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
    """Launch resident worker once; model loads once per rank."""
    assert_frame_num_for_pack(frame_num)
    # Validate required inputs early before spawning distributed workers.
    _ = iter_case_inputs(paths, case_filter=case_filter)

    worker_mod = "pipeline.generate_worker"
    base_cmd = [
        "--dataset-root",
        str(paths.dataset_root),
        "--rdt-root",
        str(paths.rdt_root),
        "--work-dir",
        str(paths.work_dir),
        "--out-dir",
        str(paths.out_dir),
        "--ckpt-dir",
        str(ckpt_dir),
        "--base-seed",
        str(base_seed),
        "--sample-guide-scale",
        str(sample_guide_scale),
        "--sample-steps",
        str(sample_steps),
        "--wan-size",
        wan_size,
        "--frame-num",
        str(frame_num),
    ]
    if case_filter is not None:
        base_cmd.extend(["--case", case_filter])
    if skip_done:
        base_cmd.append("--skip-done")
    if offload_model:
        base_cmd.append("--offload-model")
    if t5_cpu:
        base_cmd.append("--t5-cpu")
    if t5_fsdp:
        base_cmd.append("--t5-fsdp")

    if nproc <= 1:
        cmd = [sys.executable, "-m", worker_mod, *base_cmd]
    else:
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "-m",
            worker_mod,
            "--ulysses-size",
            str(nproc),
            *base_cmd,
        ]

    print("[RUN] resident worker")
    print(shlex.join(cmd))
    ret = subprocess.run(cmd, cwd=str(paths.wan_repo()), env=os.environ.copy())
    if ret.returncode != 0:
        raise RuntimeError(f"Resident generate worker failed with exit code {ret.returncode}")


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
    """Legacy mode: one process spawn per case (kept as fallback)."""
    assert_frame_num_for_pack(frame_num)
    cases = iter_case_inputs(paths, case_filter=case_filter)

    wan_repo = paths.wan_repo()
    generate_py = wan_repo / "generate.py"
    if not generate_py.exists():
        raise FileNotFoundError(generate_py)

    mem_suffix = _memory_cli_args(offload_model, t5_cpu, sample_steps)

    for c in cases:
        if skip_done and c.done_flag.exists():
            print(f"[SKIP] {c.case}")
            continue
        case = c.case
        neg_args: list[str] = (
            ["--sample_neg_prompt", c.neg_prompt] if c.neg_prompt else []
        )

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
                str(c.start_img),
                "--prompt",
                c.prompt,
                "--frame_num",
                str(frame_num),
                "--save_file",
                str(c.save_file),
                "--base_seed",
                str(base_seed + c.run_idx),
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
                str(c.start_img),
                "--prompt",
                c.prompt,
                "--frame_num",
                str(frame_num),
                "--save_file",
                str(c.save_file),
                "--dit_fsdp",
            ]
            if t5_fsdp:
                cmd.append("--t5_fsdp")
            cmd.extend(
                [
                    "--ulysses_size",
                    str(nproc),
                    "--base_seed",
                    str(base_seed + c.run_idx),
                    "--sample_guide_scale",
                    str(sample_guide_scale),
                ]
            )
            cmd.extend(mem_suffix)
            cmd.extend(neg_args)

        print("[RUN]", case)
        print(shlex.join(cmd))

        env = os.environ.copy()
        with c.log_path.open("w", encoding="utf-8") as log_f:
            ret = subprocess.run(cmd, cwd=str(wan_repo), env=env, stdout=log_f, stderr=subprocess.STDOUT)

        if ret.returncode != 0:
            print(f"[ERROR] {case} (log: {c.log_path})")
            continue

        if not c.save_file.exists():
            # generate.py may still name output differently in edge cases; try any mp4
            mp4s = list(c.raw_dir.glob("*.mp4"))
            if not mp4s:
                print(f"[ERROR] {case}: no output mp4 in {c.raw_dir}")
                continue
            latest = max(mp4s, key=lambda p: p.stat().st_mtime)
            if latest != c.save_file:
                latest.replace(c.save_file)

        c.done_flag.write_text("ok", encoding="utf-8")
        print(f"[DONE] {case}")
