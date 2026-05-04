"""Turn raw Wan MP4 + RDT CSVs into submission folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from pipeline.defaults import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PRED_INDEX_START,
    TARGET_ACTION_ROWS,
    parse_size_spec,
)
from pipeline.paths import PipelinePaths


def _read_video_frames(path: Path) -> tuple[list, float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _resolve_submit_dims(
    paths: PipelinePaths, submit_size: str | None
) -> tuple[int, int]:
    if submit_size:
        return parse_size_spec(submit_size)
    meta_path = paths.wan_generate_meta_path()
    if meta_path.is_file():
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(data["submit_width"]), int(data["submit_height"])
    return OUTPUT_WIDTH, OUTPUT_HEIGHT


def _write_video_mp4(
    frames: list,
    path: Path,
    fps: float,
    out_w: int,
    out_h: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        fps if fps > 0 else 10.0,
        (out_w, out_h),
    )
    for f in frames:
        if f.shape[1] != out_w or f.shape[0] != out_h:
            f = cv2.resize(
                f,
                (out_w, out_h),
                interpolation=cv2.INTER_CUBIC,
            )
        writer.write(f)
    writer.release()


def fix_index_csv(src: Path, dst: Path, start_index: int, target_len: int) -> None:
    df = pd.read_csv(src)
    if len(df) > target_len:
        df = df.iloc[-target_len:].copy()
    elif len(df) < target_len:
        last = df.iloc[-1:].copy()
        while len(df) < target_len:
            df = pd.concat([df, last], ignore_index=True)
    df = df.iloc[:target_len].copy()
    if "Unnamed: 0" in df.columns:
        df["Unnamed: 0"] = list(range(start_index, start_index + target_len))
    df.to_csv(dst, index=False)


def pack_submission(
    paths: PipelinePaths,
    case_filter: str | None = None,
    submit_size: str | None = None,
) -> Path:
    manifest = paths.manifest_path()
    meta_path = paths.video_meta_path()
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Run `extract` first for FPS meta: missing {meta_path}"
        )

    submit_w, submit_h = _resolve_submit_dims(paths, submit_size)

    df = pd.read_csv(manifest)
    meta = pd.read_csv(meta_path)
    meta_by_case = meta.set_index("case")["fps"].to_dict()

    if case_filter is not None:
        df = df[df["case"] == case_filter]

    out_root = paths.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case = row["case"]
        raw_mp4 = paths.raw_dir(case) / "raw.mp4"
        if not raw_mp4.exists():
            mp4s = sorted((paths.raw_dir(case)).glob("*.mp4"))
            if not mp4s:
                raise FileNotFoundError(f"No raw mp4 for case {case}")
            raw_mp4 = max(mp4s, key=lambda p: p.stat().st_mtime)

        frames, file_fps = _read_video_frames(raw_mp4)
        if len(frames) < 2:
            raise RuntimeError(f"{case}: raw video has fewer than 2 frames: {raw_mp4}")
        fps = float(meta_by_case.get(case, file_fps))
        if fps <= 0:
            fps = 10.0

        future_frames = frames[1 : 1 + 50]
        if len(future_frames) < 50:
            while len(future_frames) < 50:
                future_frames.append(future_frames[-1].copy() if future_frames else frames[0])

        out_dir = out_root / case
        out_dir.mkdir(parents=True, exist_ok=True)

        _write_video_mp4(
            future_frames[:50],
            out_dir / "video.mp4",
            fps=fps,
            out_w=submit_w,
            out_h=submit_h,
        )

        instr_src = Path(row["instruction_path"])
        shutil.copyfile(instr_src, out_dir / "instructions.txt")

        rdt_action = Path(row["rdt_action"])
        rdt_joint = Path(row["rdt_joint"])
        if not rdt_action.exists():
            raise FileNotFoundError(rdt_action)
        if not rdt_joint.exists():
            raise FileNotFoundError(rdt_joint)

        fix_index_csv(
            rdt_action,
            out_dir / "action.txt",
            PRED_INDEX_START,
            TARGET_ACTION_ROWS,
        )
        fix_index_csv(
            rdt_joint,
            out_dir / "joint.txt",
            PRED_INDEX_START,
            TARGET_ACTION_ROWS,
        )

    return out_root
