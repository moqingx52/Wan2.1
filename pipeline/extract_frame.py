"""Extract frame index 15 from each test video as I2V conditioning image."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from pipeline.defaults import (
    START_FRAME_INDEX,
    WAN_FRAME_NUM_DEFAULT,
    WAN_SIZE_DEFAULT,
    assert_frame_num_for_pack,
    parse_size_spec,
)
from pipeline.paths import PipelinePaths


def extract_start_frames(
    paths: PipelinePaths,
    case_filter: str | None = None,
    wan_size: str = WAN_SIZE_DEFAULT,
    submit_size: str = "1280*720",
    frame_num: int = WAN_FRAME_NUM_DEFAULT,
) -> Path:
    assert_frame_num_for_pack(frame_num)
    manifest = paths.manifest_path()
    if not manifest.exists():
        raise FileNotFoundError(f"Run `manifest` first: missing {manifest}")

    df = pd.read_csv(manifest)
    if case_filter is not None:
        df = df[df["case"] == case_filter]

    gen_w, gen_h = parse_size_spec(wan_size)
    sub_w, sub_h = parse_size_spec(submit_size)

    start_dir = paths.start_frames_dir()
    meta_dir = paths.work_dir / "meta"
    start_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case = row["case"]
        video_path = Path(row["test_video"])

        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_15 = None
        for i in range(max(total, START_FRAME_INDEX + 1)):
            ok, frame = cap.read()
            if not ok:
                break
            if i == START_FRAME_INDEX:
                frame_15 = frame
                break
        cap.release()

        if frame_15 is None:
            raise RuntimeError(
                f"[{case}] cannot read frame index {START_FRAME_INDEX} from {video_path}"
            )

        frame_15 = cv2.resize(
            frame_15,
            (gen_w, gen_h),
            interpolation=cv2.INTER_AREA,
        )
        out_png = start_dir / f"{case}.png"
        cv2.imwrite(str(out_png), frame_15)

        meta_rows.append(
            {
                "case": case,
                "fps": fps if fps > 0 else 10.0,
                "test_frame_count": total,
                "start_frame": str(out_png.resolve()),
            }
        )

    meta_path = paths.video_meta_path()
    pd.DataFrame(meta_rows).to_csv(meta_path, index=False)

    gen_meta = {
        "wan_size": wan_size,
        "gen_width": gen_w,
        "gen_height": gen_h,
        "submit_width": sub_w,
        "submit_height": sub_h,
        "frame_num": frame_num,
    }
    paths.wan_generate_meta_path().write_text(
        json.dumps(gen_meta, indent=2), encoding="utf-8"
    )

    return meta_path
