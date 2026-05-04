"""Validate submission directory layout and shapes."""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

from pipeline.defaults import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PRED_INDEX_START,
    TARGET_ACTION_ROWS,
)


REQUIRED = ("video.mp4", "action.txt", "joint.txt", "instructions.txt")


def validate_submission(
    out_dir: Path,
    expected_width: int = OUTPUT_WIDTH,
    expected_height: int = OUTPUT_HEIGHT,
    expected_video_frames: int = 50,
    expected_csv_rows: int = TARGET_ACTION_ROWS,
) -> list[tuple[str, str, str]]:
    """Return list of (case, field, message) for problems; empty if OK."""
    bad: list[tuple[str, str, str]] = []

    if not out_dir.is_dir():
        return [("", "", f"not a directory: {out_dir}")]

    case_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())

    for d in case_dirs:
        name = d.name
        for fname in REQUIRED:
            if not (d / fname).exists():
                bad.append((name, fname, "missing"))

        video = d / "video.mp4"
        if video.exists():
            cap = cv2.VideoCapture(str(video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if n != expected_video_frames:
                bad.append((name, "video.mp4", f"frame_count={n}, expected={expected_video_frames}"))
            if w != expected_width or h != expected_height:
                bad.append(
                    (
                        name,
                        "video.mp4",
                        f"resolution={w}x{h}, expected={expected_width}x{expected_height}",
                    )
                )

        for csv_name in ("action.txt", "joint.txt"):
            p = d / csv_name
            if not p.exists():
                continue
            df = pd.read_csv(p)
            nrows = len(df)
            if nrows != expected_csv_rows:
                bad.append(
                    (name, csv_name, f"rows={nrows}, expected={expected_csv_rows}")
                )
            if "Unnamed: 0" in df.columns:
                seq = [int(x) for x in df["Unnamed: 0"].tolist()]
                expected_seq = list(
                    range(PRED_INDEX_START, PRED_INDEX_START + expected_csv_rows)
                )
                if seq != expected_seq:
                    bad.append(
                        (
                            name,
                            csv_name,
                            f"Unnamed: 0 sequence mismatch (got start={seq[:3]}...)",
                        )
                    )

    return bad
