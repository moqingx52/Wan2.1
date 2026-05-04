"""Build manifest.csv listing test cases and RDT outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.paths import PipelinePaths, resolve_instruction_file


def build_manifest(
    paths: PipelinePaths,
    case_filter: str | None = None,
    allow_missing_rdt: bool = False,
) -> Path:
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    test_root = paths.test_root
    if not test_root.is_dir():
        raise FileNotFoundError(f"Missing test directory: {test_root}")

    for case_dir in sorted(test_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        if case_filter is not None and case != case_filter:
            continue

        video_path = case_dir / "video.mp4"
        if not video_path.exists():
            continue

        try:
            instr_path = resolve_instruction_file(case_dir)
        except FileNotFoundError:
            continue

        rdt_case = paths.rdt_root / case
        rdt_action = rdt_case / "action.txt"
        rdt_joint = rdt_case / "joint.txt"

        if not rdt_action.is_file() or not rdt_joint.is_file():
            msg = (
                f"[{case}] missing RDT files under {rdt_case}: "
                f"action.txt exists={rdt_action.is_file()}, "
                f"joint.txt exists={rdt_joint.is_file()}"
            )
            if allow_missing_rdt:
                print("WARNING:", msg)
            else:
                raise FileNotFoundError(msg)

        rows.append(
            {
                "case": case,
                "test_video": str(video_path.resolve()),
                "instruction_path": str(instr_path.resolve()),
                "rdt_action": str(rdt_action.resolve()),
                "rdt_joint": str(rdt_joint.resolve()),
                "out_case_dir": str((paths.out_dir / case).resolve()),
            }
        )

    if not rows:
        raise RuntimeError(
            "No cases found. Check --dataset-root / --rdt-root / --case filter."
        )

    df = pd.DataFrame(rows)
    out = paths.manifest_path()
    df.to_csv(out, index=False)
    return out
