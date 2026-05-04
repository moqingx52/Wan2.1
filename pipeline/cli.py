"""CLI for Wan2.1 prediction video pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.extract_frame import extract_start_frames
from pipeline.generate_batch import generate_batch
from pipeline.manifest import build_manifest
from pipeline.pack import pack_submission
from pipeline.paths import PipelinePaths
from pipeline.prompts import build_prompts
from pipeline.validate import validate_submission


def _paths(ns: argparse.Namespace) -> PipelinePaths:
    return PipelinePaths(
        dataset_root=Path(ns.dataset_root).resolve(),
        rdt_root=Path(ns.rdt_root).resolve(),
        work_dir=Path(ns.work_dir).resolve(),
        out_dir=Path(ns.out_dir).resolve(),
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Directory containing test/ (and typically train/).",
    )
    p.add_argument(
        "--rdt-root",
        type=str,
        required=True,
        help="Root folder with per-case action.txt / joint.txt (e.g. sample_result_rdt/).",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Scratch outputs: manifest, start_frames, prompts, raw/.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Final submission tree (per-case subfolders).",
    )
    p.add_argument(
        "--case",
        type=str,
        default=None,
        help="If set, only process this case id (e.g. 1_1).",
    )


def cmd_manifest(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = build_manifest(paths, case_filter=args.case)
    print(out)


def cmd_extract(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = extract_start_frames(paths, case_filter=args.case)
    print(out)


def cmd_prompts(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = build_prompts(paths, case_filter=args.case)
    print(out)


def cmd_generate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    generate_batch(
        paths,
        ckpt_dir=Path(args.ckpt_dir).resolve(),
        case_filter=args.case,
        nproc=args.nproc,
        base_seed=args.base_seed,
        skip_done=not args.no_skip_done,
        sample_guide_scale=args.sample_guide_scale,
    )


def cmd_pack(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = pack_submission(paths, case_filter=args.case)
    print(out)


def cmd_validate(args: argparse.Namespace) -> None:
    bad = validate_submission(Path(args.out_dir).resolve())
    if bad:
        for case, field, msg in bad:
            print(f"[FAIL] {case or '.'} {field}: {msg}")
        raise SystemExit(1)
    print("OK")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_manifest(args)
    cmd_extract(args)
    cmd_prompts(args)
    cmd_generate(args)
    cmd_pack(args)
    cmd_validate(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wan2.1 prediction video pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_man = sub.add_parser("manifest", help="Create work-dir/manifest.csv")
    _add_common(p_man)
    p_man.set_defaults(func=cmd_manifest)

    p_ext = sub.add_parser("extract", help="Extract frame 16 PNGs + meta/video_meta.csv")
    _add_common(p_ext)
    p_ext.set_defaults(func=cmd_extract)

    p_pr = sub.add_parser("prompts", help="Write prompts/*.txt from instructions")
    _add_common(p_pr)
    p_pr.set_defaults(func=cmd_prompts)

    p_gen = sub.add_parser("generate", help="Run Wan generate.py per case")
    _add_common(p_gen)
    p_gen.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
        help="Wan2.1-I2V-14B-720P checkpoint directory.",
    )
    p_gen.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="torchrun --nproc_per_node (1 = plain python, no FSDP).",
    )
    p_gen.add_argument("--base-seed", type=int, default=2026)
    p_gen.add_argument(
        "--no-skip-done",
        action="store_true",
        help="Re-run even if raw/<case>/DONE exists.",
    )
    p_gen.add_argument("--sample-guide-scale", type=float, default=5.0)
    p_gen.set_defaults(func=cmd_generate)

    p_pack = sub.add_parser("pack", help="Slice 50 frames + RDT CSVs into out-dir")
    _add_common(p_pack)
    p_pack.set_defaults(func=cmd_pack)

    p_val = sub.add_parser("validate", help="Check out-dir submission format")
    p_val.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Submission root to validate.",
    )
    p_val.set_defaults(func=cmd_validate)

    p_all = sub.add_parser("all", help="manifest → extract → prompts → generate → pack → validate")
    _add_common(p_all)
    p_all.add_argument("--ckpt-dir", type=str, required=True)
    p_all.add_argument("--nproc", type=int, default=1)
    p_all.add_argument("--base-seed", type=int, default=2026)
    p_all.add_argument("--no-skip-done", action="store_true")
    p_all.add_argument("--sample-guide-scale", type=float, default=5.0)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
