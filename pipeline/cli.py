"""CLI for Wan2.1 prediction video pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.extract_frame import extract_start_frames
from pipeline.generate_batch import generate_batch, launch_generate_worker
from pipeline.manifest import build_manifest
from pipeline.pack import pack_submission
from pipeline.paths import PipelinePaths
from pipeline.prompts import build_prompts
from pipeline.defaults import (
    WAN_FRAME_NUM_DEFAULT,
    WAN_SAMPLE_STEPS_DEFAULT,
    WAN_SIZE_DEFAULT,
)
from pipeline.validate import validate_submission


def _add_wan_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wan-size",
        type=str,
        default=WAN_SIZE_DEFAULT,
        metavar="WxH",
        help=(
            "Wan generate.py --size (e.g. 832*480 with Wan2.1-I2V-14B-480P; "
            "1280*720 with 720P ckpt). Start frames are resized to match."
        ),
    )
    p.add_argument(
        "--submit-size",
        type=str,
        default="1280*720",
        metavar="WxH",
        help="Pack output video resolution (upscale raw with INTER_CUBIC).",
    )
    p.add_argument(
        "--frame-num",
        type=int,
        default=WAN_FRAME_NUM_DEFAULT,
        metavar="N",
        help=(
            "Wan --frame_num (must be 4n+1 and >=51). Default 53 for lighter 480P runs; "
            "use 81 with 720P."
        ),
    )


def _add_generate_memory_args(p: argparse.ArgumentParser) -> None:
    """Defaults tuned for 14B I2V on multi-GPU (offload + T5 CPU + fewer steps).

    By default we do not pass --t5_fsdp when using torchrun: with --t5_cpu it is often
    redundant and can add overhead/OOM risk; pass --t5-fsdp to enable it if needed.
    """
    p.set_defaults(offload_model=True, t5_cpu=True, t5_fsdp=False)
    p.add_argument(
        "--no-offload-model",
        dest="offload_model",
        action="store_false",
        help="Do not pass --offload_model true (Wan multi-GPU default is false).",
    )
    p.add_argument(
        "--no-t5-cpu",
        dest="t5_cpu",
        action="store_false",
        help="Omit --t5_cpu (keep T5 on GPU; higher VRAM).",
    )
    p.add_argument(
        "--t5-fsdp",
        dest="t5_fsdp",
        action="store_true",
        help="Add --t5_fsdp with torchrun (off by default; use if stable and not OOM).",
    )
    p.add_argument(
        "--no-t5-fsdp",
        dest="t5_fsdp",
        action="store_false",
        help="Same as default now; kept for scripts that already pass this flag.",
    )
    p.add_argument(
        "--sample-steps",
        type=int,
        default=WAN_SAMPLE_STEPS_DEFAULT,
        metavar="N",
        help=(
            f"I2V sampling steps (default {WAN_SAMPLE_STEPS_DEFAULT}; "
            "Wan I2V default is 40)."
        ),
    )


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
    out = build_manifest(
        paths,
        case_filter=args.case,
        allow_missing_rdt=args.allow_missing_rdt,
    )
    print(out)


def cmd_extract(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = extract_start_frames(
        paths,
        case_filter=args.case,
        wan_size=args.wan_size,
        submit_size=args.submit_size,
        frame_num=args.frame_num,
    )
    print(out)


def cmd_prompts(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = build_prompts(paths, case_filter=args.case)
    print(out)


def cmd_generate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    runner = generate_batch if args.legacy_per_case else launch_generate_worker
    runner(
        paths,
        ckpt_dir=Path(args.ckpt_dir).resolve(),
        case_filter=args.case,
        nproc=args.nproc,
        base_seed=args.base_seed,
        skip_done=not args.no_skip_done,
        sample_guide_scale=args.sample_guide_scale,
        offload_model=args.offload_model,
        t5_cpu=args.t5_cpu,
        sample_steps=args.sample_steps,
        t5_fsdp=args.t5_fsdp,
        wan_size=args.wan_size,
        frame_num=args.frame_num,
    )


def cmd_pack(args: argparse.Namespace) -> None:
    paths = _paths(args)
    out = pack_submission(
        paths,
        case_filter=args.case,
        submit_size=args.submit_size,
    )
    print(out)


def cmd_validate(args: argparse.Namespace) -> None:
    expected = None if args.expected_cases == 0 else args.expected_cases
    bad = validate_submission(
        Path(args.out_dir).resolve(),
        expected_case_count=expected,
    )
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
    p_man.add_argument(
        "--allow-missing-rdt",
        action="store_true",
        help="Warn only if RDT action.txt/joint.txt are missing (default: fail).",
    )
    p_man.set_defaults(func=cmd_manifest)

    p_ext = sub.add_parser("extract", help="Extract frame 16 PNGs + meta/video_meta.csv")
    _add_common(p_ext)
    _add_wan_shape_args(p_ext)
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
        help="Checkpoint dir (e.g. Wan2.1-I2V-14B-480P for --wan-size 832*480, or 720P ckpt).",
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
    p_gen.add_argument(
        "--legacy-per-case",
        action="store_true",
        help="Fallback to old behavior: spawn one process per case.",
    )
    _add_wan_shape_args(p_gen)
    _add_generate_memory_args(p_gen)
    p_gen.set_defaults(func=cmd_generate)

    p_pack = sub.add_parser("pack", help="Slice 50 frames + RDT CSVs into out-dir")
    _add_common(p_pack)
    p_pack.add_argument(
        "--submit-size",
        type=str,
        default=None,
        metavar="WxH",
        help=(
            "Output video WxH (default: read meta/wan_generate.json from extract, "
            "else 1280*720)."
        ),
    )
    p_pack.set_defaults(func=cmd_pack)

    p_val = sub.add_parser("validate", help="Check out-dir submission format")
    p_val.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Submission root to validate.",
    )
    p_val.add_argument(
        "--expected-cases",
        type=int,
        default=100,
        metavar="N",
        help="Require exactly N case subdirs (default 100). Use 0 to skip.",
    )
    p_val.set_defaults(func=cmd_validate)

    p_all = sub.add_parser("all", help="manifest → extract → prompts → generate → pack → validate")
    _add_common(p_all)
    p_all.add_argument("--ckpt-dir", type=str, required=True)
    p_all.add_argument("--nproc", type=int, default=1)
    p_all.add_argument("--base-seed", type=int, default=2026)
    p_all.add_argument("--no-skip-done", action="store_true")
    p_all.add_argument("--sample-guide-scale", type=float, default=5.0)
    p_all.add_argument(
        "--legacy-per-case",
        action="store_true",
        help="Fallback to old per-case spawn path for generate stage.",
    )
    _add_wan_shape_args(p_all)
    _add_generate_memory_args(p_all)
    p_all.add_argument(
        "--allow-missing-rdt",
        action="store_true",
        help="Passed through to manifest.",
    )
    p_all.add_argument(
        "--expected-cases",
        type=int,
        default=100,
        metavar="N",
        help="Passed through to validate (0 = skip case count).",
    )
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
