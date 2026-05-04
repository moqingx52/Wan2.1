"""Resident generation worker: load model once, loop all cases."""

from __future__ import annotations

import argparse
import logging
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video

from pipeline.defaults import WAN_FRAME_NUM_DEFAULT, WAN_SAMPLE_STEPS_DEFAULT, WAN_SIZE_DEFAULT
from pipeline.generate_batch import iter_case_inputs
from pipeline.paths import PipelinePaths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resident Wan I2V worker")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--rdt-root", type=str, required=True)
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--case", type=str, default=None)
    p.add_argument("--wan-size", type=str, default=WAN_SIZE_DEFAULT)
    p.add_argument("--frame-num", type=int, default=WAN_FRAME_NUM_DEFAULT)
    p.add_argument("--base-seed", type=int, default=2026)
    p.add_argument("--sample-guide-scale", type=float, default=5.0)
    p.add_argument("--sample-steps", type=int, default=WAN_SAMPLE_STEPS_DEFAULT)
    p.add_argument("--sample-shift", type=float, default=None)
    p.add_argument("--skip-done", action="store_true", default=False)
    p.add_argument("--offload-model", action="store_true", default=False)
    p.add_argument("--t5-cpu", action="store_true", default=False)
    p.add_argument("--t5-fsdp", action="store_true", default=False)
    p.add_argument("--ulysses-size", type=int, default=1)
    p.add_argument("--ring-size", type=int, default=1)
    return p.parse_args()


def _setup_logging(rank: int) -> None:
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.ERROR, format="[%(asctime)s] %(levelname)s: %(message)s")


def _setup_dist(args: argparse.Namespace) -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    _setup_logging(rank)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return rank, world_size, local_rank


def _setup_ulysses_if_needed(args: argparse.Namespace, world_size: int) -> None:
    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size
        from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )


def _default_sample_shift(size: str) -> float:
    # Match generate.py behavior for i2v.
    return 3.0 if size in ("832*480", "480*832") else 5.0


def main() -> None:
    args = _parse_args()
    rank, world_size, local_rank = _setup_dist(args)
    _setup_ulysses_if_needed(args, world_size)
    if rank == 0:
        logging.info("Distributed world_size=%s (all ranks run each case together).", world_size)

    paths = PipelinePaths(
        dataset_root=Path(args.dataset_root).resolve(),
        rdt_root=Path(args.rdt_root).resolve(),
        work_dir=Path(args.work_dir).resolve(),
        out_dir=Path(args.out_dir).resolve(),
    )
    cases = iter_case_inputs(paths, case_filter=args.case)

    cfg = WAN_CONFIGS["i2v-14B"]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0

    if dist.is_initialized():
        seed_list = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_list, src=0)
        args.base_seed = int(seed_list[0])

    logging.info("Loading WanI2V once per rank...")
    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=(world_size > 1),
        use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
        t5_cpu=args.t5_cpu,
    )
    logging.info("Model loaded. Begin resident inference loop.")

    shift = args.sample_shift if args.sample_shift is not None else _default_sample_shift(args.wan_size)
    for c in cases:
        # FSDP / USP: every rank must enter the same generate() collectives per case.
        if args.skip_done and c.done_flag.exists():
            if rank == 0:
                logging.info(f"[SKIP] {c.case}")
            if dist.is_initialized():
                dist.barrier()
            continue

        try:
            c.raw_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(c.start_img).convert("RGB")
            n_prompt = c.neg_prompt or ""
            video = wan_i2v.generate(
                c.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.wan_size],
                frame_num=args.frame_num,
                shift=shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                n_prompt=n_prompt,
                seed=args.base_seed + c.run_idx,
                offload_model=args.offload_model,
            )
            if rank == 0:
                save_ok = cache_video(
                    tensor=video[None],
                    save_file=str(c.save_file),
                    fps=cfg.sample_fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1),
                )
                if save_ok is None:
                    raise RuntimeError("cache_video returned None")
                c.done_flag.write_text("ok", encoding="utf-8")
                logging.info(f"[DONE] {c.case}")
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            err = traceback.format_exc()
            if rank == 0:
                c.log_path.write_text(err, encoding="utf-8")
                logging.error(f"[ERROR] {c.case}\n{err}")
            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
    logging.info(f"Resident worker finished on rank {rank}.")


if __name__ == "__main__":
    main()
