"""Path helpers for dataset / work / output roots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    dataset_root: Path
    rdt_root: Path
    work_dir: Path
    out_dir: Path

    @property
    def test_root(self) -> Path:
        return self.dataset_root / "test"

    def manifest_path(self) -> Path:
        return self.work_dir / "manifest.csv"

    def video_meta_path(self) -> Path:
        return self.work_dir / "meta" / "video_meta.csv"

    def start_frames_dir(self) -> Path:
        return self.work_dir / "start_frames"

    def prompts_dir(self) -> Path:
        return self.work_dir / "prompts"

    def raw_dir(self, case: str) -> Path:
        return self.work_dir / "raw" / case

    def wan_repo(self) -> Path:
        """Directory containing generate.py (parent of pipeline/)."""
        return Path(__file__).resolve().parent.parent


def resolve_instruction_file(case_dir: Path) -> Path:
    """Prefer instructions.txt; fallback to instruction.txt (sample naming)."""
    for name in ("instructions.txt", "instruction.txt"):
        p = case_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No instructions.txt or instruction.txt under {case_dir}"
    )
