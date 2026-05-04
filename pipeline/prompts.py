"""Write per-case positive and negative prompt files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pipeline.defaults import DEFAULT_NEGATIVE_PROMPT, wrap_instruction_prompt
from pipeline.paths import PipelinePaths


def build_prompts(paths: PipelinePaths, case_filter: str | None = None) -> Path:
    manifest = paths.manifest_path()
    if not manifest.exists():
        raise FileNotFoundError(f"Run `manifest` first: missing {manifest}")

    df = pd.read_csv(manifest)
    if case_filter is not None:
        df = df[df["case"] == case_filter]

    out_dir = paths.prompts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        case = row["case"]
        instr_path = Path(row["instruction_path"])
        text = instr_path.read_text(encoding="utf-8", errors="ignore").strip()
        prompt = wrap_instruction_prompt(text)

        (out_dir / f"{case}.prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / f"{case}.negative.txt").write_text(
            DEFAULT_NEGATIVE_PROMPT, encoding="utf-8"
        )

    return out_dir
