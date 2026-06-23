from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm.workflow_config import DEFAULT_INPUT_PATH, DEFAULT_TEMPLATE_PATH
from llm.workflow_io import (
    build_target_mask,
    load_input_frame,
    load_template,
    render_prompt,
    resolve_job_dir,
    resolve_output_path,
    save_output,
)
from llm.workflow_steps import run_batch_follow_up, run_batch_submit, run_sync


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Use the OpenAI API to reclassify ETF rows that still contain 'Other'.")
    parser.add_argument("--mode", choices=["auto", "sync", "batch-submit", "batch-status", "batch-apply"], default="auto")
    parser.add_argument("--batch-threshold", type=int, default=50)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--batch-id", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--completion-window", type=str, default="24h")
    return parser.parse_args()


def main() -> None:
    """Run the requested classification workflow."""
    args = parse_args()
    summary = run_batch_follow_up(args) if args.mode in {"batch-status", "batch-apply"} else run_primary_flow(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_primary_flow(args: argparse.Namespace) -> dict[str, Any]:
    """Run auto, sync, or batch-submit mode."""
    input_path = DEFAULT_INPUT_PATH.resolve()
    output_path = resolve_output_path(input_path, None)
    df = load_input_frame(input_path)
    template = load_template(DEFAULT_TEMPLATE_PATH.resolve())
    target_rows = df.loc[build_target_mask(df)].copy()
    target_count = len(target_rows)

    if target_rows.empty:
        save_output(df, output_path)
        return {
            "mode": "noop",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "target_rows": 0,
            "message": "No rows contained 'Other' in the target fields.",
        }

    mode = "sync" if args.mode == "auto" and target_count <= args.batch_threshold else args.mode
    mode = "batch-submit" if mode == "auto" else mode
    if mode == "sync":
        return run_sync(
            df=df,
            target_rows=target_rows,
            template=template,
            input_path=input_path,
            output_path=output_path,
            model=args.model,
        )
    if mode == "batch-submit":
        return run_batch_submit(
            df=df,
            target_rows=target_rows,
            template=template,
            template_path=DEFAULT_TEMPLATE_PATH.resolve(),
            input_path=input_path,
            output_path=output_path,
            model=args.model,
            completion_window=args.completion_window,
            job_dir=resolve_job_dir(input_path=input_path, job_dir=args.job_dir),
        )
    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc))

