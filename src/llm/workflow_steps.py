from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from tqdm import tqdm

from llm.client import (
    build_batch_requests,
    check_batch_status,
    classify_prompt,
    get_client,
    get_model_name,
    retrieve_batch_results,
    save_batch_input,
    submit_batch_job,
)
from llm.workflow_config import CONTEXT_COLUMNS
from llm.workflow_io import (
    apply_classification,
    build_custom_id,
    clean_cell,
    load_input_frame,
    load_manifest,
    render_prompt,
    resolve_output_path,
    save_manifest,
    save_output,
    validate_context_row,
)


def run_sync(
    *,
    df: pd.DataFrame,
    target_rows: pd.DataFrame,
    template: Template,
    input_path: Path,
    output_path: Path,
    model: str | None,
) -> dict[str, Any]:
    """Run synchronous classification for a small set of ETFs."""
    client = get_client()
    model_name = get_model_name(model)
    updated_df = df.copy()
    success_count = 0
    warnings: list[str] = []

    for row_index in tqdm(target_rows.index, total=len(target_rows), desc="Classifying ETFs"):
        try:
            result = classify_prompt(client=client, prompt=render_prompt(template, target_rows.loc[row_index]), model=model_name)
            apply_classification(updated_df, row_index, result)
            success_count += 1
        except Exception as exc:
            ticker = clean_cell(target_rows.loc[row_index, "ticker"])
            warnings.append(f"{ticker}: {exc}")

    save_output(updated_df, output_path)
    return _classification_summary("sync", model_name, input_path, output_path, len(target_rows), success_count, warnings)


def run_batch_submit(
    *,
    df: pd.DataFrame,
    target_rows: pd.DataFrame,
    template: Template,
    template_path: Path,
    input_path: Path,
    output_path: Path,
    model: str | None,
    completion_window: str,
    job_dir: Path,
) -> dict[str, Any]:
    """Prepare prompts, save batch artifacts, and submit a batch job."""
    client = get_client()
    model_name = get_model_name(model)
    job_dir.mkdir(parents=True, exist_ok=True)
    prompts_by_id, context_records = _build_batch_payloads(target_rows, template)

    batch_input_path = save_batch_input(build_batch_requests(prompts_by_id, model=model_name), job_dir / "batch_input.jsonl")
    context_path = job_dir / "batch_context.csv"
    pd.DataFrame(context_records, columns=CONTEXT_COLUMNS).to_csv(context_path, index=False)
    batch = submit_batch_job(
        client=client,
        input_path=batch_input_path,
        completion_window=completion_window,
        metadata={"project": "business-intelligence-etf-classification", "row_count": str(len(context_records))},
    )
    save_manifest(
        job_dir,
        {
            "batch_id": batch.id,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "template_path": str(template_path),
            "batch_input_path": str(batch_input_path.resolve()),
            "context_path": str(context_path.resolve()),
            "job_dir": str(job_dir.resolve()),
            "model": model_name,
            "target_rows": len(context_records),
        },
    )
    return {
        "mode": "batch-submit",
        "model": model_name,
        "batch_id": batch.id,
        "job_dir": str(job_dir),
        "batch_input_path": str(batch_input_path),
        "context_path": str(context_path),
        "output_path": str(output_path),
        "target_rows": len(context_records),
        "message": "Batch submitted. Run batch-status or batch-apply later with the same job_dir.",
    }


def run_batch_follow_up(args: argparse.Namespace) -> dict[str, Any]:
    """Run batch-status or batch-apply mode."""
    if args.job_dir is None:
        raise ValueError("--job-dir is required for batch-status and batch-apply.")

    manifest = load_manifest(args.job_dir.resolve())
    batch_id = args.batch_id or manifest["batch_id"]
    if args.mode == "batch-status":
        batch = check_batch_status(get_client(), batch_id)
        return {"mode": "batch-status", "batch_id": batch.id, "status": batch.status, "job_dir": str(args.job_dir.resolve())}
    if args.mode == "batch-apply":
        return run_batch_apply(job_dir=args.job_dir.resolve(), manifest=manifest, batch_id=batch_id, output_path=args.output)
    raise ValueError(f"Unsupported mode: {args.mode}")


def run_batch_apply(
    *,
    job_dir: Path,
    manifest: dict[str, Any],
    batch_id: str,
    output_path: Path | None,
) -> dict[str, Any]:
    """Download completed batch results and apply them to the original dataset."""
    input_path = Path(manifest["input_path"])
    resolved_output_path = resolve_output_path(input_path, output_path or Path(manifest["output_path"]))
    df = load_input_frame(input_path)
    updated_df = df.copy()
    context_df = pd.read_csv(Path(manifest["context_path"]))
    context_df["row_index"] = context_df["row_index"].astype(int)
    batch, results, errors = retrieve_batch_results(get_client(), batch_id)
    success_count = 0
    warnings: list[str] = []

    for context_row in context_df.to_dict(orient="records"):
        custom_id = str(context_row["custom_id"])
        row_index = int(context_row["row_index"])
        validate_context_row(df, row_index, context_row)
        if custom_id in results:
            apply_classification(updated_df, row_index, results[custom_id])
            success_count += 1
        else:
            warnings.append(f"{custom_id}: {errors.get(custom_id, 'No batch result returned for this row.')}")

    save_output(updated_df, resolved_output_path)
    return {
        "mode": "batch-apply",
        "batch_id": batch.id,
        "status": batch.status,
        "job_dir": str(job_dir),
        "output_path": str(resolved_output_path),
        "target_rows": len(context_df),
        "successful_rows": success_count,
        "error_rows": len(warnings),
        "warnings": warnings[:20],
    }


def _build_batch_payloads(target_rows: pd.DataFrame, template: Template) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Render prompts and context rows for a batch job."""
    prompts_by_id: dict[str, str] = {}
    context_records: list[dict[str, Any]] = []
    for row_index in tqdm(target_rows.index, total=len(target_rows), desc="Preparing batch"):
        row = target_rows.loc[row_index]
        custom_id = build_custom_id(row_index=row_index, ticker=row["ticker"])
        prompts_by_id[custom_id] = render_prompt(template, row)
        context_record = {field: clean_cell(row[field]) for field in CONTEXT_COLUMNS[2:]}
        context_record.update({"custom_id": custom_id, "row_index": int(row_index)})
        context_records.append(context_record)
    return prompts_by_id, context_records


def _classification_summary(
    mode: str,
    model_name: str,
    input_path: Path,
    output_path: Path,
    target_rows: int,
    success_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    """Build a compact summary for sync classification."""
    return {
        "mode": mode,
        "model": model_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "target_rows": target_rows,
        "successful_rows": success_count,
        "error_rows": len(warnings),
        "warnings": warnings[:20],
    }

