#!/usr/bin/env python3
"""Safeguarded prompt-baseline run scaffolding.

This module prepares canonical experiment artifact directories and metadata.
Full model inference remains an explicit runtime step; final Nahw-Passage runs
are disabled unless the caller opts in deliberately.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

from scripts.baseline_prompts import (
    PromptDemo,
    prompt_sha256,
    render_b1_prompt,
    render_b2_prompt,
)
from scripts.nahw_baseline_utils import parse_model_response


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs"
DEFAULT_MAX_NEW_TOKENS = 32
PRIVATE_REPOSITORY_ROOTS = (
    ROOT / "data" / "processed",
    DEFAULT_OUTPUTS,
)
EXPERIMENT_ID_RE = re.compile(
    r"^(B[0-2]|F[1-4])-P[0-9]+__"
    r"[a-z0-9][a-z0-9.-]*__"
    r"[a-z0-9][a-z0-9.-]*__"
    r"s[0-9]+__r[0-9]{2}$"
)


class RunSafetyError(ValueError):
    """Raised when a baseline run would violate a frozen safety rule."""


@dataclass(frozen=True)
class PromptRecord:
    record_id: str
    passage: str
    error: str
    gold_correction: str | None
    metadata: dict


@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    protocol_id: str
    model_slug: str
    evaluation_slug: str
    seed: int
    replicate: int


class GemmaGenerator:
    """Lazy, revision-pinned greedy Gemma generation backend."""

    def __init__(self, model_id: str, revision: str, max_new_tokens: int) -> None:
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None
        self.metadata = {
            "backend": "transformers",
            "model_id": model_id,
            "model_revision": revision,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "python_version": platform.python_version(),
        }

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except (ImportError, OSError) as error:
            raise RunSafetyError("Gemma inference dependencies are unavailable") from error

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            device = "cuda"
        else:
            dtype = torch.float32
            device = "cpu"
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                revision=self.revision,
                padding_side="left",
            )
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=dtype,
                device_map="auto",
            ).eval()
        except Exception as error:
            raise RunSafetyError("unable to initialize pinned Gemma backend") from error
        self.metadata.update(
            {
                "torch_version": torch.__version__,
                "transformers_version": importlib.metadata.version("transformers"),
                "device": device,
                "dtype": str(dtype),
                "cuda_available": torch.cuda.is_available(),
            }
        )

    def __call__(self, prompt: str) -> str:
        if self.model is None or self.processor is None:
            self._load()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        return self.processor.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_private_path(path: Path, *, label: str) -> Path:
    """Reject text-bearing repository paths outside ignored private roots."""

    resolved = Path(path).expanduser().resolve()
    if _is_relative_to(resolved, ROOT) and not any(
        _is_relative_to(resolved, root) for root in PRIVATE_REPOSITORY_ROOTS
    ):
        raise RunSafetyError(
            f"{label} path inside the repository must stay under "
            "data/processed/ or outputs/"
        )
    return resolved


def _require_string(value: object, *, field: str, line_number: int) -> str:
    if not isinstance(value, str):
        raise RunSafetyError(f"{field} must be a string at input line {line_number}")
    return value


def load_prompt_records(path: Path) -> list[PromptRecord]:
    """Load private prompt records without logging text-bearing fields."""

    input_path = validate_private_path(path, label="input")
    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RunSafetyError(
                        f"invalid JSON at input line {line_number}"
                    ) from error
                if not isinstance(payload, dict):
                    raise RunSafetyError(
                        f"input line {line_number} must be a JSON object"
                    )
                record_id = _require_string(
                    payload.get("record_id"),
                    field="record_id",
                    line_number=line_number,
                )
                if not record_id:
                    raise RunSafetyError(
                        f"record_id must be non-empty at input line {line_number}"
                    )
                if record_id in seen_ids:
                    raise RunSafetyError(f"duplicate record_id at input line {line_number}")
                passage = _require_string(
                    payload.get("passage"),
                    field="passage",
                    line_number=line_number,
                )
                error = _require_string(
                    payload.get("error"),
                    field="error",
                    line_number=line_number,
                )
                if not error:
                    raise RunSafetyError(
                        f"error must be non-empty at input line {line_number}"
                    )
                gold = payload.get("gold_correction")
                if gold is not None and not isinstance(gold, str):
                    raise RunSafetyError(
                        "gold_correction must be a string or null at "
                        f"input line {line_number}"
                    )
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise RunSafetyError(
                        f"metadata must be an object at input line {line_number}"
                    )
                records.append(
                    PromptRecord(
                        record_id=record_id,
                        passage=passage,
                        error=error,
                        gold_correction=gold,
                        metadata=metadata,
                    )
                )
                seen_ids.add(record_id)
    except OSError as error:
        raise RunSafetyError("unable to read private input file") from error
    if not records:
        raise RunSafetyError("private input file contains no records")
    return records


def load_b1_demos(path: Path) -> list[PromptDemo]:
    """Load exactly five demonstrations from the existing private bundle."""

    bundle_path = validate_private_path(path, label="bundle")
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunSafetyError("unable to read valid private B1 bundle") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RunSafetyError("B1 bundle must use schema_version 1")
    rows = payload.get("demonstrations")
    if not isinstance(rows, list) or len(rows) != 5:
        raise RunSafetyError("B1 bundle must contain exactly five demonstrations")
    demos: list[PromptDemo] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise RunSafetyError(f"B1 demonstration {index} must be an object")
        values = []
        for field in ("source", "error", "correction"):
            value = row.get(field)
            if not isinstance(value, str):
                raise RunSafetyError(
                    f"B1 demonstration {index} {field} must be a string"
                )
            values.append(value)
        demos.append(PromptDemo(*values))
    return demos


def load_protocol_demos(
    protocol_id: str,
    bundle_path: Path | None,
) -> list[PromptDemo]:
    if protocol_id == "B1-P1":
        if bundle_path is None:
            raise RunSafetyError("B1-P1 requires --bundle")
        return load_b1_demos(bundle_path)
    if protocol_id == "B2-P1":
        if bundle_path is not None:
            raise RunSafetyError("B2-P1 does not accept --bundle")
        return []
    raise RunSafetyError(f"Unsupported prompt protocol: {protocol_id}")


def validate_output_root(
    outputs_root: Path,
    *,
    allow_outside_private_output: bool,
) -> Path:
    resolved = Path(outputs_root).expanduser().resolve()
    if not allow_outside_private_output and not _is_relative_to(
        resolved, DEFAULT_OUTPUTS
    ):
        raise RunSafetyError(
            "private outputs must stay under outputs/ unless "
            "--allow-outside-private-output is set"
        )
    validate_private_path(resolved, label="output")
    return resolved


def render_record_prompt(
    protocol_id: str,
    demos: list[PromptDemo],
    record: PromptRecord,
) -> str:
    if protocol_id == "B1-P1":
        return render_b1_prompt(demos, record.passage, record.error)
    if protocol_id == "B2-P1":
        if demos:
            raise RunSafetyError("B2-P1 cannot render with demonstrations")
        return render_b2_prompt(record.passage, record.error)
    raise RunSafetyError(f"Unsupported prompt protocol: {protocol_id}")


def aggregate_prompt_sha256(prompt_hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(prompt_hashes).encode("utf-8")).hexdigest()


def summarize_prompt_predictions(
    rows: list[dict],
    *,
    expected_records: int,
) -> dict:
    scored = [row for row in rows if row["exact_match"] is not None]
    return {
        "number_of_records": expected_records,
        "completed_records": len(rows),
        "number_scored": len(scored),
        "number_correct": sum(row["exact_match"] is True for row in scored),
        "invalid_or_empty_count": sum(
            not row["parsed_correction"] for row in rows
        ),
        "suspicious_output_count": sum(
            bool(set(row["parsing_warnings"]) - {"outer_formatting_removed"})
            for row in rows
        ),
        "multi_token_count": sum(
            "multiple_words" in row["parsing_warnings"] for row in rows
        ),
    }


def _write_summary(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )


def execute_run(
    config: RunConfig,
    records: list[PromptRecord],
    demos: list[PromptDemo],
    generate: Callable[[str], str],
    *,
    outputs_root: Path,
    input_path: Path,
    prompt_template_path: Path,
    bundle_path: Path | None = None,
    runtime_metadata: dict | None = None,
    allow_outside_private_output: bool = False,
) -> dict:
    """Execute a private prompt run while retaining auditable failure artifacts."""

    validate_experiment_id(config.experiment_id)
    if not records:
        raise RunSafetyError("prompt inference requires at least one record")
    if config.protocol_id == "B1-P1" and len(demos) != 5:
        raise RunSafetyError("B1-P1 requires exactly five demonstrations")
    if config.protocol_id == "B2-P1" and demos:
        raise RunSafetyError("B2-P1 cannot execute with demonstrations")
    input_path = validate_private_path(input_path, label="input")
    prompt_template_path = Path(prompt_template_path).expanduser().resolve()
    if bundle_path is not None:
        bundle_path = validate_private_path(bundle_path, label="bundle")
    try:
        sha256_file(input_path)
        sha256_file(prompt_template_path)
        sha256_file(bundle_path)
    except OSError as error:
        raise RunSafetyError("unable to hash required run input") from error
    safe_outputs = validate_output_root(
        outputs_root,
        allow_outside_private_output=allow_outside_private_output,
    )
    run_dir = prepare_run_directory(safe_outputs, config.experiment_id)
    predictions_path = run_dir / "predictions.jsonl"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "run.log"
    prediction_rows: list[dict] = []
    prompt_hashes: list[str] = []
    try:
        with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                prompt = render_record_prompt(config.protocol_id, demos, record)
                current_prompt_hash = prompt_sha256(prompt)
                raw_response = generate(prompt)
                if not isinstance(raw_response, str):
                    raise TypeError("generation backend must return a string")
                parsed, warnings = parse_model_response(raw_response)
                exact_match = (
                    parsed == record.gold_correction
                    if record.gold_correction is not None
                    else None
                )
                row = {
                    "record_id": record.record_id,
                    "metadata": record.metadata,
                    "prompt": prompt,
                    "prompt_sha256": current_prompt_hash,
                    "raw_response": raw_response,
                    "parsed_correction": parsed,
                    "parsing_warnings": warnings,
                    "gold_correction": record.gold_correction,
                    "exact_match": exact_match,
                }
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                stream.flush()
                prediction_rows.append(row)
                prompt_hashes.append(current_prompt_hash)
        summary = build_summary(
            config,
            input_path=input_path,
            prompt_template_path=prompt_template_path,
            predictions_path=predictions_path,
            bundle_path=bundle_path,
            run_status="complete",
        )
        summary.update(
            {
                "aggregate_prompt_sha256": aggregate_prompt_sha256(prompt_hashes),
                "counts": summarize_prompt_predictions(
                    prediction_rows,
                    expected_records=len(records),
                ),
                "runtime": runtime_metadata or {},
            }
        )
        _write_summary(summary_path, summary)
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("run completed\n")
        return summary
    except Exception as error:
        invalid_summary = build_summary(
            config,
            input_path=input_path,
            prompt_template_path=prompt_template_path,
            predictions_path=(predictions_path if predictions_path.exists() else None),
            bundle_path=bundle_path,
            run_status="invalid",
        )
        invalid_summary.update(
            {
                "aggregate_prompt_sha256": aggregate_prompt_sha256(prompt_hashes),
                "counts": summarize_prompt_predictions(
                    prediction_rows,
                    expected_records=len(records),
                ),
                "runtime": runtime_metadata or {},
                "error_type": type(error).__name__,
            }
        )
        _write_summary(summary_path, invalid_summary)
        with log_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("run invalid: inference execution failed\n")
        raise RunSafetyError("inference execution failed; run preserved as invalid") from error


def experiment_id(
    protocol_id: str,
    model_slug: str,
    evaluation_slug: str,
    seed: int,
    replicate: int,
) -> str:
    run_id = f"{protocol_id}__{model_slug}__{evaluation_slug}__s{seed}__r{replicate:02d}"
    return validate_experiment_id(run_id)


def validate_experiment_id(run_id: str) -> str:
    if not EXPERIMENT_ID_RE.fullmatch(run_id):
        raise RunSafetyError(f"Invalid experiment ID: {run_id}")
    return run_id


def assert_final_eval_allowed(evaluation_slug: str, *, confirm_final_eval: bool) -> None:
    if evaluation_slug == "nahw-passage" and not confirm_final_eval:
        raise RunSafetyError(
            "Nahw-Passage final evaluation requires explicit confirmation"
        )


def prepare_run_directory(outputs_root: Path, run_id: str) -> Path:
    run_id = validate_experiment_id(run_id)
    run_dir = Path(outputs_root) / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise RunSafetyError(f"Run directory already exists: {run_dir}") from error
    return run_dir


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_summary(
    config: RunConfig,
    *,
    input_path: Path,
    prompt_template_path: Path,
    predictions_path: Path | None = None,
    bundle_path: Path | None = None,
    run_status: str = "planned",
) -> dict:
    """Build a corpus-text-free summary from artifact hashes."""

    validate_experiment_id(config.experiment_id)
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "run_status": run_status,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": git_commit_sha(),
        "config": asdict(config),
        "input_sha256": sha256_file(input_path),
        "prompt_template_sha256": sha256_file(prompt_template_path),
        "bundle_sha256": sha256_file(bundle_path),
        "prediction_sha256": sha256_file(predictions_path),
        "artifact_layout": {
            "predictions": "predictions.jsonl",
            "summary": "summary.json",
            "log": "run.log",
        },
        "safeguards": {
            "nahw_passage_requires_explicit_confirmation": True,
            "overwrite_existing_run_directory": False,
            "summary_contains_private_text": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", required=True, choices=("B1-P1", "B2-P1"))
    parser.add_argument("--model-slug", default="gemma3-4b-it")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--model-revision")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--evaluation-slug", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--confirm-final-eval", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-outside-private-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        assert_final_eval_allowed(
            args.evaluation_slug,
            confirm_final_eval=args.confirm_final_eval,
        )
        if args.max_new_tokens <= 0:
            raise RunSafetyError("--max-new-tokens must be positive")
        if args.execute and not args.model_revision:
            raise RunSafetyError("--execute requires --model-revision")
        safe_outputs = validate_output_root(
            args.outputs_root,
            allow_outside_private_output=args.allow_outside_private_output,
        )
        input_path = validate_private_path(args.input, label="input")
        prompt_template_path = Path(args.prompt_template).expanduser().resolve()
        bundle_path = (
            validate_private_path(args.bundle, label="bundle")
            if args.bundle is not None
            else None
        )
        run_id = experiment_id(
            args.protocol_id,
            args.model_slug,
            args.evaluation_slug,
            args.seed,
            args.replicate,
        )
        config = RunConfig(
            experiment_id=run_id,
            protocol_id=args.protocol_id,
            model_slug=args.model_slug,
            evaluation_slug=args.evaluation_slug,
            seed=args.seed,
            replicate=args.replicate,
        )
        if args.execute:
            records = load_prompt_records(input_path)
            demos = load_protocol_demos(args.protocol_id, bundle_path)
            generator = GemmaGenerator(
                args.model,
                args.model_revision,
                args.max_new_tokens,
            )
            summary = execute_run(
                config,
                records,
                demos,
                generator,
                outputs_root=safe_outputs,
                input_path=input_path,
                prompt_template_path=prompt_template_path,
                bundle_path=bundle_path,
                runtime_metadata=generator.metadata,
                allow_outside_private_output=args.allow_outside_private_output,
            )
            run_dir = safe_outputs / run_id
        else:
            run_dir = prepare_run_directory(safe_outputs, run_id)
            summary = build_summary(
                config,
                input_path=input_path,
                prompt_template_path=prompt_template_path,
                bundle_path=bundle_path,
                run_status="planned",
            )
            _write_summary(run_dir / "summary.json", summary)
            with (run_dir / "run.log").open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                stream.write("planned run scaffold created; model inference not executed\n")
    except (RunSafetyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(
        json.dumps(
            {
                "experiment_id": run_id,
                "run_dir": str(run_dir),
                "run_status": summary["run_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
