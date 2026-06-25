#!/usr/bin/env python3
"""Safeguarded prompt-baseline run scaffolding.

This module prepares canonical experiment artifact directories and metadata.
Full model inference remains an explicit runtime step; final Nahw-Passage runs
are disabled unless the caller opts in deliberately.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "outputs"
EXPERIMENT_ID_RE = re.compile(
    r"^(B[0-2]|F[1-4])-P[0-9]+__"
    r"[a-z0-9][a-z0-9.-]*__"
    r"[a-z0-9][a-z0-9.-]*__"
    r"s[0-9]+__r[0-9]{2}$"
)


class RunSafetyError(ValueError):
    """Raised when a baseline run would violate a frozen safety rule."""


@dataclass(frozen=True)
class RunConfig:
    experiment_id: str
    protocol_id: str
    model_slug: str
    evaluation_slug: str
    seed: int
    replicate: int


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
    parser.add_argument("--evaluation-slug", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--confirm-final-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        assert_final_eval_allowed(
            args.evaluation_slug,
            confirm_final_eval=args.confirm_final_eval,
        )
        run_id = experiment_id(
            args.protocol_id,
            args.model_slug,
            args.evaluation_slug,
            args.seed,
            args.replicate,
        )
        run_dir = prepare_run_directory(args.outputs_root, run_id)
        config = RunConfig(
            experiment_id=run_id,
            protocol_id=args.protocol_id,
            model_slug=args.model_slug,
            evaluation_slug=args.evaluation_slug,
            seed=args.seed,
            replicate=args.replicate,
        )
        summary = build_summary(
            config,
            input_path=args.input,
            prompt_template_path=args.prompt_template,
            bundle_path=args.bundle,
            run_status="planned",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run.log").write_text(
            "planned run scaffold created; model inference not executed\n",
            encoding="utf-8",
        )
    except (RunSafetyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(json.dumps({"experiment_id": run_id, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
