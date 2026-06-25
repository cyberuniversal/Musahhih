#!/usr/bin/env python3
"""Build the private B1-P1 prompt demonstration bundle.

The public functions are intentionally testable with synthetic fixtures. Real
QALB text-bearing output must stay in ignored private paths such as
``data/processed/qalb/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.prepare_qalb_manifests import (
    DEFAULT_ARCHIVE,
    DEFAULT_OUTPUT_DIR,
    RELEASE,
    ROOT_NAME,
    SplitSpec,
    decode_member,
    open_hashed_zip,
    parse_sent,
    sha256_bytes,
)


B1_PROTOCOL_ID = "B1-P1"
EXPECTED_CANDIDATE_ANNOTATIONS = 3116
EXPECTED_DISTINCT_RECORDS = 458
EXPECTED_SELECTED_IDENTITY_SHA256 = (
    "76edd4c3de4b6cb5a985464faa066dea40faf9b25b8fa2912b3bf9c4750a9e8c"
)
DEFAULT_TRAIN_MANIFEST = DEFAULT_OUTPUT_DIR / "qalb_train_selection.jsonl"
DEFAULT_BUNDLE = DEFAULT_OUTPUT_DIR / "b1_p1_prompt_bundle.json"
PRIVATE_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR.resolve()


class BundleError(ValueError):
    """Raised when the frozen B1 bundle selection safeguards fail."""


@dataclass(frozen=True)
class M2Edit:
    start: int
    end: int
    kind: str
    correction: str


@dataclass(frozen=True)
class PrivateQalbRecord:
    record_key: str
    release: str
    year: int
    track: str
    split: str
    eligible_for_training: bool
    source: str
    m2_block: str


@dataclass(frozen=True)
class SelectedCandidate:
    record_key: str
    start: int
    end: int
    source: str
    error: str
    correction: str
    identity: str
    identity_digest: str
    source_sha256: str


def parse_m2_edits(block: str) -> list[M2Edit]:
    """Parse M2 edit lines without normalizing Arabic text."""

    edits: list[M2Edit] = []
    for line in block.splitlines():
        if not line.startswith("A "):
            continue
        fields = line[2:].split("|||")
        if len(fields) < 3:
            raise BundleError("Malformed M2 annotation")
        span = fields[0].split()
        if len(span) != 2:
            raise BundleError("Malformed M2 span")
        try:
            start = int(span[0])
            end = int(span[1])
        except ValueError as error:
            raise BundleError("Malformed M2 span") from error
        edits.append(
            M2Edit(
                start=start,
                end=end,
                kind=fields[1],
                correction=fields[2],
            )
        )
    return edits


def candidate_identity(record_key: str, start: int, end: int) -> str:
    return f"{record_key}|{start}:{end}"


def candidate_identity_digest(identity: str) -> str:
    payload = f"{identity}|{B1_PROTOCOL_ID}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_identity_hash(identities: list[str]) -> str:
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def _candidate_from_edit(
    record: PrivateQalbRecord,
    edit: M2Edit,
) -> SelectedCandidate | None:
    if (
        record.release != RELEASE
        or record.year != 2014
        or record.track != "L1"
        or record.split != "train"
        or not record.eligible_for_training
        or edit.kind != "Edit"
    ):
        return None
    source_tokens = record.source.split()
    correction_tokens = edit.correction.split()
    if not (5 <= len(source_tokens) <= 40):
        return None
    if edit.end - edit.start != 1:
        return None
    if edit.start < 0 or edit.end > len(source_tokens):
        return None
    if len(correction_tokens) != 1:
        return None
    error = source_tokens[edit.start]
    correction = correction_tokens[0]
    if not correction or correction == error:
        return None
    if source_tokens.count(error) != 1:
        return None

    identity = candidate_identity(record.record_key, edit.start, edit.end)
    return SelectedCandidate(
        record_key=record.record_key,
        start=edit.start,
        end=edit.end,
        source=record.source,
        error=error,
        correction=correction,
        identity=identity,
        identity_digest=candidate_identity_digest(identity),
        source_sha256=sha256_bytes(record.source.encode("utf-8")),
    )


def build_candidates(records: list[PrivateQalbRecord]) -> list[SelectedCandidate]:
    candidates: list[SelectedCandidate] = []
    for record in records:
        for edit in parse_m2_edits(record.m2_block):
            candidate = _candidate_from_edit(record, edit)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def select_b1_candidates(
    records: list[PrivateQalbRecord],
    *,
    limit: int = 5,
) -> tuple[list[SelectedCandidate], dict]:
    """Select B1 demonstrations with the frozen content-neutral rule."""

    candidates = build_candidates(records)
    selected: list[SelectedCandidate] = []
    seen_records: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.identity_digest):
        if candidate.record_key in seen_records:
            continue
        selected.append(candidate)
        seen_records.add(candidate.record_key)
        if len(selected) == limit:
            break
    if len(selected) < limit and len(candidates) >= limit:
        raise BundleError("Could not select the requested number of distinct records")

    identities = [candidate.identity for candidate in selected]
    summary = {
        "schema_version": 1,
        "protocol_id": B1_PROTOCOL_ID,
        "candidate_annotations": len(candidates),
        "distinct_candidate_records": len({candidate.record_key for candidate in candidates}),
        "selected_count": len(selected),
        "selected_identities": identities,
        "selected_identity_sha256": candidate_identity_hash(identities),
    }
    return selected, summary


def validate_selection_summary(
    summary: dict,
    *,
    expected_candidate_annotations: int = EXPECTED_CANDIDATE_ANNOTATIONS,
    expected_distinct_records: int = EXPECTED_DISTINCT_RECORDS,
    expected_identity_sha256: str = EXPECTED_SELECTED_IDENTITY_SHA256,
) -> None:
    if summary["candidate_annotations"] != expected_candidate_annotations:
        raise BundleError("Unexpected B1 candidate annotation count")
    if summary["distinct_candidate_records"] != expected_distinct_records:
        raise BundleError("Unexpected B1 distinct candidate record count")
    if summary["selected_identity_sha256"] != expected_identity_sha256:
        raise BundleError("Unexpected B1 selected identity SHA-256")


def write_private_bundle(
    output_path: Path,
    selected: list[SelectedCandidate],
    summary: dict,
    *,
    allow_outside_private_root: bool = False,
) -> dict:
    """Write a text-bearing private bundle and refuse overwrite."""

    output_path = Path(output_path)
    resolved_output = output_path.resolve()
    if not allow_outside_private_root:
        try:
            resolved_output.relative_to(PRIVATE_OUTPUT_ROOT)
        except ValueError as error:
            raise BundleError(
                "Private B1 bundle output must stay under "
                f"{PRIVATE_OUTPUT_ROOT}. Use --allow-outside-private-output only "
                "for temporary local diagnostics, never for committed artifacts."
            ) from error
    if output_path.exists():
        raise BundleError(f"Private bundle already exists: {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol_id": B1_PROTOCOL_ID,
        "selection": summary,
        "demonstrations": [
            {
                "record_key": candidate.record_key,
                "identity": candidate.identity,
                "identity_digest": candidate.identity_digest,
                "source_sha256": candidate.source_sha256,
                "source": candidate.source,
                "error": candidate.error,
                "correction": candidate.correction,
            }
            for candidate in selected
        ],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return {
        "bundle_path": str(output_path),
        "bundle_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "selected_identity_sha256": summary["selected_identity_sha256"],
    }


def _load_manifest_rows(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_2014_l1_train_blocks(archive_path: Path) -> tuple[list[str], list[str], str]:
    spec = SplitSpec(2014, "L1", "train")
    sent_member = f"{spec.stem}.sent"
    m2_member = f"{spec.stem}.m2"
    with open_hashed_zip(Path(archive_path)) as (archive, archive_sha256):
        try:
            sent_text = decode_member(archive.read(sent_member), sent_member)
            m2_text = decode_member(archive.read(m2_member), m2_member)
        except KeyError as error:
            raise BundleError("Missing required QALB 2014 L1 train member") from error
    sent_rows = parse_sent(sent_text, sent_member)
    m2_blocks = [block for block in m2_text.split("\n\n") if block.strip()]
    if len(sent_rows) != len(m2_blocks):
        raise BundleError("QALB train .sent and .m2 record counts differ")
    sources = [source for _, source in sent_rows]
    for index, (source, block) in enumerate(zip(sources, m2_blocks), 1):
        first_line = next((line for line in block.splitlines() if line.startswith("S ")), "")
        if first_line[2:] != source:
            raise BundleError(f"QALB M2 source mismatch at train line {index}")
    return sources, m2_blocks, archive_sha256


def load_private_records_from_archive(
    archive_path: Path,
    train_manifest_path: Path,
) -> tuple[list[PrivateQalbRecord], str]:
    """Load text-bearing private records from QALB archive and train manifest."""

    manifest_rows = _load_manifest_rows(train_manifest_path)
    sources, m2_blocks, archive_sha256 = _load_2014_l1_train_blocks(archive_path)
    records: list[PrivateQalbRecord] = []
    for row in manifest_rows:
        if not (
            row["release"] == RELEASE
            and row["year"] == 2014
            and row["track"] == "L1"
            and row["split"] == "train"
        ):
            continue
        line_number = row["line_number"]
        if not isinstance(line_number, int) or line_number < 1 or line_number > len(sources):
            raise BundleError("Invalid train manifest line number")
        source = sources[line_number - 1]
        if sha256_bytes(source.encode("utf-8")) != row["source_sha256"]:
            raise BundleError("Train manifest source hash mismatch")
        records.append(
            PrivateQalbRecord(
                record_key=row["record_key"],
                release=row["release"],
                year=row["year"],
                track=row["track"],
                split=row["split"],
                eligible_for_training=row["eligible_for_training"],
                source=source,
                m2_block=m2_blocks[line_number - 1],
            )
        )
    return records, archive_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--allow-outside-private-output",
        action="store_true",
        help=(
            "Allow writing the text-bearing private bundle outside "
            "data/processed/qalb/ for temporary local diagnostics only."
        ),
    )
    parser.add_argument(
        "--skip-frozen-count-checks",
        action="store_true",
        help="Only for synthetic/local diagnostics; do not use for the frozen run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        records, archive_sha256 = load_private_records_from_archive(
            args.archive,
            args.train_manifest,
        )
        selected, summary = select_b1_candidates(records, limit=args.limit)
        if not args.skip_frozen_count_checks:
            validate_selection_summary(summary)
        metadata = write_private_bundle(
            args.output,
            selected,
            summary,
            allow_outside_private_root=args.allow_outside_private_output,
        )
    except (BundleError, OSError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        raise SystemExit(f"ERROR: {error}") from error
    public_summary = {
        "archive_sha256": archive_sha256,
        "bundle_sha256": metadata["bundle_sha256"],
        "candidate_annotations": summary["candidate_annotations"],
        "distinct_candidate_records": summary["distinct_candidate_records"],
        "selected_count": summary["selected_count"],
        "selected_identity_sha256": summary["selected_identity_sha256"],
    }
    print(json.dumps(public_summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
