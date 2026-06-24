#!/usr/bin/env python3
"""Build text-free QALB selection manifests from the unchanged release archive."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


RELEASE = "0.9.1"
ROOT_NAME = "QALB-0.9.1-Dec03-2021-SharedTasks"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT / "data" / "raw" / "qalb" / "QALB-0.9.1-Dec03-2021-SharedTasks.zip"
)
DEFAULT_NAHW = ROOT / "data" / "raw" / "nahw" / "Nahw-Passage.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "qalb"
JSONL_OUTPUT_FILENAMES = (
    "qalb_registry.jsonl",
    "qalb_train_selection.jsonl",
    "qalb_dev_selection.jsonl",
)
SUMMARY_OUTPUT_FILENAME = "qalb_manifest_summary.json"
EXPECTED_OUTPUT_FILENAMES = frozenset(
    (*JSONL_OUTPUT_FILENAMES, SUMMARY_OUTPUT_FILENAME)
)
OUTPUT_TEMP_PREFIXES = tuple(
    f".{filename}." for filename in sorted(EXPECTED_OUTPUT_FILENAMES)
)
PUBLIC_RECORD_KEYS = {
    "record_key",
    "release",
    "year",
    "track",
    "split",
    "document_id",
    "line_number",
    "sent_member",
    "cor_member",
    "m2_member",
    "source_sha256",
    "correction_sha256",
    "source_codepoints",
    "correction_codepoints",
    "source_equals_correction",
    "duplicate_source_within_split",
    "exact_source_overlap_with_qalb_test",
    "exact_source_overlap_with_nahw",
    "eligible_for_training",
    "eligible_for_development",
    "selection_reason",
}
PUBLIC_METADATA_KEYS = {
    "archive_sha256",
    "archive_filename",
    "nahw_sha256",
    "nahw_filename",
    "member_sha256",
    "split_counts",
    "nahw_passage_source_sha256",
}


class ManifestError(ValueError):
    """Raised when QALB or Nahw inputs fail a reproducibility safeguard."""


@dataclass(frozen=True)
class SplitSpec:
    year: int
    track: str
    split: str

    @property
    def stem(self) -> str:
        title_split = {"train": "Train", "dev": "Dev", "test": "Test"}[self.split]
        return (
            f"{ROOT_NAME}/data/{self.year}/{self.split}/"
            f"QALB-{self.year}-{self.track}-{title_split}"
        )


SPLITS = (
    SplitSpec(2014, "L1", "train"),
    SplitSpec(2014, "L1", "dev"),
    SplitSpec(2014, "L1", "test"),
    SplitSpec(2015, "L1", "test"),
    SplitSpec(2015, "L2", "train"),
    SplitSpec(2015, "L2", "dev"),
    SplitSpec(2015, "L2", "test"),
)
CANONICAL_SPLIT_COUNTS = {
    (2014, "L1", "train"): 19411,
    (2014, "L1", "dev"): 1017,
    (2014, "L1", "test"): 968,
    (2015, "L1", "test"): 920,
    (2015, "L2", "train"): 310,
    (2015, "L2", "dev"): 154,
    (2015, "L2", "test"): 158,
}


def split_key(spec: SplitSpec) -> str:
    return f"{spec.year}:{spec.track}:{spec.split}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextmanager
def open_hashed_zip(path: Path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            yield archive, digest.hexdigest()


def decode_member(payload: bytes, member: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ManifestError(f"Archive member is not valid UTF-8: {member}") from error


def validate_archive_members(archive: zipfile.ZipFile) -> None:
    seen = set()
    for info in archive.infolist():
        member = info.filename
        if member in seen:
            raise ManifestError(f"Duplicate ZIP member name: {member}")
        seen.add(member)
        if (
            member.startswith("/")
            or "\\" in member
            or ".." in PurePosixPath(member).parts
            or (len(member) > 1 and member[1] == ":")
        ):
            raise ManifestError(f"Unsafe ZIP member path: {member}")
        if info.flag_bits & 0x1:
            raise ManifestError(f"Encrypted ZIP member is not supported: {member}")


def is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def required_member_names():
    names = {f"{ROOT_NAME}/README.txt", f"{ROOT_NAME}/LICENSE.txt"}
    for spec in SPLITS:
        names.update(f"{spec.stem}.{suffix}" for suffix in ("sent", "cor", "m2"))
    return names


def invalid_metadata(field: str) -> ManifestError:
    return ManifestError(f"Invalid manifest metadata field {field}")


def invalid_row(index: int, field: str) -> ManifestError:
    return ManifestError(f"Invalid registry row {index} field {field}")


def validate_manifest_inputs(registry, metadata) -> None:
    if not isinstance(registry, list) or any(
        not isinstance(row, dict) or set(row) != PUBLIC_RECORD_KEYS for row in registry
    ):
        raise ManifestError("Invalid registry schema")
    if not isinstance(metadata, dict) or set(metadata) != PUBLIC_METADATA_KEYS:
        raise ManifestError("Invalid manifest metadata schema")
    if not is_sha256(metadata["archive_sha256"]) or not is_sha256(
        metadata["nahw_sha256"]
    ):
        raise invalid_metadata("archive_sha256/nahw_sha256")
    for key in ("archive_filename", "nahw_filename"):
        filename = metadata[key]
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
            or (len(filename) > 1 and filename[1] == ":")
        ):
            raise invalid_metadata(key)
    member_hashes = metadata["member_sha256"]
    if (
        not isinstance(member_hashes, dict)
        or set(member_hashes) != required_member_names()
        or any(not is_sha256(digest) for digest in member_hashes.values())
    ):
        raise invalid_metadata("member_sha256")

    expected_split_keys = {split_key(spec) for spec in SPLITS}
    split_counts = metadata["split_counts"]
    if (
        not isinstance(split_counts, dict)
        or set(split_counts) != expected_split_keys
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in split_counts.values()
        )
    ):
        raise invalid_metadata("split_counts")
    nahw_hashes = metadata["nahw_passage_source_sha256"]
    if (
        not isinstance(nahw_hashes, list)
        or any(not is_sha256(digest) for digest in nahw_hashes)
        or nahw_hashes != sorted(set(nahw_hashes))
    ):
        raise invalid_metadata("nahw_passage_source_sha256")

    bool_fields = (
        "source_equals_correction",
        "duplicate_source_within_split",
        "exact_source_overlap_with_qalb_test",
        "exact_source_overlap_with_nahw",
        "eligible_for_training",
        "eligible_for_development",
    )
    valid_specs = {(spec.year, spec.track, spec.split): spec for spec in SPLITS}
    for index, row in enumerate(registry):
        if row["release"] != RELEASE:
            raise invalid_row(index, "release")
        if not isinstance(row["year"], int) or isinstance(row["year"], bool):
            raise invalid_row(index, "year")
        if not isinstance(row["track"], str):
            raise invalid_row(index, "track")
        if not isinstance(row["split"], str):
            raise invalid_row(index, "split")
        spec = valid_specs.get((row["year"], row["track"], row["split"]))
        if spec is None:
            raise invalid_row(index, "split")
        if not isinstance(row["line_number"], int) or isinstance(
            row["line_number"], bool
        ) or row["line_number"] <= 0:
            raise invalid_row(index, "line_number")
        if not isinstance(row["document_id"], str) or not row["document_id"]:
            raise invalid_row(index, "document_id")
        for field in ("source_codepoints", "correction_codepoints"):
            value = row[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise invalid_row(index, field)
        for field in ("source_sha256", "correction_sha256"):
            if not is_sha256(row[field]):
                raise invalid_row(index, field)
        for field in bool_fields:
            if not isinstance(row[field], bool):
                raise invalid_row(index, field)
        reasons = row["selection_reason"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise invalid_row(index, "selection_reason")
        for suffix in ("sent", "cor", "m2"):
            field = f"{suffix}_member"
            if row[field] != f"{spec.stem}.{suffix}":
                raise invalid_row(index, field)
        expected_record_key = (
            f"qalb-{RELEASE}:{spec.year}:{spec.track}:{spec.split}:"
            f"{row['line_number']:06d}:{row['document_id']}"
        )
        if row["record_key"] != expected_record_key:
            raise invalid_row(index, "record_key")
        if row["source_equals_correction"] != (
            row["source_sha256"] == row["correction_sha256"]
        ):
            raise invalid_row(index, "source_equals_correction")

    observed_counts = Counter(
        f"{row['year']}:{row['track']}:{row['split']}" for row in registry
    )
    if dict(sorted(observed_counts.items())) != split_counts:
        raise invalid_metadata("split_counts")
    within_counts = Counter(
        (row["year"], row["track"], row["split"], row["source_sha256"])
        for row in registry
    )
    qalb_test_hashes = {
        row["source_sha256"] for row in registry if row["split"] == "test"
    }
    nahw_hash_set = set(nahw_hashes)
    for index, row in enumerate(registry):
        group_key = (
            row["year"],
            row["track"],
            row["split"],
            row["source_sha256"],
        )
        qalb_overlap = row["source_sha256"] in qalb_test_hashes
        nahw_overlap = row["source_sha256"] in nahw_hash_set
        train_ok = row["split"] == "train" and not qalb_overlap and not nahw_overlap
        dev_ok = row["split"] == "dev" and not qalb_overlap and not nahw_overlap
        reasons = []
        if row["split"] == "test":
            reasons.append("official_test_split")
        if qalb_overlap and row["split"] != "test":
            reasons.append("exact_source_overlap_with_qalb_test")
        if nahw_overlap:
            reasons.append("exact_source_overlap_with_nahw")
        if train_ok:
            reasons.append("official_train_split")
        if dev_ok:
            reasons.append("official_dev_split")
        derived = {
            "duplicate_source_within_split": within_counts[group_key] > 1,
            "exact_source_overlap_with_qalb_test": qalb_overlap,
            "exact_source_overlap_with_nahw": nahw_overlap,
            "eligible_for_training": train_ok,
            "eligible_for_development": dev_ok,
            "selection_reason": reasons,
        }
        for field, expected in derived.items():
            if row[field] != expected:
                raise invalid_row(index, field)


def parse_sent(text: str, member: str):
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if " " not in line:
            raise ManifestError(f"Malformed .sent row: {member}:{line_number}")
        document_id, source = line.split(" ", 1)
        if not document_id or not source:
            raise ManifestError(f"Malformed .sent row: {member}:{line_number}")
        rows.append((document_id, source))
    return rows


def parse_cor(text: str, member: str):
    corrections = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("S "):
            raise ManifestError(f"Malformed .cor row: {member}:{line_number}")
        corrections.append(line[2:])
    return corrections


def parse_m2_sources(text: str):
    return [line[2:] for line in text.splitlines() if line.startswith("S ")]


def load_nahw_hashes(path: Path):
    try:
        payload = path.read_bytes()
        rows = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read Nahw-Passage JSON: {path.name}") from error
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("passage"), str)
        for row in rows
    ):
        raise ManifestError(
            "Nahw-Passage JSON must be a list of records with string passage fields"
        )
    return (
        {sha256_bytes(row["passage"].encode("utf-8")) for row in rows},
        sha256_bytes(payload),
    )


def build_manifest_data(
    archive_path: Path,
    nahw_path: Path,
    *,
    expected_split_counts=None,
):
    archive_path = Path(archive_path)
    nahw_path = Path(nahw_path)
    if not archive_path.is_file():
        raise ManifestError(f"Missing QALB archive: {archive_path.name}")
    if not nahw_path.is_file():
        raise ManifestError(f"Missing Nahw-Passage file: {nahw_path.name}")

    if expected_split_counts is None:
        expected_split_counts = CANONICAL_SPLIT_COUNTS
    expected_keys = {(spec.year, spec.track, spec.split) for spec in SPLITS}
    if (
        not isinstance(expected_split_counts, dict)
        or set(expected_split_counts) != expected_keys
        or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in expected_split_counts.values()
        )
    ):
        raise ManifestError("Invalid expected split counts")

    nahw_hashes, nahw_sha256 = load_nahw_hashes(nahw_path)
    records = []
    member_hashes = {}
    observed_split_counts = {}
    required_docs = {f"{ROOT_NAME}/README.txt", f"{ROOT_NAME}/LICENSE.txt"}

    with open_hashed_zip(archive_path) as (archive, archive_sha256):
        validate_archive_members(archive)
        names = set(archive.namelist())
        missing_docs = sorted(required_docs - names)
        if missing_docs:
            raise ManifestError(f"Missing required archive member: {missing_docs[0]}")
        for member in sorted(required_docs):
            payload = archive.read(member)
            decode_member(payload, member)
            member_hashes[member] = sha256_bytes(payload)

        for spec in SPLITS:
            members = {
                suffix: f"{spec.stem}.{suffix}" for suffix in ("sent", "cor", "m2")
            }
            for member in members.values():
                if member not in names:
                    raise ManifestError(f"Missing required archive member: {member}")
            payloads = {
                suffix: archive.read(member) for suffix, member in members.items()
            }
            member_hashes.update(
                {
                    members[suffix]: sha256_bytes(payload)
                    for suffix, payload in payloads.items()
                }
            )
            sent_rows = parse_sent(
                decode_member(payloads["sent"], members["sent"]), members["sent"]
            )
            corrections = parse_cor(
                decode_member(payloads["cor"], members["cor"]), members["cor"]
            )
            m2_sources = parse_m2_sources(
                decode_member(payloads["m2"], members["m2"])
            )
            if len({len(sent_rows), len(corrections), len(m2_sources)}) != 1:
                raise ManifestError(
                    f"Parallel record count mismatch: {spec.stem} "
                    f"(sent={len(sent_rows)}, cor={len(corrections)}, "
                    f"m2={len(m2_sources)})"
                )
            observed_count = len(sent_rows)
            observed_split_counts[split_key(spec)] = observed_count
            expected_count = expected_split_counts[
                (spec.year, spec.track, spec.split)
            ]
            if observed_count != expected_count:
                raise ManifestError(
                    f"Split count mismatch: {split_key(spec)} "
                    f"observed={observed_count} expected={expected_count}"
                )
            document_ids = [document_id for document_id, _ in sent_rows]
            if len(document_ids) != len(set(document_ids)):
                raise ManifestError(f"Duplicate document ID: {spec.stem}")
            sources = [source for _, source in sent_rows]
            if sources != m2_sources:
                mismatch_line = next(
                    line_number
                    for line_number, (source, m2_source) in enumerate(
                        zip(sources, m2_sources), 1
                    )
                    if source != m2_source
                )
                raise ManifestError(
                    f"M2 source order mismatch: {spec.stem} at line {mismatch_line}"
                )
            for line_number, ((document_id, source), correction) in enumerate(
                zip(sent_rows, corrections), 1
            ):
                source_hash = sha256_bytes(source.encode("utf-8"))
                records.append(
                    {
                        "record_key": (
                            f"qalb-{RELEASE}:{spec.year}:{spec.track}:{spec.split}:"
                            f"{line_number:06d}:{document_id}"
                        ),
                        "release": RELEASE,
                        "year": spec.year,
                        "track": spec.track,
                        "split": spec.split,
                        "document_id": document_id,
                        "line_number": line_number,
                        "sent_member": members["sent"],
                        "cor_member": members["cor"],
                        "m2_member": members["m2"],
                        "source_sha256": source_hash,
                        "correction_sha256": sha256_bytes(correction.encode("utf-8")),
                        "source_codepoints": len(source),
                        "correction_codepoints": len(correction),
                        "source_equals_correction": source == correction,
                    }
                )

    within_counts = Counter(
        (row["year"], row["track"], row["split"], row["source_sha256"])
        for row in records
    )
    qalb_test_hashes = {
        row["source_sha256"] for row in records if row["split"] == "test"
    }
    for row in records:
        group_key = (
            row["year"],
            row["track"],
            row["split"],
            row["source_sha256"],
        )
        qalb_overlap = row["source_sha256"] in qalb_test_hashes
        nahw_overlap = row["source_sha256"] in nahw_hashes
        reasons = []
        train_ok = row["split"] == "train" and not qalb_overlap and not nahw_overlap
        dev_ok = row["split"] == "dev" and not qalb_overlap and not nahw_overlap
        if row["split"] == "test":
            reasons.append("official_test_split")
        if qalb_overlap and row["split"] != "test":
            reasons.append("exact_source_overlap_with_qalb_test")
        if nahw_overlap:
            reasons.append("exact_source_overlap_with_nahw")
        if train_ok:
            reasons.append("official_train_split")
        if dev_ok:
            reasons.append("official_dev_split")
        row.update(
            {
                "duplicate_source_within_split": within_counts[group_key] > 1,
                "exact_source_overlap_with_qalb_test": qalb_overlap,
                "exact_source_overlap_with_nahw": nahw_overlap,
                "eligible_for_training": train_ok,
                "eligible_for_development": dev_ok,
                "selection_reason": reasons,
            }
        )

    metadata = {
        "archive_sha256": archive_sha256,
        "archive_filename": archive_path.name,
        "nahw_sha256": nahw_sha256,
        "nahw_filename": nahw_path.name,
        "member_sha256": dict(sorted(member_hashes.items())),
        "split_counts": dict(sorted(observed_split_counts.items())),
        "nahw_passage_source_sha256": sorted(nahw_hashes),
    }
    return records, metadata


def render_jsonl(rows) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def remove_generator_orphan_temps(output_dir: Path) -> None:
    for entry in output_dir.iterdir():
        if not any(
            entry.name.startswith(prefix) and len(entry.name) > len(prefix)
            for prefix in OUTPUT_TEMP_PREFIXES
        ):
            continue
        if entry.is_symlink() or entry.is_file():
            entry.unlink()


def write_manifests(registry, metadata, output_dir: Path):
    validate_manifest_inputs(registry, metadata)
    output_dir = Path(output_dir)
    train_rows = [row for row in registry if row["eligible_for_training"]]
    dev_rows = [row for row in registry if row["eligible_for_development"]]
    payloads = {
        "qalb_registry.jsonl": render_jsonl(registry),
        "qalb_train_selection.jsonl": render_jsonl(train_rows),
        "qalb_dev_selection.jsonl": render_jsonl(dev_rows),
    }
    summary = {
        "schema_version": 1,
        "release": RELEASE,
        "inputs": metadata,
        "counts": {
            "registry": len(registry),
            "train_selected": len(train_rows),
            "dev_selected": len(dev_rows),
            "test_records": sum(row["split"] == "test" for row in registry),
            "within_split_duplicate_records_flagged": sum(
                row["duplicate_source_within_split"] for row in registry
            ),
            "train_dev_qalb_test_overlap_excluded": sum(
                row["split"] in {"train", "dev"}
                and row["exact_source_overlap_with_qalb_test"]
                for row in registry
            ),
            "train_dev_nahw_overlap_excluded": sum(
                row["split"] in {"train", "dev"}
                and row["exact_source_overlap_with_nahw"]
                for row in registry
            ),
        },
        "selection_policy": {
            "preserve_within_split_duplicates": True,
            "exclude_exact_qalb_test_overlap": True,
            "exclude_exact_nahw_overlap": True,
            "normalization": (
                "none; exact UTF-8 strings after file-format prefix removal"
            ),
            "qalb_test_role": "evaluation-only",
        },
        "output_sha256": {
            name: sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
    }
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_generator_orphan_temps(output_dir)
    existing_names = {path.name for path in output_dir.iterdir()}
    if not existing_names <= EXPECTED_OUTPUT_FILENAMES:
        raise ManifestError("Output directory contains unexpected output entries")
    summary_path = output_dir / SUMMARY_OUTPUT_FILENAME
    summary_path.unlink(missing_ok=True)
    try:
        for name, payload in payloads.items():
            atomic_write(output_dir / name, payload)
        atomic_write(summary_path, summary_payload)
    except BaseException:
        summary_path.unlink(missing_ok=True)
        raise
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--nahw-passage", type=Path, default=DEFAULT_NAHW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        registry, metadata = build_manifest_data(args.archive, args.nahw_passage)
        summary = write_manifests(registry, metadata, args.output_dir)
    except (ManifestError, zipfile.BadZipFile, OSError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    print(f"Registry records: {summary['counts']['registry']}")
    print(f"Training selected: {summary['counts']['train_selected']}")
    print(f"Development selected: {summary['counts']['dev_selected']}")
    print(f"Private outputs: {args.output_dir}")
    print(
        "IMPORTANT: QALB test records are evaluation-only; never commit generated "
        "manifests or corpus data."
    )


if __name__ == "__main__":
    main()
