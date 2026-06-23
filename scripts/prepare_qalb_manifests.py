#!/usr/bin/env python3
"""Build text-free QALB selection manifests from the unchanged release archive."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


RELEASE = "0.9.1"
ROOT_NAME = "QALB-0.9.1-Dec03-2021-SharedTasks"
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_member(payload: bytes, member: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ManifestError(f"Archive member is not valid UTF-8: {member}") from error


def validate_archive_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        member = info.filename
        if (
            member.startswith("/")
            or "\\" in member
            or ".." in PurePosixPath(member).parts
            or (len(member) > 1 and member[1] == ":")
        ):
            raise ManifestError(f"Unsafe ZIP member path: {member}")
        if info.flag_bits & 0x1:
            raise ManifestError(f"Encrypted ZIP member is not supported: {member}")


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
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot read Nahw-Passage JSON: {path.name}") from error
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("passage"), str)
        for row in rows
    ):
        raise ManifestError(
            "Nahw-Passage JSON must be a list of records with string passage fields"
        )
    return {sha256_bytes(row["passage"].encode("utf-8")) for row in rows}


def build_manifest_data(archive_path: Path, nahw_path: Path):
    archive_path = Path(archive_path)
    nahw_path = Path(nahw_path)
    if not archive_path.is_file():
        raise ManifestError(f"Missing QALB archive: {archive_path.name}")
    if not nahw_path.is_file():
        raise ManifestError(f"Missing Nahw-Passage file: {nahw_path.name}")

    nahw_hashes = load_nahw_hashes(nahw_path)
    records = []
    member_hashes = {}
    required_docs = {f"{ROOT_NAME}/README.txt", f"{ROOT_NAME}/LICENSE.txt"}

    with zipfile.ZipFile(archive_path) as archive:
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
                raise ManifestError(f"Parallel record count mismatch: {spec.stem}")
            document_ids = [document_id for document_id, _ in sent_rows]
            if len(document_ids) != len(set(document_ids)):
                raise ManifestError(f"Duplicate document ID: {spec.stem}")
            if [source for _, source in sent_rows] != m2_sources:
                raise ManifestError(f"M2 source order mismatch: {spec.stem}")
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
        "archive_sha256": sha256_path(archive_path),
        "archive_filename": archive_path.name,
        "nahw_sha256": sha256_path(nahw_path),
        "nahw_filename": nahw_path.name,
        "member_sha256": dict(sorted(member_hashes.items())),
    }
    return records, metadata
