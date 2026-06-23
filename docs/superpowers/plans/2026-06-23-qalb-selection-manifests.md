# QALB Selection Manifests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build deterministic, text-free QALB 0.9.1 registry and train/dev selection manifests directly from the unchanged licensed ZIP while excluding exact QALB-test and Nahw overlaps.

**Architecture:** A single standard-library Python CLI validates the canonical archive, parses the seven official split/track groups in fixed order, computes exact UTF-8 identities, and applies the approved selection policy in memory. It writes a complete registry, eligible train/dev selections, and a checksum summary only after validation succeeds; generated private files remain under Git-ignored `data/processed/qalb/`.

**Tech Stack:** Python 3 standard library (`argparse`, `collections`, `dataclasses`, `hashlib`, `json`, `pathlib`, `tempfile`, `zipfile`), `unittest`, Git, Notion Research Hub.

---

## File map

- Create `scripts/prepare_qalb_manifests.py`: archive validation, parsing, selection, deterministic serialization, and CLI.
- Create `tests/test_prepare_qalb_manifests.py`: synthetic ZIP fixtures and red/green behavior tests without real QALB text.
- Modify `README.md`: private QALB manifest command and licensing boundary.
- Modify `docs/dataset_audit.md`: finalized selection policy and verified output counts/hashes.
- Modify `results/qalb_0.9.1_intake.md`: actual manifest-generation verification without corpus text.
- Generate ignored files under `data/processed/qalb/`: registry, train selection, dev selection, and summary.

## Public record contract

The implementation must emit exactly these record keys:

```python
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
```

No source, correction, annotation, prompt, completion, or absolute filesystem path may appear in a generated record.

### Task 1: Build a synthetic archive fixture and the happy-path registry

**Files:**
- Create: `tests/test_prepare_qalb_manifests.py`
- Create: `scripts/prepare_qalb_manifests.py`

- [ ] **Step 1: Write the first failing test and fixture builder**

Create `tests/test_prepare_qalb_manifests.py` with this initial content:

```python
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_qalb_manifests import PUBLIC_RECORD_KEYS, build_manifest_data


ROOT_NAME = "QALB-0.9.1-Dec03-2021-SharedTasks"
GROUPS = {
    (2014, "L1", "train"): [
        ("t1.ar", "TRAIN_KEEP", "TRAIN_FIXED"),
        ("t2.ar", "TRAIN_DUP", "TRAIN_DUP_FIXED_1"),
        ("t3.ar", "TRAIN_DUP", "TRAIN_DUP_FIXED_2"),
        ("t4.ar", "QALB_TEST_MATCH", "TRAIN_LEAK_FIXED"),
    ],
    (2014, "L1", "dev"): [
        ("d1.ar", "DEV_KEEP", "DEV_FIXED"),
        ("d2.ar", "NAHW_MATCH", "DEV_NAHW_FIXED"),
    ],
    (2014, "L1", "test"): [("q1.ar", "QALB_TEST_MATCH", "TEST_FIXED")],
    (2015, "L1", "test"): [("q2.ar", "L1_TEST_ONLY", "L1_TEST_FIXED")],
    (2015, "L2", "train"): [("l2t.ar", "L2_TRAIN_KEEP", "L2_TRAIN_FIXED")],
    (2015, "L2", "dev"): [("l2d.ar", "L2_DEV_KEEP", "L2_DEV_FIXED")],
    (2015, "L2", "test"): [("l2q.ar", "L2_TEST_ONLY", "L2_TEST_FIXED")],
}


def member_stem(year, track, split):
    title_split = {"train": "Train", "dev": "Dev", "test": "Test"}[split]
    return f"{ROOT_NAME}/data/{year}/{split}/QALB-{year}-{track}-{title_split}"


def group_members(rows):
    sent = "".join(f"{doc_id} {source}\n" for doc_id, source, _ in rows)
    cor = "".join(f"S {correction}\n" for _, _, correction in rows)
    m2 = "".join(f"S {source}\n\n" for _, source, _ in rows)
    return sent.encode("utf-8-sig"), cor.encode("utf-8-sig"), m2.encode("utf-8-sig")


def write_fixture_archive(path, groups=GROUPS, extra_members=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{ROOT_NAME}/README.txt", "fixture readme".encode("utf-8-sig"))
        archive.writestr(f"{ROOT_NAME}/LICENSE.txt", "fixture license".encode("utf-8-sig"))
        for (year, track, split), rows in groups.items():
            sent, cor, m2 = group_members(rows)
            stem = member_stem(year, track, split)
            archive.writestr(f"{stem}.sent", sent)
            archive.writestr(f"{stem}.cor", cor)
            archive.writestr(f"{stem}.m2", m2)
        for name, payload in extra_members or []:
            archive.writestr(name, payload)


class QalbManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.archive = self.root / "qalb.zip"
        self.nahw = self.root / "Nahw-Passage.json"
        write_fixture_archive(self.archive)
        self.nahw.write_text(
            json.dumps([{"passage": "NAHW_MATCH"}], ensure_ascii=False),
            encoding="utf-8",
        )

    def test_preserves_within_train_duplicates_and_applies_leakage_policy(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)

        self.assertEqual(len(registry), 11)
        self.assertTrue(all(set(row) == PUBLIC_RECORD_KEYS for row in registry))
        duplicate_hash = hashlib.sha256("TRAIN_DUP".encode("utf-8")).hexdigest()
        duplicate_rows = [row for row in registry if row["source_sha256"] == duplicate_hash]
        self.assertEqual(len(duplicate_rows), 2)
        self.assertTrue(all(row["duplicate_source_within_split"] for row in duplicate_rows))
        self.assertTrue(all(row["eligible_for_training"] for row in duplicate_rows))

        qalb_leak = next(row for row in registry if row["document_id"] == "t4.ar")
        self.assertTrue(qalb_leak["exact_source_overlap_with_qalb_test"])
        self.assertFalse(qalb_leak["eligible_for_training"])

        nahw_leak = next(row for row in registry if row["document_id"] == "d2.ar")
        self.assertTrue(nahw_leak["exact_source_overlap_with_nahw"])
        self.assertFalse(nahw_leak["eligible_for_development"])

        self.assertTrue(all(not row["eligible_for_training"] for row in registry if row["split"] == "test"))
        self.assertTrue(all(not row["eligible_for_development"] for row in registry if row["split"] == "test"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
python -m unittest tests.test_prepare_qalb_manifests.QalbManifestTests.test_preserves_within_train_duplicates_and_applies_leakage_policy -v
```

Expected: import failure stating that `scripts.prepare_qalb_manifests` does not exist.

- [ ] **Step 3: Implement the minimal archive parser and selection policy**

Create `scripts/prepare_qalb_manifests.py` with these interfaces and behavior:

```python
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
    "record_key", "release", "year", "track", "split", "document_id",
    "line_number", "sent_member", "cor_member", "m2_member",
    "source_sha256", "correction_sha256", "source_codepoints",
    "correction_codepoints", "source_equals_correction",
    "duplicate_source_within_split", "exact_source_overlap_with_qalb_test",
    "exact_source_overlap_with_nahw", "eligible_for_training",
    "eligible_for_development", "selection_reason",
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
        return f"{ROOT_NAME}/data/{self.year}/{self.split}/QALB-{self.year}-{self.track}-{title_split}"


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
    return payload.decode("utf-8-sig")


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
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("passage"), str) for row in rows):
        raise ManifestError("Nahw-Passage JSON must be a list of records with string passage fields")
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
        names = set(archive.namelist())
        missing_docs = sorted(required_docs - names)
        if missing_docs:
            raise ManifestError(f"Missing required archive member: {missing_docs[0]}")
        for member in sorted(required_docs):
            payload = archive.read(member)
            decode_member(payload, member)
            member_hashes[member] = sha256_bytes(payload)

        for spec in SPLITS:
            members = {suffix: f"{spec.stem}.{suffix}" for suffix in ("sent", "cor", "m2")}
            for member in members.values():
                if member not in names:
                    raise ManifestError(f"Missing required archive member: {member}")
            payloads = {suffix: archive.read(member) for suffix, member in members.items()}
            member_hashes.update({members[suffix]: sha256_bytes(payload) for suffix, payload in payloads.items()})
            sent_rows = parse_sent(decode_member(payloads["sent"], members["sent"]), members["sent"])
            corrections = parse_cor(decode_member(payloads["cor"], members["cor"]), members["cor"])
            m2_sources = parse_m2_sources(decode_member(payloads["m2"], members["m2"]))
            for line_number, ((document_id, source), correction) in enumerate(zip(sent_rows, corrections), 1):
                source_hash = sha256_bytes(source.encode("utf-8"))
                records.append({
                    "record_key": f"qalb-{RELEASE}:{spec.year}:{spec.track}:{spec.split}:{line_number:06d}:{document_id}",
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
                })

    within_counts = Counter((row["year"], row["track"], row["split"], row["source_sha256"]) for row in records)
    qalb_test_hashes = {row["source_sha256"] for row in records if row["split"] == "test"}
    for row in records:
        group_key = (row["year"], row["track"], row["split"], row["source_sha256"])
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
        row.update({
            "duplicate_source_within_split": within_counts[group_key] > 1,
            "exact_source_overlap_with_qalb_test": qalb_overlap,
            "exact_source_overlap_with_nahw": nahw_overlap,
            "eligible_for_training": train_ok,
            "eligible_for_development": dev_ok,
            "selection_reason": reasons,
        })

    metadata = {
        "archive_sha256": sha256_path(archive_path),
        "archive_filename": archive_path.name,
        "nahw_sha256": sha256_path(nahw_path),
        "nahw_filename": nahw_path.name,
        "member_sha256": dict(sorted(member_hashes.items())),
    }
    return records, metadata
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
python -m unittest tests.test_prepare_qalb_manifests.QalbManifestTests.test_preserves_within_train_duplicates_and_applies_leakage_policy -v
```

Expected: one passing test.

- [ ] **Step 5: Commit the first vertical slice**

```powershell
git add scripts/prepare_qalb_manifests.py tests/test_prepare_qalb_manifests.py
git commit -m "Add QALB manifest selection core"
```

### Task 2: Prove malformed or unsafe inputs fail before output

**Files:**
- Modify: `tests/test_prepare_qalb_manifests.py`
- Modify: `scripts/prepare_qalb_manifests.py`

- [ ] **Step 1: Add the shared fixture-rebuild helper**

Add this method to `QalbManifestTests`:

```python
    def rebuild_with_groups(self, groups, extra_members=None):
        self.archive.unlink()
        write_fixture_archive(self.archive, groups=groups, extra_members=extra_members)
```

- [ ] **Step 2: Write and run the failing parallel-count test**

Add:

```python
    def test_rejects_parallel_count_mismatch(self):
        self.archive.unlink()
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(f"{ROOT_NAME}/README.txt", b"readme")
            archive.writestr(f"{ROOT_NAME}/LICENSE.txt", b"license")
            for (year, track, split), rows in GROUPS.items():
                sent, cor, m2 = group_members(rows)
                stem = member_stem(year, track, split)
                archive.writestr(f"{stem}.sent", sent)
                if (year, track, split) == (2015, "L2", "dev"):
                    cor += "S EXTRA\n".encode("utf-8")
                archive.writestr(f"{stem}.cor", cor)
                archive.writestr(f"{stem}.m2", m2)
        with self.assertRaisesRegex(ValueError, "Parallel record count mismatch"):
            build_manifest_data(self.archive, self.nahw)
```

Run:

```powershell
python -m unittest tests.test_prepare_qalb_manifests.QalbManifestTests.test_rejects_parallel_count_mismatch -v
```

Expected RED: failure stating that `ManifestError` was not raised.

- [ ] **Step 3: Add the parallel-count guard and verify GREEN**

Immediately after parsing `sent_rows`, `corrections`, and `m2_sources`, add:

```python
            if not (len(sent_rows) == len(corrections) == len(m2_sources)):
                raise ManifestError(f"Parallel record count mismatch for {spec.stem}")
```

Rerun the focused test. Expected: pass.

- [ ] **Step 4: Write and run the failing duplicate-ID test**

Add:

```python
    def test_rejects_duplicate_document_ids(self):
        groups = dict(GROUPS)
        groups[(2015, "L2", "train")] = [
            ("same.ar", "ONE", "ONE_FIXED"),
            ("same.ar", "TWO", "TWO_FIXED"),
        ]
        self.rebuild_with_groups(groups)
        with self.assertRaisesRegex(ValueError, "Duplicate document ID"):
            build_manifest_data(self.archive, self.nahw)
```

Run the single test. Expected RED: `ManifestError` was not raised.

- [ ] **Step 5: Add the duplicate-ID guard and verify GREEN**

Add after the count guard:

```python
            document_ids = [document_id for document_id, _ in sent_rows]
            if len(document_ids) != len(set(document_ids)):
                raise ManifestError(f"Duplicate document ID in {spec.stem}")
```

Rerun the single test. Expected: pass.

- [ ] **Step 6: Write and run the failing M2-alignment test**

Add:

```python
    def test_rejects_m2_source_mismatch(self):
        self.archive.unlink()
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(f"{ROOT_NAME}/README.txt", b"readme")
            archive.writestr(f"{ROOT_NAME}/LICENSE.txt", b"license")
            for (year, track, split), rows in GROUPS.items():
                sent, cor, m2 = group_members(rows)
                stem = member_stem(year, track, split)
                archive.writestr(f"{stem}.sent", sent)
                archive.writestr(f"{stem}.cor", cor)
                if (year, track, split) == (2014, "L1", "dev"):
                    m2 = m2.replace(b"DEV_KEEP", b"DEV_WRONG", 1)
                archive.writestr(f"{stem}.m2", m2)
        with self.assertRaisesRegex(ValueError, "source order mismatch"):
            build_manifest_data(self.archive, self.nahw)
```

Run the single test. Expected RED: `ManifestError` was not raised.

- [ ] **Step 7: Add the exact M2-alignment guard and verify GREEN**

Add after the duplicate-ID guard:

```python
            sources = [source for _, source in sent_rows]
            if sources != m2_sources:
                raise ManifestError(f".sent and .m2 source order mismatch for {spec.stem}")
```

Rerun the single test. Expected: pass.

- [ ] **Step 8: Write and run the failing invalid-UTF-8 test**

Add:

```python
    def test_rejects_invalid_utf8_with_member_name(self):
        self.archive.unlink()
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(f"{ROOT_NAME}/README.txt", b"readme")
            archive.writestr(f"{ROOT_NAME}/LICENSE.txt", b"license")
            for (year, track, split), rows in GROUPS.items():
                sent, cor, m2 = group_members(rows)
                stem = member_stem(year, track, split)
                archive.writestr(f"{stem}.sent", b"\xff" if (year, track, split) == (2015, "L2", "train") else sent)
                archive.writestr(f"{stem}.cor", cor)
                archive.writestr(f"{stem}.m2", m2)
        with self.assertRaisesRegex(ValueError, "not valid UTF-8: .*L2-Train.sent"):
            build_manifest_data(self.archive, self.nahw)
```

Run the single test. Expected RED: a raw codec message that does not contain the member name.

- [ ] **Step 9: Wrap UTF-8 failures and verify GREEN**

Replace `decode_member` with:

```python
def decode_member(payload: bytes, member: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ManifestError(f"Archive member is not valid UTF-8: {member}") from error
```

Rerun the single test. Expected: pass.

- [ ] **Step 10: Write and run the failing unsafe-member test**

Add:

```python
    def test_rejects_unsafe_zip_member_even_when_unused(self):
        self.rebuild_with_groups(GROUPS, extra_members=[("../escape.txt", b"unsafe")])
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP member path"):
            build_manifest_data(self.archive, self.nahw)
```

Run the single test. Expected RED: `ManifestError` was not raised.

- [ ] **Step 11: Validate every ZIP member and verify GREEN**

Add:

```python
def validate_archive_members(archive: zipfile.ZipFile) -> None:
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        unsafe = (
            info.filename.startswith("/")
            or "\\" in info.filename
            or ".." in path.parts
            or (len(info.filename) > 1 and info.filename[1] == ":")
        )
        if unsafe:
            raise ManifestError(f"Unsafe ZIP member path: {info.filename}")
        if info.flag_bits & 0x1:
            raise ManifestError(f"Encrypted ZIP member is not supported: {info.filename}")
```

Call `validate_archive_members(archive)` as the first statement inside the archive-open `with` block. Rerun the unsafe-member test, then run:

```powershell
python -m unittest tests.test_prepare_qalb_manifests -v
```

Expected: six passing tests.

- [ ] **Step 12: Commit validation coverage**

```powershell
git add tests/test_prepare_qalb_manifests.py scripts/prepare_qalb_manifests.py
git commit -m "Test QALB manifest input safeguards"
```

### Task 3: Add deterministic text-free output and the CLI

**Files:**
- Modify: `tests/test_prepare_qalb_manifests.py`
- Modify: `scripts/prepare_qalb_manifests.py`

- [ ] **Step 1: Add failing output and CLI tests**

Extend imports in the test file:

```python
import subprocess
import sys

from scripts.prepare_qalb_manifests import (
    PUBLIC_RECORD_KEYS,
    build_manifest_data,
    write_manifests,
)
```

Add these methods:

```python
    def test_writes_deterministic_text_free_lf_outputs(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)
        output_dir = self.root / "out"
        first_summary = write_manifests(registry, metadata, output_dir)
        first_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
        second_summary = write_manifests(registry, metadata, output_dir)
        second_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}

        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(set(first_bytes), {
            "qalb_registry.jsonl",
            "qalb_train_selection.jsonl",
            "qalb_dev_selection.jsonl",
            "qalb_manifest_summary.json",
        })
        for payload in first_bytes.values():
            self.assertNotIn(b"\r\n", payload)
            for corpus_value in (b"TRAIN_KEEP", b"TRAIN_FIXED", b"QALB_TEST_MATCH", b"NAHW_MATCH"):
                self.assertNotIn(corpus_value, payload)

        registry_rows = [json.loads(line) for line in first_bytes["qalb_registry.jsonl"].decode("utf-8").splitlines()]
        forbidden = {"source", "correction", "annotation", "prompt", "completion", "passage"}
        self.assertTrue(all(not (forbidden & set(row)) for row in registry_rows))
        self.assertEqual(first_summary["counts"]["registry"], 11)
        self.assertEqual(first_summary["counts"]["train_selected"], 4)
        self.assertEqual(first_summary["counts"]["dev_selected"], 2)

    def test_cli_accepts_explicit_private_paths(self):
        output_dir = self.root / "cli-out"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_qalb_manifests.py",
                "--archive",
                str(self.archive),
                "--nahw-passage",
                str(self.nahw),
                "--output-dir",
                str(output_dir),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("Registry records: 11", result.stdout)
        self.assertIn("Training selected: 4", result.stdout)
        self.assertIn("Development selected: 2", result.stdout)

    def test_cli_validation_failure_creates_no_output(self):
        self.rebuild_with_groups(GROUPS, extra_members=[("../escape.txt", b"unsafe")])
        output_dir = self.root / "failed-output"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_qalb_manifests.py",
                "--archive",
                str(self.archive),
                "--nahw-passage",
                str(self.nahw),
                "--output-dir",
                str(output_dir),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe ZIP member path", result.stderr)
        self.assertFalse(output_dir.exists())
```

The expected synthetic training count is four: `t1`, both preserved duplicate rows, and `l2t`. The synthetic development count is two: `d1` and `l2d`.

- [ ] **Step 2: Run the new output test and verify RED**

```powershell
python -m unittest tests.test_prepare_qalb_manifests.QalbManifestTests.test_writes_deterministic_text_free_lf_outputs -v
```

Expected: import failure because `write_manifests` does not exist.

- [ ] **Step 3: Implement deterministic rendering, summary hashes, atomic replacement, and CLI defaults**

Add these imports and constants to `scripts/prepare_qalb_manifests.py`:

```python
import argparse
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "qalb" / "QALB-0.9.1-Dec03-2021-SharedTasks.zip"
DEFAULT_NAHW = ROOT / "data" / "raw" / "nahw" / "Nahw-Passage.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "qalb"
```

Add these functions:

```python
def render_jsonl(rows) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def write_manifests(registry, metadata, output_dir: Path):
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
            "within_split_duplicate_records_flagged": sum(row["duplicate_source_within_split"] for row in registry),
            "train_dev_qalb_test_overlap_excluded": sum(
                row["split"] in {"train", "dev"} and row["exact_source_overlap_with_qalb_test"]
                for row in registry
            ),
            "train_dev_nahw_overlap_excluded": sum(
                row["split"] in {"train", "dev"} and row["exact_source_overlap_with_nahw"]
                for row in registry
            ),
        },
        "selection_policy": {
            "preserve_within_split_duplicates": True,
            "exclude_exact_qalb_test_overlap": True,
            "exclude_exact_nahw_overlap": True,
            "normalization": "none; exact UTF-8 strings after file-format prefix removal",
            "qalb_test_role": "evaluation-only",
        },
        "output_sha256": {name: sha256_bytes(payload) for name, payload in sorted(payloads.items())},
    }
    summary_payload = (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    for name, payload in payloads.items():
        atomic_write(output_dir / name, payload)
    atomic_write(output_dir / "qalb_manifest_summary.json", summary_payload)
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
    print("IMPORTANT: QALB test records are evaluation-only; never commit generated manifests or corpus data.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the complete focused suite and verify GREEN**

```powershell
python -m unittest tests.test_prepare_qalb_manifests -v
```

Expected: nine passing tests.

- [ ] **Step 5: Run the entire repository suite**

```powershell
python -m unittest discover -s tests -v
python -m compileall scripts
```

Expected: all prior tests plus eight new tests pass; compilation exits 0.

- [ ] **Step 6: Commit the CLI**

```powershell
git add scripts/prepare_qalb_manifests.py tests/test_prepare_qalb_manifests.py
git commit -m "Add deterministic QALB manifest CLI"
```

### Task 4: Run the CLI against the supplied QALB archive

**Files:**
- Generate, do not commit: `data/processed/qalb/qalb_registry.jsonl`
- Generate, do not commit: `data/processed/qalb/qalb_train_selection.jsonl`
- Generate, do not commit: `data/processed/qalb/qalb_dev_selection.jsonl`
- Generate, do not commit: `data/processed/qalb/qalb_manifest_summary.json`

- [ ] **Step 1: Run the real private manifest generation**

```powershell
python scripts/prepare_qalb_manifests.py
```

Expected:

```text
Registry records: 22938
Training selected: 19720
Development selected: 1171
```

- [ ] **Step 2: Verify schema, counts, selection policy, and lack of corpus-text fields**

Run:

```powershell
@'
import json
from pathlib import Path

root = Path('data/processed/qalb')
summary = json.loads((root / 'qalb_manifest_summary.json').read_text(encoding='utf-8'))
registry = [json.loads(line) for line in (root / 'qalb_registry.jsonl').read_text(encoding='utf-8').splitlines()]
train = [json.loads(line) for line in (root / 'qalb_train_selection.jsonl').read_text(encoding='utf-8').splitlines()]
dev = [json.loads(line) for line in (root / 'qalb_dev_selection.jsonl').read_text(encoding='utf-8').splitlines()]
forbidden = {'source', 'correction', 'annotation', 'prompt', 'completion', 'passage'}
assert len(registry) == summary['counts']['registry'] == 22938
assert len(train) == summary['counts']['train_selected'] == 19720
assert len(dev) == summary['counts']['dev_selected'] == 1171
assert all(not (forbidden & set(row)) for row in registry + train + dev)
assert all(row['split'] == 'train' and row['eligible_for_training'] for row in train)
assert all(row['split'] == 'dev' and row['eligible_for_development'] for row in dev)
assert all(not row['exact_source_overlap_with_qalb_test'] for row in train + dev)
assert all(not row['exact_source_overlap_with_nahw'] for row in train + dev)
assert summary['counts']['train_dev_qalb_test_overlap_excluded'] == 1
assert summary['counts']['train_dev_nahw_overlap_excluded'] == 0
print('real_manifest_validation=PASS')
print(json.dumps(summary['counts'], indent=2, sort_keys=True))
print(json.dumps(summary['output_sha256'], indent=2, sort_keys=True))
'@ | python -
```

Expected: `real_manifest_validation=PASS`, followed by counts and three JSONL hashes.

- [ ] **Step 3: Verify deterministic rerun bytes**

```powershell
$before = Get-FileHash data\processed\qalb\* -Algorithm SHA256 | Sort-Object Path | Select-Object Path, Hash
python scripts/prepare_qalb_manifests.py
$after = Get-FileHash data\processed\qalb\* -Algorithm SHA256 | Sort-Object Path | Select-Object Path, Hash
if (Compare-Object $before $after -Property Path, Hash) { throw 'Manifest rerun changed bytes' }
Write-Output 'deterministic_rerun=PASS'
```

Expected: `deterministic_rerun=PASS`.

- [ ] **Step 4: Verify private outputs are ignored and not staged**

```powershell
git check-ignore data/processed/qalb/qalb_registry.jsonl data/processed/qalb/qalb_train_selection.jsonl data/processed/qalb/qalb_dev_selection.jsonl data/processed/qalb/qalb_manifest_summary.json
git status --short
```

Expected: all four paths are printed by `git check-ignore`; none appears in `git status --short`.

### Task 5: Document the command and verified results

**Files:**
- Modify: `README.md`
- Modify: `docs/dataset_audit.md`
- Modify: `results/qalb_0.9.1_intake.md`

- [ ] **Step 1: Add the private manifest command to README**

Add a concise `QALB private manifests` subsection after the data-preparation instructions:

```markdown
### QALB private manifests

After obtaining the registered QALB 0.9.1 archive, place it at the Git-ignored path `data/raw/qalb/QALB-0.9.1-Dec03-2021-SharedTasks.zip`, then run:

```bash
python scripts/prepare_qalb_manifests.py
```

The script reads the unchanged ZIP directly and writes text-free private registry and train/dev selection manifests under `data/processed/qalb/`. It preserves within-split duplicates, excludes exact QALB-test and Nahw overlaps from train/dev selections, and never emits corpus sentences or corrections. QALB test splits remain evaluation-only. Do not commit or redistribute QALB data or generated private manifests.
```

- [ ] **Step 2: Record actual counts and hashes in the dataset audit**

Append a `Private selection manifests` paragraph to the QALB 0.9.1 section of `docs/dataset_audit.md` using the exact values from `qalb_manifest_summary.json`. Include:

- the command;
- registry, train-selected, and dev-selected counts;
- QALB-test and Nahw exclusion counts;
- preservation of within-split duplicates;
- the three JSONL SHA-256 values;
- the fact that outputs contain metadata/hashes only and remain Git-ignored.

Do not copy any QALB sentence, correction, or M2 annotation into the audit.

- [ ] **Step 3: Add execution evidence to the intake report**

Append a `Manifest generation` section to `results/qalb_0.9.1_intake.md` with the same verified counts and hashes, the script path, code commit, deterministic-rerun result, and statement that no model training or QALB test evaluation occurred.

- [ ] **Step 4: Verify documentation contains no corpus or secret files**

```powershell
git diff --check
git status --short
git diff --name-only
```

Expected tracked changes: `README.md`, `docs/dataset_audit.md`, and `results/qalb_0.9.1_intake.md`. No file under `data/` or `outputs/` may appear.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md docs/dataset_audit.md results/qalb_0.9.1_intake.md
git commit -m "Document private QALB manifest workflow"
```

### Task 6: Final verification, Research Hub update, and publication

**Files:**
- Verify: all tracked implementation and documentation files
- Update externally: Musahhih Research Hub task `Prepare private QALB train/dev manifests`

- [ ] **Step 1: Run fresh full verification**

```powershell
python -m compileall scripts
python -m unittest discover -s tests -v
git diff --check
git status --short
```

Expected: compilation exits 0, every test passes, no diff errors, and the worktree is clean after commits.

- [ ] **Step 2: Revalidate real private artifacts from their recorded summary**

Run the schema/count verification command from Task 4 Step 2 again. Expected: `real_manifest_validation=PASS`.

- [ ] **Step 3: Update Notion after searching and fetching the existing task**

Set the task to `Done` only after all local verification passes. Record:

- implementation commit;
- archive and Nahw input hashes;
- registry/train/dev counts;
- excluded overlap counts;
- generated JSONL hashes;
- deterministic-rerun result;
- links to `scripts/prepare_qalb_manifests.py`, `docs/dataset_audit.md`, and `results/qalb_0.9.1_intake.md`;
- reminder that outputs are private, Git-ignored, and contain no corpus text.

Update the Hub current milestone to state that QALB selection manifests are ready and that no fine-tuning has started.

- [ ] **Step 4: Push commits and verify the remote head**

```powershell
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Expected: local and remote `main` hashes match.

- [ ] **Step 5: Report the next research gate**

The handoff must state that the next task is freezing B1/B2 prompt protocols using literature or the eligible QALB development selection, never Nahw-Passage. Fine-tuning remains blocked until that protocol is fixed and the team has institutional guidance for any persistent transformed QALB corpus copies.
