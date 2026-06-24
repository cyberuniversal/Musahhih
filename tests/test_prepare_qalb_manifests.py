import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import scripts.prepare_qalb_manifests as manifest_module
from scripts.prepare_qalb_manifests import (
    ManifestError,
    PUBLIC_RECORD_KEYS,
    build_manifest_data,
    validate_archive_members,
    write_manifests,
)


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


def expected_counts(groups):
    return {key: len(rows) for key, rows in groups.items()}


def member_stem(year, track, split):
    title_split = {"train": "Train", "dev": "Dev", "test": "Test"}[split]
    return f"{ROOT_NAME}/data/{year}/{split}/QALB-{year}-{track}-{title_split}"


def group_members(rows):
    sent = "".join(f"{doc_id} {source}\n" for doc_id, source, _ in rows)
    cor = "".join(f"S {correction}\n" for _, _, correction in rows)
    m2 = "".join(f"S {source}\n\n" for _, source, _ in rows)
    return sent.encode("utf-8-sig"), cor.encode("utf-8-sig"), m2.encode("utf-8-sig")


def write_fixture_archive(path, groups=GROUPS, extra_members=None):
    extra_members = dict(extra_members or [])
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{ROOT_NAME}/README.txt", "fixture readme".encode("utf-8-sig"))
        archive.writestr(f"{ROOT_NAME}/LICENSE.txt", "fixture license".encode("utf-8-sig"))
        for (year, track, split), rows in groups.items():
            sent, cor, m2 = group_members(rows)
            stem = member_stem(year, track, split)
            for suffix, payload in (("sent", sent), ("cor", cor), ("m2", m2)):
                name = f"{stem}.{suffix}"
                archive.writestr(name, extra_members.pop(name, payload))
        for name, payload in extra_members.items():
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

    def rebuild_with_groups(self, groups, extra_members=None):
        self.archive.unlink()
        write_fixture_archive(self.archive, groups, extra_members)

    def build_fixture_manifest(self, groups=GROUPS):
        return build_manifest_data(
            self.archive,
            self.nahw,
            expected_split_counts=expected_counts(groups),
        )

    def assert_rejected_row_tamper(self, field, value, *, row_index=0):
        registry, metadata = self.build_fixture_manifest()
        tampered = [dict(row) for row in registry]
        tampered[row_index][field] = value
        output_dir = self.root / f"invalid-{field}"

        with self.assertRaisesRegex(
            ManifestError, rf"registry row {row_index} field {field}"
        ) as caught:
            write_manifests(tampered, metadata, output_dir)

        self.assertFalse(output_dir.exists())
        self.assertNotIn(str(value), str(caught.exception))

    def recompute_record_key(self, row):
        row["record_key"] = (
            f"qalb-{row['release']}:{row['year']}:{row['track']}:"
            f"{row['split']}:{row['line_number']:06d}:{row['document_id']}"
        )

    def test_rejects_aligned_archive_with_noncanonical_split_count_by_default(self):
        with self.assertRaisesRegex(
            ManifestError,
            r"Split count mismatch: 2014:L1:train observed=4 expected=19411$",
        ) as caught:
            build_manifest_data(self.archive, self.nahw)

        message = str(caught.exception)
        self.assertNotIn("t1.ar", message)
        self.assertNotIn(str(self.archive.resolve()), message)

    def test_records_deterministic_observed_split_counts_in_metadata(self):
        _, metadata = self.build_fixture_manifest()

        expected = {
            "2014:L1:dev": 2,
            "2014:L1:test": 1,
            "2014:L1:train": 4,
            "2015:L1:test": 1,
            "2015:L2:dev": 1,
            "2015:L2:test": 1,
            "2015:L2:train": 1,
        }
        self.assertEqual(metadata["split_counts"], expected)
        self.assertEqual(
            set(metadata),
            {
                "archive_sha256",
                "archive_filename",
                "nahw_sha256",
                "nahw_filename",
                "member_sha256",
                "split_counts",
                "nahw_passage_source_sha256",
            },
        )

    def test_hashes_the_same_input_bytes_used_for_parsing(self):
        expected_archive_hash = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        expected_nahw_hash = hashlib.sha256(self.nahw.read_bytes()).hexdigest()
        open_counts = {self.archive.resolve(): 0, self.nahw.resolve(): 0}
        original_open = io.open

        def counting_open(file, *args, **kwargs):
            if not isinstance(file, int):
                resolved = Path(file).resolve()
                if resolved in open_counts:
                    open_counts[resolved] += 1
            return original_open(file, *args, **kwargs)

        with mock.patch("io.open", side_effect=counting_open):
            _, metadata = self.build_fixture_manifest()

        self.assertEqual(metadata["archive_sha256"], expected_archive_hash)
        self.assertEqual(metadata["nahw_sha256"], expected_nahw_hash)
        self.assertEqual(open_counts[self.archive.resolve()], 1)
        self.assertEqual(open_counts[self.nahw.resolve()], 1)

    def test_preserves_within_train_duplicates_and_applies_leakage_policy(self):
        registry, metadata = self.build_fixture_manifest()

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

        self.assertTrue(
            all(not row["eligible_for_training"] for row in registry if row["split"] == "test")
        )
        self.assertTrue(
            all(not row["eligible_for_development"] for row in registry if row["split"] == "test")
        )

    def test_writes_deterministic_text_free_manifests(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "processed" / "qalb"

        first_summary = write_manifests(registry, metadata, output_dir)
        first_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }
        second_summary = write_manifests(registry, metadata, output_dir)
        second_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }

        expected_files = {
            "qalb_registry.jsonl",
            "qalb_train_selection.jsonl",
            "qalb_dev_selection.jsonl",
            "qalb_manifest_summary.json",
        }
        self.assertEqual(set(first_files), expected_files)
        self.assertEqual(first_files, second_files)
        self.assertEqual(first_summary, second_summary)
        private_values = {
            value
            for rows in GROUPS.values()
            for _, source, correction in rows
            for value in (source, correction)
        }
        private_values.update({"NAHW_MATCH", str(self.root.resolve())})
        for payload in first_files.values():
            self.assertNotIn(b"\r\n", payload)
            for private_value in private_values:
                self.assertNotIn(private_value.encode("utf-8"), payload)

        registry_rows = [
            json.loads(line)
            for line in first_files["qalb_registry.jsonl"].decode("utf-8").splitlines()
        ]
        self.assertEqual(len(registry_rows), 11)
        self.assertTrue(all(set(row) == PUBLIC_RECORD_KEYS for row in registry_rows))
        forbidden_keys = {
            "source",
            "correction",
            "annotation",
            "prompt",
            "completion",
            "passage",
        }
        self.assertTrue(all(forbidden_keys.isdisjoint(row) for row in registry_rows))

        self.assertEqual(first_summary["counts"]["registry"], 11)
        self.assertEqual(first_summary["counts"]["train_selected"], 4)
        self.assertEqual(first_summary["counts"]["dev_selected"], 2)
        for filename in (
            "qalb_registry.jsonl",
            "qalb_train_selection.jsonl",
            "qalb_dev_selection.jsonl",
        ):
            self.assertEqual(
                first_summary["output_sha256"][filename],
                hashlib.sha256(first_files[filename]).hexdigest(),
            )

    def test_rejects_private_registry_fields_before_creating_output(self):
        registry, metadata = self.build_fixture_manifest()
        private_sentinel = "PRIVATE_SOURCE_SENTINEL"
        invalid_registry = [dict(row) for row in registry]
        invalid_registry[0]["source"] = private_sentinel
        output_dir = self.root / "invalid-registry-output"

        with self.assertRaisesRegex(ManifestError, "registry schema") as caught:
            write_manifests(invalid_registry, metadata, output_dir)

        self.assertFalse(output_dir.exists())
        self.assertNotIn(private_sentinel, str(caught.exception))

    def test_rejects_invalid_metadata_before_creating_output(self):
        registry, metadata = self.build_fixture_manifest()
        absolute_path = str(self.archive.resolve())
        cases = {
            "unexpected_absolute_path_field": {
                **metadata,
                "archive_path": absolute_path,
            },
            "absolute_filename": {
                **metadata,
                "archive_filename": absolute_path,
            },
            "drive_relative_filename": {
                **metadata,
                "archive_filename": "C:private.zip",
            },
        }

        for case_name, invalid_metadata in cases.items():
            with self.subTest(case_name=case_name):
                output_dir = self.root / f"invalid-metadata-{case_name}"
                with self.assertRaisesRegex(ManifestError, "metadata") as caught:
                    write_manifests(registry, invalid_metadata, output_dir)
                self.assertFalse(output_dir.exists())
                self.assertNotIn(absolute_path, str(caught.exception))

    def test_writer_rejects_absolute_or_tampered_member_paths(self):
        cases = {
            "absolute": str(self.archive.resolve()),
            "wrong_split": member_stem(2014, "L1", "dev") + ".sent",
        }

        for case_name, value in cases.items():
            with self.subTest(case_name=case_name):
                self.assert_rejected_row_tamper("sent_member", value)

    def test_writer_rejects_official_test_row_marked_training_eligible(self):
        registry, _ = self.build_fixture_manifest()
        test_index = next(
            index for index, row in enumerate(registry) if row["split"] == "test"
        )

        self.assert_rejected_row_tamper(
            "eligible_for_training", True, row_index=test_index
        )

    def test_writer_rejects_inconsistent_derived_registry_fields(self):
        registry, _ = self.build_fixture_manifest()
        duplicate_index = next(
            index
            for index, row in enumerate(registry)
            if row["duplicate_source_within_split"]
        )
        cases = (
            ("record_key", "TAMPERED_RECORD_KEY", 0),
            ("selection_reason", ["TAMPERED_REASON"], 0),
            ("duplicate_source_within_split", False, duplicate_index),
        )

        for field, value, row_index in cases:
            with self.subTest(field=field):
                self.assert_rejected_row_tamper(
                    field, value, row_index=row_index
                )

    def test_writer_rejects_duplicate_document_id_with_recomputed_key(self):
        registry, metadata = self.build_fixture_manifest()
        tampered = [dict(row) for row in registry]
        tampered[1]["document_id"] = tampered[0]["document_id"]
        self.recompute_record_key(tampered[1])

        with self.assertRaisesRegex(
            ManifestError, r"registry group 2014:L1:train at row 1$"
        ) as caught:
            write_manifests(tampered, metadata, self.root / "duplicate-document")

        self.assertNotIn(tampered[0]["document_id"], str(caught.exception))

    def test_writer_rejects_noncontiguous_lines_with_recomputed_keys(self):
        registry, metadata = self.build_fixture_manifest()
        for case_name, line_number in (("duplicate", 1), ("skipped", 3)):
            with self.subTest(case_name=case_name):
                tampered = [dict(row) for row in registry]
                tampered[1]["line_number"] = line_number
                self.recompute_record_key(tampered[1])

                with self.assertRaisesRegex(
                    ManifestError, r"registry group 2014:L1:train at row 1$"
                ):
                    write_manifests(
                        tampered,
                        metadata,
                        self.root / f"noncontiguous-{case_name}",
                    )

    def test_writer_rejects_out_of_order_or_reappearing_split_group(self):
        registry, metadata = self.build_fixture_manifest()
        tampered = [dict(row) for row in registry]
        tampered[3], tampered[4] = tampered[4], tampered[3]

        with self.assertRaisesRegex(
            ManifestError, r"registry group 2014:L1:train at row 3$"
        ):
            write_manifests(tampered, metadata, self.root / "out-of-order")

    def test_writer_rejects_malformed_hashes_and_field_types(self):
        cases = (
            ("source_sha256", "A" * 64),
            ("correction_sha256", "not-a-hash"),
            ("line_number", "1"),
            ("source_codepoints", -1),
            ("exact_source_overlap_with_nahw", 1),
            ("selection_reason", "official_train_split"),
        )

        for field, value in cases:
            with self.subTest(field=field):
                self.assert_rejected_row_tamper(field, value)

    def test_writer_rejects_inconsistent_metadata_invariants(self):
        registry, metadata = self.build_fixture_manifest()
        cases = {}

        missing_member = dict(metadata)
        missing_member["member_sha256"] = dict(metadata["member_sha256"])
        missing_member["member_sha256"].pop(next(iter(missing_member["member_sha256"])))
        cases["member_sha256"] = missing_member

        wrong_counts = dict(metadata)
        wrong_counts["split_counts"] = dict(metadata["split_counts"])
        wrong_counts["split_counts"]["2014:L1:train"] += 1
        cases["split_counts"] = wrong_counts

        unsorted_nahw = dict(metadata)
        unsorted_nahw["nahw_passage_source_sha256"] = [
            "f" * 64,
            "0" * 64,
        ]
        cases["nahw_passage_source_sha256"] = unsorted_nahw

        for field, invalid_metadata in cases.items():
            with self.subTest(field=field):
                output_dir = self.root / f"invalid-metadata-{field}"
                with self.assertRaisesRegex(
                    ManifestError, rf"metadata field {field}"
                ):
                    write_manifests(registry, invalid_metadata, output_dir)
                self.assertFalse(output_dir.exists())

    def test_failed_jsonl_replacement_invalidates_and_can_recover_generation(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "recoverable-output"
        write_manifests(registry, metadata, output_dir)
        valid_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }
        original_atomic_write = manifest_module.atomic_write

        def fail_during_jsonl_replacement(output_dir, name, payload):
            if name == "qalb_train_selection.jsonl":
                raise OSError("injected JSONL replacement failure")
            original_atomic_write(output_dir, name, payload)

        with mock.patch.object(
            manifest_module,
            "atomic_write",
            side_effect=fail_during_jsonl_replacement,
        ):
            with self.assertRaisesRegex(OSError, "injected JSONL"):
                write_manifests(registry, metadata, output_dir)

        self.assertFalse((output_dir / "qalb_manifest_summary.json").exists())
        write_manifests(registry, metadata, output_dir)
        recovered_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }
        self.assertEqual(recovered_files, valid_files)

    def test_rerun_removes_reserved_generator_orphan_temp_file(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "orphan-temp-output"
        write_manifests(registry, metadata, output_dir)
        valid_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }
        reserved_dir = output_dir / ".qalb_manifest_tmp"
        reserved_dir.mkdir()
        orphan = reserved_dir / "qalb_registry.jsonl.tmp"
        orphan.write_bytes(b"PRIVATE_ORPHAN_BYTES")

        summary = write_manifests(registry, metadata, output_dir)
        recovered_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }

        self.assertFalse(orphan.exists())
        self.assertEqual(recovered_files, valid_files)
        self.assertEqual(summary["counts"]["registry"], len(registry))

    def test_rejects_and_preserves_unexpected_reserved_temp_child(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "orphan-directory-output"
        write_manifests(registry, metadata, output_dir)
        reserved_dir = output_dir / ".qalb_manifest_tmp"
        reserved_dir.mkdir()
        unexpected = reserved_dir / "unexpected.backup"
        unexpected.write_bytes(b"PRESERVE_ME")

        with self.assertRaisesRegex(ManifestError, "reserved temporary directory"):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(unexpected.read_bytes(), b"PRESERVE_ME")

    def test_rejects_and_preserves_similar_root_backup_file(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "root-backup-output"
        write_manifests(registry, metadata, output_dir)
        backup = output_dir / ".qalb_registry.jsonl.backup"
        backup.write_bytes(b"PRESERVE_ROOT_BACKUP")

        with self.assertRaisesRegex(ManifestError, "unexpected output"):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(backup.read_bytes(), b"PRESERVE_ROOT_BACKUP")

    def test_rejects_reserved_temp_path_when_it_is_not_a_directory(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "reserved-file-output"
        write_manifests(registry, metadata, output_dir)
        reserved_path = output_dir / ".qalb_manifest_tmp"
        reserved_path.write_bytes(b"PRESERVE_RESERVED_FILE")

        with self.assertRaisesRegex(ManifestError, "reserved temporary directory"):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(reserved_path.read_bytes(), b"PRESERVE_RESERVED_FILE")

    def test_atomic_writes_use_exact_reserved_temp_filenames(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "exact-temp-output"
        replacements = []
        original_replace = manifest_module.os.replace

        def record_replace(source, destination):
            replacements.append((Path(source), Path(destination)))
            original_replace(source, destination)

        with mock.patch.object(
            manifest_module.os, "replace", side_effect=record_replace
        ):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(
            {source.name for source, _ in replacements},
            {f"{name}.tmp" for name in manifest_module.EXPECTED_OUTPUT_FILENAMES},
        )
        self.assertTrue(
            all(
                source.parent == output_dir / ".qalb_manifest_tmp"
                and destination.parent == output_dir
                and source.name == f"{destination.name}.tmp"
                for source, destination in replacements
            )
        )
        self.assertFalse((output_dir / ".qalb_manifest_tmp").exists())

    def test_rejects_unrelated_stale_file_without_invalidating_summary(self):
        registry, metadata = self.build_fixture_manifest()
        output_dir = self.root / "stale-output"
        write_manifests(registry, metadata, output_dir)
        summary_path = output_dir / "qalb_manifest_summary.json"
        original_summary = summary_path.read_bytes()
        (output_dir / "unrelated.txt").write_text("stale", encoding="utf-8")

        with self.assertRaisesRegex(ManifestError, "unexpected output"):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(summary_path.read_bytes(), original_summary)

    def test_cli_enforces_canonical_counts_for_explicit_private_paths(self):
        output_dir = self.root / "cli-output"

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
        self.assertIn(
            "Split count mismatch: 2014:L1:train observed=4 expected=19411",
            result.stderr,
        )
        self.assertFalse(output_dir.exists())

    def test_cli_validation_failure_creates_no_output(self):
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[("../escape.txt", b"unsafe")],
        )
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

    def test_rejects_parallel_record_count_mismatch(self):
        key = (2015, "L2", "dev")
        stem = member_stem(*key)
        _, cor, _ = group_members(GROUPS[key])
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[(f"{stem}.cor", cor + b"S EXTRA\n")],
        )

        with self.assertRaisesRegex(
            ManifestError, "Parallel record count mismatch"
        ) as caught:
            self.build_fixture_manifest()
        self.assertIn("sent=1, cor=2, m2=1", str(caught.exception))

    def test_rejects_duplicate_document_ids_within_split(self):
        groups = dict(GROUPS)
        groups[(2015, "L2", "train")] = [
            ("same.ar", "FIRST_SOURCE", "FIRST_CORRECTION"),
            ("same.ar", "SECOND_SOURCE", "SECOND_CORRECTION"),
        ]
        self.rebuild_with_groups(groups)

        with self.assertRaisesRegex(ManifestError, "Duplicate document ID"):
            self.build_fixture_manifest(groups)

    def test_rejects_m2_source_order_mismatch(self):
        key = (2014, "L1", "dev")
        stem = member_stem(*key)
        _, _, m2 = group_members(GROUPS[key])
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[
                (f"{stem}.m2", m2.replace(b"DEV_KEEP", b"DEV_WRONG"))
            ],
        )

        with self.assertRaisesRegex(
            ManifestError, "source order mismatch"
        ) as caught:
            self.build_fixture_manifest()
        self.assertIn("line 1", str(caught.exception))

    def test_reports_invalid_utf8_archive_member(self):
        stem = member_stem(2015, "L2", "train")
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[(f"{stem}.sent", b"\xff")],
        )

        with self.assertRaisesRegex(
            ManifestError, r"not valid UTF-8: .*L2-Train\.sent"
        ):
            self.build_fixture_manifest()

    def test_rejects_unsafe_zip_member_path(self):
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[("../escape.txt", b"unused")],
        )

        with self.assertRaisesRegex(ManifestError, "Unsafe ZIP member path"):
            self.build_fixture_manifest()

    def test_rejects_duplicate_zip_member_name(self):
        member = f"{ROOT_NAME}/README.txt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr(member, b"duplicate readme")

        with self.assertRaisesRegex(
            ManifestError, rf"Duplicate ZIP member name: {member}"
        ):
            self.build_fixture_manifest()

    def test_rejects_encrypted_zip_member(self):
        info = zipfile.ZipInfo("encrypted.txt")
        info.flag_bits |= 0x1

        class Archive:
            def infolist(self):
                return [info]

        with self.assertRaisesRegex(ManifestError, "Encrypted ZIP member"):
            validate_archive_members(Archive())


if __name__ == "__main__":
    unittest.main()
