import hashlib
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

        self.assertTrue(
            all(not row["eligible_for_training"] for row in registry if row["split"] == "test")
        )
        self.assertTrue(
            all(not row["eligible_for_development"] for row in registry if row["split"] == "test")
        )

    def test_writes_deterministic_text_free_manifests(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)
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
        registry, metadata = build_manifest_data(self.archive, self.nahw)
        private_sentinel = "PRIVATE_SOURCE_SENTINEL"
        invalid_registry = [dict(row) for row in registry]
        invalid_registry[0]["source"] = private_sentinel
        output_dir = self.root / "invalid-registry-output"

        with self.assertRaisesRegex(ManifestError, "registry schema") as caught:
            write_manifests(invalid_registry, metadata, output_dir)

        self.assertFalse(output_dir.exists())
        self.assertNotIn(private_sentinel, str(caught.exception))

    def test_rejects_invalid_metadata_before_creating_output(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)
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

    def test_failed_jsonl_replacement_invalidates_and_can_recover_generation(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)
        output_dir = self.root / "recoverable-output"
        write_manifests(registry, metadata, output_dir)
        valid_files = {
            path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
        }
        original_atomic_write = manifest_module.atomic_write

        def fail_during_jsonl_replacement(path, payload):
            if Path(path).name == "qalb_train_selection.jsonl":
                raise OSError("injected JSONL replacement failure")
            original_atomic_write(path, payload)

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

    def test_rejects_unrelated_stale_file_without_invalidating_summary(self):
        registry, metadata = build_manifest_data(self.archive, self.nahw)
        output_dir = self.root / "stale-output"
        write_manifests(registry, metadata, output_dir)
        summary_path = output_dir / "qalb_manifest_summary.json"
        original_summary = summary_path.read_bytes()
        (output_dir / "unrelated.txt").write_text("stale", encoding="utf-8")

        with self.assertRaisesRegex(ManifestError, "unexpected output"):
            write_manifests(registry, metadata, output_dir)

        self.assertEqual(summary_path.read_bytes(), original_summary)

    def test_cli_accepts_explicit_private_paths(self):
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
            check=True,
        )

        self.assertIn("Registry records: 11", result.stdout)
        self.assertIn("Training selected: 4", result.stdout)
        self.assertIn("Development selected: 2", result.stdout)

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
            build_manifest_data(self.archive, self.nahw)
        self.assertIn("sent=1, cor=2, m2=1", str(caught.exception))

    def test_rejects_duplicate_document_ids_within_split(self):
        groups = dict(GROUPS)
        groups[(2015, "L2", "train")] = [
            ("same.ar", "FIRST_SOURCE", "FIRST_CORRECTION"),
            ("same.ar", "SECOND_SOURCE", "SECOND_CORRECTION"),
        ]
        self.rebuild_with_groups(groups)

        with self.assertRaisesRegex(ManifestError, "Duplicate document ID"):
            build_manifest_data(self.archive, self.nahw)

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
            build_manifest_data(self.archive, self.nahw)
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
            build_manifest_data(self.archive, self.nahw)

    def test_rejects_unsafe_zip_member_path(self):
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[("../escape.txt", b"unused")],
        )

        with self.assertRaisesRegex(ManifestError, "Unsafe ZIP member path"):
            build_manifest_data(self.archive, self.nahw)

    def test_rejects_duplicate_zip_member_name(self):
        member = f"{ROOT_NAME}/README.txt"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr(member, b"duplicate readme")

        with self.assertRaisesRegex(
            ManifestError, rf"Duplicate ZIP member name: {member}"
        ):
            build_manifest_data(self.archive, self.nahw)

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
