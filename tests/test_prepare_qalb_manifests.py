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

    def test_rejects_parallel_record_count_mismatch(self):
        key = (2015, "L2", "dev")
        stem = member_stem(*key)
        _, cor, _ = group_members(GROUPS[key])
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[(f"{stem}.cor", cor + b"S EXTRA\n")],
        )

        with self.assertRaisesRegex(ValueError, "Parallel record count mismatch"):
            build_manifest_data(self.archive, self.nahw)

    def test_rejects_duplicate_document_ids_within_split(self):
        groups = dict(GROUPS)
        groups[(2015, "L2", "train")] = [
            ("same.ar", "FIRST_SOURCE", "FIRST_CORRECTION"),
            ("same.ar", "SECOND_SOURCE", "SECOND_CORRECTION"),
        ]
        self.rebuild_with_groups(groups)

        with self.assertRaisesRegex(ValueError, "Duplicate document ID"):
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

        with self.assertRaisesRegex(ValueError, "source order mismatch"):
            build_manifest_data(self.archive, self.nahw)

    def test_reports_invalid_utf8_archive_member(self):
        stem = member_stem(2015, "L2", "train")
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[(f"{stem}.sent", b"\xff")],
        )

        with self.assertRaisesRegex(ValueError, r"not valid UTF-8: .*L2-Train\.sent"):
            build_manifest_data(self.archive, self.nahw)

    def test_rejects_unsafe_zip_member_path(self):
        self.rebuild_with_groups(
            GROUPS,
            extra_members=[("../escape.txt", b"unused")],
        )

        with self.assertRaisesRegex(ValueError, "Unsafe ZIP member path"):
            build_manifest_data(self.archive, self.nahw)


if __name__ == "__main__":
    unittest.main()
