import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_b1_prompt_bundle import (
    BundleError,
    PrivateQalbRecord,
    candidate_identity_digest,
    candidate_identity_hash,
    parse_m2_edits,
    select_b1_candidates,
    validate_selection_summary,
    write_private_bundle,
)


def edit(start, end, correction, kind="Edit"):
    return f"A {start} {end}|||{kind}|||{correction}|||REQUIRED|||-NONE-|||0"


class B1PromptBundleTests(unittest.TestCase):
    def make_record(self, key, source, annotation, **overrides):
        defaults = {
            "release": "0.9.1",
            "year": 2014,
            "track": "L1",
            "split": "train",
            "eligible_for_training": True,
        }
        defaults.update(overrides)
        return PrivateQalbRecord(
            record_key=key,
            source=source,
            m2_block=f"S {source}\n{annotation}\n",
            **defaults,
        )

    def test_parse_m2_edits_preserves_correction_text(self):
        edits = parse_m2_edits(
            "S واحد اثنان ثلاثة\n"
            "A 1 2|||Edit|||اثنانَ|||REQUIRED|||-NONE-|||0\n"
            "A 2 3|||noop|||ثلاثة|||REQUIRED|||-NONE-|||0\n"
        )
        self.assertEqual(len(edits), 2)
        self.assertEqual(edits[0].start, 1)
        self.assertEqual(edits[0].end, 2)
        self.assertEqual(edits[0].kind, "Edit")
        self.assertEqual(edits[0].correction, "اثنانَ")

    def test_filters_only_frozen_eligible_single_token_candidates(self):
        records = [
            self.make_record("r-good", "أ ب ج د هـ و", edit(1, 2, "باء")),
            self.make_record(
                "r-dev", "أ ب ج د هـ و", edit(1, 2, "باء"), split="dev"
            ),
            self.make_record(
                "r-ineligible",
                "أ ب ج د هـ و",
                edit(1, 2, "باء"),
                eligible_for_training=False,
            ),
            self.make_record("r-multi-source", "أ ب ج د هـ و", edit(1, 3, "باء")),
            self.make_record("r-empty-correction", "أ ب ج د هـ و", edit(1, 2, "")),
            self.make_record(
                "r-multi-correction", "أ ب ج د هـ و", edit(1, 2, "باء جيم")
            ),
            self.make_record("r-same-correction", "أ ب ج د هـ و", edit(1, 2, "ب")),
            self.make_record("r-repeated-token", "أ ب ج ب هـ و", edit(1, 2, "باء")),
            self.make_record("r-short", "أ ب ج د", edit(1, 2, "باء")),
            self.make_record("r-kind", "أ ب ج د هـ و", edit(1, 2, "باء", "noop")),
        ]

        selected, summary = select_b1_candidates(records, limit=5)

        self.assertEqual([candidate.record_key for candidate in selected], ["r-good"])
        self.assertEqual(summary["candidate_annotations"], 1)
        self.assertEqual(summary["distinct_candidate_records"], 1)

    def test_deterministic_order_uses_identity_digest_and_distinct_records(self):
        records = [
            self.make_record("r-alpha", "أ ب ج د هـ و", edit(1, 2, "باء")),
            self.make_record("r-alpha", "أ ب ج د هـ و", edit(2, 3, "جيم")),
            self.make_record("r-beta", "ح ط ي ك ل م", edit(2, 3, "ياء")),
            self.make_record("r-gamma", "ن س ع ف ص ق", edit(3, 4, "فاء")),
            self.make_record("r-delta", "ر ش ت ث خ ذ", edit(4, 5, "خاء")),
            self.make_record("r-epsilon", "ض ظ غ ء ؤ ئ", edit(0, 1, "ضاد")),
        ]

        selected, summary = select_b1_candidates(records, limit=5)
        identities = [candidate.identity for candidate in selected]
        sorted_all = sorted(
            [
                "r-alpha|1:2",
                "r-alpha|2:3",
                "r-beta|2:3",
                "r-gamma|3:4",
                "r-delta|4:5",
                "r-epsilon|0:1",
            ],
            key=candidate_identity_digest,
        )
        expected = []
        seen_records = set()
        for identity in sorted_all:
            record_key = identity.split("|", 1)[0]
            if record_key not in seen_records:
                expected.append(identity)
                seen_records.add(record_key)
            if len(expected) == 5:
                break

        self.assertEqual(identities, expected)
        self.assertEqual(summary["candidate_annotations"], 6)
        self.assertEqual(summary["distinct_candidate_records"], 5)
        self.assertEqual(
            candidate_identity_hash(identities),
            hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
        )

    def test_validate_selection_summary_fails_closed(self):
        _, summary = select_b1_candidates(
            [self.make_record("r-good", "أ ب ج د هـ و", edit(1, 2, "باء"))],
            limit=1,
        )

        validate_selection_summary(
            summary,
            expected_candidate_annotations=1,
            expected_distinct_records=1,
            expected_identity_sha256=summary["selected_identity_sha256"],
        )
        with self.assertRaisesRegex(BundleError, "candidate annotation count"):
            validate_selection_summary(
                summary,
                expected_candidate_annotations=2,
                expected_distinct_records=1,
                expected_identity_sha256=summary["selected_identity_sha256"],
            )
        with self.assertRaisesRegex(BundleError, "selected identity SHA-256"):
            validate_selection_summary(
                summary,
                expected_candidate_annotations=1,
                expected_distinct_records=1,
                expected_identity_sha256="0" * 64,
            )

    def test_write_private_bundle_refuses_overwrite_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle.json"
            selected, summary = select_b1_candidates(
                [self.make_record("r-good", "أ ب ج د هـ و", edit(1, 2, "باء"))],
                limit=1,
            )

            metadata = write_private_bundle(output, selected, summary)
            payload = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["demonstrations"][0]["record_key"], "r-good")
            self.assertEqual(payload["demonstrations"][0]["error"], "ب")
            self.assertEqual(payload["demonstrations"][0]["correction"], "باء")
            self.assertEqual(metadata["bundle_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            with self.assertRaisesRegex(BundleError, "already exists"):
                write_private_bundle(output, selected, summary)


if __name__ == "__main__":
    unittest.main()
