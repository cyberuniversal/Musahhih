import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_prompt_baseline import (
    RunConfig,
    RunSafetyError,
    assert_final_eval_allowed,
    build_summary,
    experiment_id,
    prepare_run_directory,
    sha256_file,
    validate_experiment_id,
)


class PromptBaselineRunTests(unittest.TestCase):
    def test_experiment_id_uses_canonical_pattern(self):
        run_id = experiment_id("B1-P1", "gemma3-4b-it", "qalb14-dev", 3407, 1)
        self.assertEqual(run_id, "B1-P1__gemma3-4b-it__qalb14-dev__s3407__r01")
        self.assertEqual(validate_experiment_id(run_id), run_id)
        with self.assertRaisesRegex(RunSafetyError, "Invalid experiment ID"):
            validate_experiment_id("B1-P1__Gemma__nahw-passage__s3407__r1")

    def test_prepare_run_directory_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_id = experiment_id("B2-P1", "gemma3-4b-it", "qalb14-dev", 3407, 1)
            created = prepare_run_directory(root, run_id)
            self.assertEqual(created, root / run_id)
            self.assertTrue(created.is_dir())
            with self.assertRaisesRegex(RunSafetyError, "already exists"):
                prepare_run_directory(root, run_id)

    def test_final_nahw_evaluation_requires_explicit_confirmation(self):
        assert_final_eval_allowed("qalb14-dev", confirm_final_eval=False)
        with self.assertRaisesRegex(RunSafetyError, "Nahw-Passage final evaluation"):
            assert_final_eval_allowed("nahw-passage", confirm_final_eval=False)
        assert_final_eval_allowed("nahw-passage", confirm_final_eval=True)

    def test_build_summary_records_hashes_without_private_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / "input.jsonl"
            bundle_file = root / "bundle.json"
            prompt_file = root / "prompt.txt"
            predictions_file = root / "predictions.jsonl"
            input_file.write_text('{"id":"1"}\n', encoding="utf-8")
            bundle_file.write_text('{"field":"licensed corpus payload"}\n', encoding="utf-8")
            prompt_file.write_text("أعد الكلمة فقط\n", encoding="utf-8")
            predictions_file.write_text('{"parsed_correction":"x","exact_match":false}\n', encoding="utf-8")

            config = RunConfig(
                experiment_id="B1-P1__gemma3-4b-it__qalb14-dev__s3407__r01",
                protocol_id="B1-P1",
                model_slug="gemma3-4b-it",
                evaluation_slug="qalb14-dev",
                seed=3407,
                replicate=1,
            )
            summary = build_summary(
                config,
                input_path=input_file,
                prompt_template_path=prompt_file,
                predictions_path=predictions_file,
                bundle_path=bundle_file,
                run_status="planned",
            )

            self.assertEqual(summary["experiment_id"], config.experiment_id)
            self.assertEqual(summary["run_status"], "planned")
            self.assertEqual(summary["input_sha256"], sha256_file(input_file))
            self.assertEqual(summary["bundle_sha256"], sha256_file(bundle_file))
            self.assertEqual(
                summary["prediction_sha256"],
                hashlib.sha256(predictions_file.read_bytes()).hexdigest(),
            )
            serialized = json.dumps(summary, ensure_ascii=False)
            self.assertNotIn("licensed corpus payload", serialized)
            self.assertNotIn("أعد الكلمة فقط", serialized)


if __name__ == "__main__":
    unittest.main()
