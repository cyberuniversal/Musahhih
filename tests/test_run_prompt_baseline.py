import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_prompt_baseline import (
    ROOT,
    PromptRecord,
    RunConfig,
    RunSafetyError,
    assert_final_eval_allowed,
    build_summary,
    execute_run,
    experiment_id,
    aggregate_prompt_sha256,
    load_prompt_records,
    load_protocol_demos,
    prepare_run_directory,
    sha256_file,
    validate_private_path,
    validate_experiment_id,
)


class PromptBaselineRunTests(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

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

    def test_load_prompt_records_preserves_valid_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.jsonl"
            self.write_jsonl(
                path,
                [
                    {
                        "record_id": "r1",
                        "passage": "alpha beta",
                        "error": "beta",
                        "gold_correction": "better",
                        "metadata": {"split": "dev"},
                    },
                    {
                        "record_id": "r2",
                        "passage": "gamma delta",
                        "error": "delta",
                    },
                ],
            )

            rows = load_prompt_records(path)

            self.assertEqual(
                rows[0],
                PromptRecord(
                    record_id="r1",
                    passage="alpha beta",
                    error="beta",
                    gold_correction="better",
                    metadata={"split": "dev"},
                ),
            )
            self.assertIsNone(rows[1].gold_correction)
            self.assertEqual(rows[1].metadata, {})

    def test_load_prompt_records_rejects_duplicates_and_invalid_fields(self):
        cases = [
            (
                [
                    {"record_id": "r1", "passage": "a", "error": "b"},
                    {"record_id": "r1", "passage": "c", "error": "d"},
                ],
                "duplicate record_id",
            ),
            ([{"record_id": "", "passage": "a", "error": "b"}], "record_id"),
            ([{"record_id": "r1", "passage": 3, "error": "b"}], "passage"),
            ([{"record_id": "r1", "passage": "a", "error": ""}], "error"),
            (
                [
                    {
                        "record_id": "r1",
                        "passage": "a",
                        "error": "b",
                        "gold_correction": 3,
                    }
                ],
                "gold_correction",
            ),
            (
                [
                    {
                        "record_id": "r1",
                        "passage": "a",
                        "error": "b",
                        "metadata": [],
                    }
                ],
                "metadata",
            ),
        ]
        for rows, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "input.jsonl"
                self.write_jsonl(path, rows)
                with self.assertRaisesRegex(RunSafetyError, message):
                    load_prompt_records(path)

    def test_private_paths_fail_closed_inside_repository(self):
        validate_private_path(Path("/private/tmp/input.jsonl"), label="input")
        validate_private_path(
            ROOT / "data" / "processed" / "qalb" / "input.jsonl",
            label="input",
        )
        validate_private_path(
            ROOT / "outputs" / "private" / "predictions.jsonl",
            label="output",
        )
        with self.assertRaisesRegex(RunSafetyError, "input path"):
            validate_private_path(ROOT / "README.md", label="input")

    def test_protocol_bundle_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle.json"
            bundle.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "demonstrations": [
                            {
                                "source": f"source-{index}",
                                "error": f"error-{index}",
                                "correction": f"fix-{index}",
                            }
                            for index in range(5)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            demos = load_protocol_demos("B1-P1", bundle)

            self.assertEqual(len(demos), 5)
            self.assertEqual(demos[0].passage, "source-0")
            with self.assertRaisesRegex(RunSafetyError, "does not accept"):
                load_protocol_demos("B2-P1", bundle)
            with self.assertRaisesRegex(RunSafetyError, "requires --bundle"):
                load_protocol_demos("B1-P1", None)

    def test_b1_bundle_rejects_wrong_schema_count_and_field_types(self):
        invalid_payloads = [
            {"schema_version": 2, "demonstrations": []},
            {"schema_version": 1, "demonstrations": []},
            {
                "schema_version": 1,
                "demonstrations": [
                    {"source": "s", "error": "e", "correction": "c"}
                    for _ in range(4)
                ]
                + [{"source": "s", "error": 2, "correction": "c"}],
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "bundle.json"
                bundle.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(RunSafetyError):
                    load_protocol_demos("B1-P1", bundle)

    def test_execute_run_captures_private_predictions_and_text_free_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            input_path.write_text('{"private":"payload"}\n', encoding="utf-8")
            prompt_path.write_text("frozen-template", encoding="utf-8")
            records = [
                PromptRecord("r1", "alpha beta", "beta", "fixed", {"split": "dev"}),
                PromptRecord("r2", "gamma delta", "delta", None, {}),
            ]
            config = RunConfig(
                experiment_id="B2-P1__gemma3-4b-it__qalb14-dev__s3407__r01",
                protocol_id="B2-P1",
                model_slug="gemma3-4b-it",
                evaluation_slug="qalb14-dev",
                seed=3407,
                replicate=1,
            )

            summary = execute_run(
                config,
                records,
                [],
                lambda prompt: "**fixed**",
                outputs_root=root / "outputs",
                input_path=input_path,
                prompt_template_path=prompt_path,
                runtime_metadata={"backend": "synthetic"},
                allow_outside_private_output=True,
            )

            run_dir = root / "outputs" / config.experiment_id
            prediction_rows = [
                json.loads(line)
                for line in (run_dir / "predictions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(summary["run_status"], "complete")
            self.assertEqual(summary["counts"]["number_of_records"], 2)
            self.assertEqual(summary["counts"]["number_scored"], 1)
            self.assertEqual(summary["counts"]["number_correct"], 1)
            self.assertEqual(summary["runtime"], {"backend": "synthetic"})
            self.assertEqual(prediction_rows[0]["parsed_correction"], "fixed")
            self.assertTrue(prediction_rows[0]["exact_match"])
            self.assertIsNone(prediction_rows[1]["exact_match"])
            self.assertIn("outer_formatting_removed", prediction_rows[0]["parsing_warnings"])
            self.assertIn("alpha beta", prediction_rows[0]["prompt"])
            self.assertEqual(
                summary["aggregate_prompt_sha256"],
                aggregate_prompt_sha256(
                    [row["prompt_sha256"] for row in prediction_rows]
                ),
            )
            self.assertEqual(
                summary["prediction_sha256"],
                hashlib.sha256((run_dir / "predictions.jsonl").read_bytes()).hexdigest(),
            )
            serialized_summary = json.dumps(summary)
            self.assertNotIn("alpha beta", serialized_summary)
            self.assertNotIn("fixed", serialized_summary)
            self.assertEqual(
                (run_dir / "run.log").read_text(encoding="utf-8"),
                "run completed\n",
            )

            with self.assertRaisesRegex(RunSafetyError, "already exists"):
                execute_run(
                    config,
                    records,
                    [],
                    lambda prompt: "fixed",
                    outputs_root=root / "outputs",
                    input_path=input_path,
                    prompt_template_path=prompt_path,
                    allow_outside_private_output=True,
                )

    def test_execute_run_preserves_invalid_partial_artifacts_without_text_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            input_path.write_text('{"private":"payload"}\n', encoding="utf-8")
            prompt_path.write_text("frozen-template", encoding="utf-8")
            records = [
                PromptRecord("r1", "first private text", "private", "fixed", {}),
                PromptRecord("r2", "second private text", "private", "fixed", {}),
            ]
            config = RunConfig(
                experiment_id="B2-P1__gemma3-4b-it__qalb14-dev__s3407__r02",
                protocol_id="B2-P1",
                model_slug="gemma3-4b-it",
                evaluation_slug="qalb14-dev",
                seed=3407,
                replicate=2,
            )
            calls = 0

            def fail_on_second(prompt):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("second private text must not leak")
                return "fixed"

            with self.assertRaisesRegex(RunSafetyError, "inference execution failed"):
                execute_run(
                    config,
                    records,
                    [],
                    fail_on_second,
                    outputs_root=root / "outputs",
                    input_path=input_path,
                    prompt_template_path=prompt_path,
                    allow_outside_private_output=True,
                )

            run_dir = root / "outputs" / config.experiment_id
            prediction_lines = (run_dir / "predictions.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            summary_text = (run_dir / "summary.json").read_text(encoding="utf-8")
            log_text = (run_dir / "run.log").read_text(encoding="utf-8")
            summary = json.loads(summary_text)
            self.assertEqual(len(prediction_lines), 1)
            self.assertEqual(summary["run_status"], "invalid")
            self.assertEqual(summary["counts"]["completed_records"], 1)
            self.assertEqual(summary["error_type"], "RuntimeError")
            self.assertNotIn("private text", summary_text)
            self.assertNotIn("private text", log_text)
            self.assertEqual(log_text, "run invalid: inference execution failed\n")

    def test_cli_help_exposes_explicit_execution_controls_without_model_loading(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.run_prompt_baseline", "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertIn("--execute", result.stdout)
        self.assertIn("--model-revision", result.stdout)
        self.assertIn("--allow-outside-private-output", result.stdout)

    def test_cli_defaults_to_planned_scaffold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            output_root = root / "outputs"
            input_path.write_text('{"private":"payload"}\n', encoding="utf-8")
            prompt_path.write_text("frozen-template", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_prompt_baseline",
                    "--protocol-id",
                    "B2-P1",
                    "--evaluation-slug",
                    "qalb14-dev",
                    "--input",
                    str(input_path),
                    "--prompt-template",
                    str(prompt_path),
                    "--outputs-root",
                    str(output_root),
                    "--allow-outside-private-output",
                ],
                check=True,
                text=True,
                capture_output=True,
            )

            run_id = "B2-P1__gemma3-4b-it__qalb14-dev__s3407__r01"
            summary = json.loads(
                (output_root / run_id / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["run_status"], "planned")
            self.assertIn('"run_status": "planned"', result.stdout)

    def test_cli_execution_contract_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            bundle_path = root / "bundle.json"
            input_path.write_text(
                '{"record_id":"r1","passage":"alpha","error":"alpha"}\n',
                encoding="utf-8",
            )
            prompt_path.write_text("frozen-template", encoding="utf-8")
            bundle_path.write_text('{}\n', encoding="utf-8")
            base = [
                sys.executable,
                "-m",
                "scripts.run_prompt_baseline",
                "--evaluation-slug",
                "qalb14-dev",
                "--input",
                str(input_path),
                "--prompt-template",
                str(prompt_path),
                "--outputs-root",
                str(root / "outputs"),
                "--allow-outside-private-output",
                "--execute",
                "--model-revision",
                "fixed-revision",
            ]

            missing_bundle = subprocess.run(
                base + ["--protocol-id", "B1-P1"],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(missing_bundle.returncode, 0)
            self.assertIn("requires --bundle", missing_bundle.stderr)

            unexpected_bundle = subprocess.run(
                base
                + [
                    "--protocol-id",
                    "B2-P1",
                    "--bundle",
                    str(bundle_path),
                ],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(unexpected_bundle.returncode, 0)
            self.assertIn("does not accept --bundle", unexpected_bundle.stderr)

    def test_cli_rejects_unpinned_execution_and_unsafe_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            input_path.write_text(
                '{"record_id":"r1","passage":"alpha","error":"alpha"}\n',
                encoding="utf-8",
            )
            prompt_path.write_text("frozen-template", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "scripts.run_prompt_baseline",
                "--protocol-id",
                "B2-P1",
                "--evaluation-slug",
                "qalb14-dev",
                "--input",
                str(input_path),
                "--prompt-template",
                str(prompt_path),
                "--outputs-root",
                str(root / "outputs"),
            ]

            unsafe = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertIn("private outputs must stay under", unsafe.stderr)

            unpinned = subprocess.run(
                command + ["--allow-outside-private-output", "--execute"],
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(unpinned.returncode, 0)
            self.assertIn("--model-revision", unpinned.stderr)


if __name__ == "__main__":
    unittest.main()
