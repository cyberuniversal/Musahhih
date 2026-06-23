#!/usr/bin/env python3
"""Convert Nahw-Passage into a frozen GEC evaluation JSONL file."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "nahw" / "Nahw-Passage.json"
OUTPUT = ROOT / "data" / "processed" / "nahw_gec_test.jsonl"

PROMPT = """صحح الكلمة الخاطئة المحددة في النص التالي.
أعد الكلمة المصححة فقط دون شرح أو علامات اقتباس.

النص:
{passage}

الكلمة الخاطئة:
{error}
"""


def main() -> None:
    if not INPUT.exists():
        raise SystemExit("Missing Nahw-Passage.json. Run scripts/download_nahw.py first.")

    with INPUT.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            record = {
                "id": f"nahw-{row['passage_id']}-{idx}",
                "passage_id": row["passage_id"],
                "prompt": PROMPT.format(
                    passage=row["passage"],
                    error=row["error"],
                ),
                "passage": row["passage"],
                "error": row["error"],
                "gold_correction": row["correction"],
                "gold_explanation": row["explanation"],
                "split": "test",
                "source": "Nahw-Passage",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} records to {OUTPUT}")
    print("IMPORTANT: keep this file test-only.")


if __name__ == "__main__":
    main()
