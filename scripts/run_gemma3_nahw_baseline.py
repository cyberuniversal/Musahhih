#!/usr/bin/env python3
"""Run a zero-shot Gemma 3 GEC baseline on the held-out Nahw-Passage test set."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import unicodedata

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "nahw_gec_test.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "gemma3_nahw_gec_predictions.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"^[\"'«»“”]+|[\"'«»“”]+$", "", text)
    text = text.splitlines()[0].strip() if text else ""
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("Evaluation file missing. Run scripts/prepare_nahw_eval.py.")

    rows = load_jsonl(args.data)
    if args.limit:
        rows = rows[: args.limit]

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, padding_side="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0

    with args.output.open("w", encoding="utf-8") as out:
        for row in tqdm(rows):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": row["prompt"]}],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            answer_ids = generated[0, prompt_len:]
            raw_prediction = processor.decode(answer_ids, skip_special_tokens=True)
            prediction = normalize(raw_prediction)
            gold = normalize(row["gold_correction"])
            is_correct = prediction == gold
            correct += int(is_correct)

            out_row = {
                **row,
                "model": args.model,
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "normalized_gold": gold,
                "exact_match": is_correct,
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    accuracy = correct / len(rows) if rows else 0.0
    summary = {
        "model": args.model,
        "examples": len(rows),
        "correct": correct,
        "exact_match_accuracy": accuracy,
        "predictions_file": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
