#!/usr/bin/env python3
"""Download the three public Nahw data files from the official QCRI repository."""

from pathlib import Path
import hashlib
import urllib.request

BASE = "https://raw.githubusercontent.com/qcri/nahw-arabic-grammar-benchmark/main/Data"
FILES = [
    "Nahw-MCQ.csv",
    "Nahw-Passage.json",
    "Synthetic_10K_Grammar_Questions.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "nahw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        url = f"{BASE}/{filename}"
        target = out_dir / filename
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            raise SystemExit(
                f"Failed to download {filename}: {exc}\n"
                "Check your internet connection and repository availability."
            ) from exc
        print(f"Saved {target} ({target.stat().st_size:,} bytes)")
        print(f"SHA256 {sha256(target)}")


if __name__ == "__main__":
    main()
