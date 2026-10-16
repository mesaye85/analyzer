"""Command-line entry point: scan a directory of .txt reports for IOCs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

from .extract import IOC_TYPES, extract_iocs

logger = logging.getLogger("ioc_analyzer")

CSV_NAME = "extracted_iocs.csv"
SUMMARY_NAME = "summary.json"
ENCODINGS = ("utf-8", "cp1252", "latin-1")  # latin-1 decodes any byte sequence


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable: latin-1 decodes any input")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ioc-analyzer",
        description="Extract IPs, domains, URLs, emails and hashes from .txt threat reports.",
    )
    parser.add_argument("--input", default="raw_reports", type=Path,
                        help="directory of .txt reports (default: raw_reports)")
    parser.add_argument("--output", default="output", type=Path,
                        help="directory for extracted_iocs.csv and summary.json (default: output)")
    parser.add_argument("--max-size-mb", default=100, type=float,
                        help="skip files larger than this many MB (default: 100)")
    parser.add_argument("--log-file", type=Path, help="also write the log to this file")
    return parser.parse_args(argv)


def configure_logging(log_file: Optional[Path]) -> None:
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def run(input_dir: Path, output_dir: Path, max_size_mb: float) -> int:
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 2

    max_bytes = int(max_size_mb * 1024 * 1024)
    processed: List[str] = []
    skipped: List[dict] = []
    rows: List[dict] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".txt":
            logger.warning("Skipping %s: not a .txt file", path.name)
            skipped.append({"file": path.name, "reason": "not .txt"})
            continue
        size = path.stat().st_size
        if size > max_bytes:
            logger.warning("Skipping %s: %d bytes exceeds --max-size-mb %s", path.name, size, max_size_mb)
            skipped.append({"file": path.name, "reason": "too large"})
            continue
        try:
            text = read_text(path)
        except OSError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            skipped.append({"file": path.name, "reason": "unreadable"})
            continue

        iocs = extract_iocs(text)
        processed.append(path.name)
        logger.info("%s: %d IOCs", path.name, len(iocs))
        rows.extend({"file": path.name, "ioc_type": i.ioc_type, "ioc": i.value, "note": i.note} for i in iocs)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "ioc_type", "ioc", "note"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["ioc_type"] for row in rows)
    summary = {
        "files_processed": len(processed),
        "files_skipped": len(skipped),
        "skipped": skipped,
        "files_with_iocs": len({row["file"] for row in rows}),
        "total_iocs": len(rows),
        "counts_by_type": {t: counts.get(t, 0) for t in IOC_TYPES},
    }
    summary_path = output_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logger.info("Processed %d file(s), skipped %d, %d IOCs. Wrote %s and %s",
                len(processed), len(skipped), len(rows), csv_path, summary_path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    return run(args.input, args.output, args.max_size_mb)
