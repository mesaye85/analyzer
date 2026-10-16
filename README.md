# Analyzer

A command-line Python tool that scans plain-text threat reports and extracts indicators of compromise (IPv4 addresses, URLs, email addresses, file hashes, domains) into a CSV, with a JSON summary of counts per type.

## Context

I built this in January 2025 as a practice project for turning unstructured threat-intel text into structured data. Earlier iterations (visible in the git history) also tried T5 summarization and spaCy tokenization; the current version settled on regex extraction with validation, plus an LSTM classifier scaffold that is not yet wired in.

## Stack

- Python 3.10 (Dockerfile) / 3.11 (pins were compiled with pip-compile on 3.11)
- pandas for output, `re` for extraction, tenacity for read retries
- TensorFlow/Keras 3, transformers and spaCy are imported but only used by the unwired model code
- Pinned dependencies in `requirements.txt`, generated from `requirements.in`

## How it works

1. `process_files()` walks `raw_reports/`, validates each file (`.txt` only, under 100 MB) and reads it, trying utf-8, latin-1 and cp1252 in turn.
2. `IOCExtractor.extract_iocs()` runs each pattern in `IOCExtractor.PATTERNS`, applies an optional validator (IPv4 octets 0-255, domain labels at most 63 chars), and de-duplicates per file.
3. `save_results()` writes `output/extracted_iocs.csv` (`file,ioc_type,ioc`) and `output/metrics_<timestamp>.json`.

`ModelBuilder` (embedding + two LSTM layers, binary output) and `ModelManager` (Hugging Face seq2seq loader) are defined but never called from `main()`.

## Running it

Use Python 3.10 or 3.11; the pinned `blis==1.2.0` (via spaCy) does not install on 3.13.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt tenacity   # tenacity is not in requirements.txt yet
mkdir -p raw_reports && cp your_report.txt raw_reports/
python main.py
```

The install pulls TensorFlow, so expect it to be large. Logs go to `analyzer.log`.

Docker: the image currently fails at import because `tenacity` is not in `requirements.txt`. Once it is added, mount the input and output directories (`.dockerignore` excludes them):

```bash
docker build -t analyzer .
docker run --rm -v "$PWD/raw_reports:/app/raw_reports" -v "$PWD/output:/app/output" analyzer
```

## Status

Prototype. The extraction path works end to end on `.txt` input. There are no tests and no CLI arguments.

## Known gaps

- `main.py` `PATTERNS["ipv6"]` uses a capturing group, so `re.findall` returns only the last group (e.g. `0370:`) and the validator then drops it. IPv6 addresses are never reported.
- `PATTERNS["urls"]` has a malformed character class (`[-\w.%[\da-fA-F]{2}]+`) and fails to match ordinary URLs such as `https://example.com/path`; the host is picked up as a domain instead.
- `PATTERNS["domains"]` matches file names (`report.txt`, `malware.exe`). There is no TLD allow-list or defanging support (`hxxp`, `[.]`).
- `PATTERNS["hashes"]` (`{32,64}`) also accepts 33-63 character hex strings that are not MD5/SHA-1/SHA-256.
- One non-`.txt` file in `raw_reports/` aborts the run: tenacity retries the validation error three times, then raises `RetryError`, which `process_files()` does not catch.
- If no IOCs are found, `save_results()` raises `KeyError: 'ioc_type'` on the empty DataFrame. A missing `raw_reports/` raises `FileNotFoundError`.
- `tenacity` is imported but missing from `requirements.in` and `requirements.txt`; `nltk` is installed (and the Dockerfile downloads `punkt`) but never used.
- Dockerfile: `--only-binary=:all:` plus the build toolchain is redundant, and the image carries TensorFlow for a regex-only workload.
- No `LICENSE` file, although an earlier version of this README said MIT.

## What I'd do next

- Fix the IPv6 and URL patterns, add defang handling, and cover each IOC type with pytest fixtures (true and false positives).
- Make the per-file error handling catch `RetryError`, handle the empty-result case, and take input/output paths as arguments.
- Move the ML code to an optional module so the default install is pandas + tenacity only.
