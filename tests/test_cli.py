import csv
import json
import subprocess
import sys
from pathlib import Path

from ioc_analyzer.cli import main

REPO = Path(__file__).resolve().parent.parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_end_to_end(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "apt.txt").write_text(
        "C2 hxxp://bad[.]example[.]net/gate and 203.0.113.9.\n"
        "Dropped loader.dll, md5 d41d8cd98f00b204e9800998ecf8427e\n",
        encoding="utf-8",
    )
    (reports / "empty.txt").write_text("", encoding="utf-8")
    (reports / "brief.pdf").write_bytes(b"%PDF-1.4 8.8.8.8")
    (reports / "subdir").mkdir()
    out = tmp_path / "out"

    assert main(["--input", str(reports), "--output", str(out)]) == 0

    rows = read_csv(out / "extracted_iocs.csv")
    assert {r["file"] for r in rows} == {"apt.txt"}
    got = [(r["ioc_type"], r["ioc"], r["note"]) for r in rows]
    assert got == [
        ("ipv4", "203.0.113.9", "scope=private"),
        ("url", "http://bad.example.net/gate", "refanged"),
        ("domain", "bad.example.net", "from url; refanged"),
        ("md5", "d41d8cd98f00b204e9800998ecf8427e", ""),
    ]

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["files_processed"] == 2
    assert summary["files_skipped"] == 1
    assert summary["skipped"] == [{"file": "brief.pdf", "reason": "not .txt"}]
    assert summary["files_with_iocs"] == 1
    assert summary["total_iocs"] == 4
    assert summary["counts_by_type"]["ipv4"] == 1
    assert summary["counts_by_type"]["ipv6"] == 0


def test_only_empty_files_still_writes_outputs(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "empty.txt").write_text("", encoding="utf-8")
    out = tmp_path / "out"

    assert main(["--input", str(reports), "--output", str(out)]) == 0
    assert read_csv(out / "extracted_iocs.csv") == []
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["files_processed"] == 1
    assert summary["total_iocs"] == 0


def test_oversized_file_skipped(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "big.txt").write_text("8.8.8.8 " * 200, encoding="utf-8")
    out = tmp_path / "out"

    assert main(["--input", str(reports), "--output", str(out), "--max-size-mb", "0.001"]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["skipped"] == [{"file": "big.txt", "reason": "too large"}]
    assert summary["files_processed"] == 0


def test_non_utf8_file_is_read(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "latin.txt").write_bytes("caf\xe9 host 8.8.8.8".encode("cp1252"))
    out = tmp_path / "out"

    assert main(["--input", str(reports), "--output", str(out)]) == 0
    assert [r["ioc"] for r in read_csv(out / "extracted_iocs.csv")] == ["8.8.8.8"]


def test_log_file(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "a.txt").write_text("8.8.8.8", encoding="utf-8")
    log = tmp_path / "run.log"

    assert main(["--input", str(reports), "--output", str(tmp_path / "out"), "--log-file", str(log)]) == 0
    assert "a.txt: 1 IOCs" in log.read_text(encoding="utf-8")


def test_missing_input_dir_exit_code_2(tmp_path):
    assert main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_main_py_entry_point(tmp_path):
    result = subprocess.run(
        [sys.executable, str(REPO / "main.py"), "--input", str(tmp_path / "missing")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Input directory not found" in result.stderr


def test_core_does_not_import_ml_stack():
    code = (
        "import sys, ioc_analyzer.cli; "
        "bad = {'pandas', 'tensorflow', 'keras', 'transformers', 'spacy', 'tenacity'} & set(sys.modules); "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0
