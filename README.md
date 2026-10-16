# Analyzer

A command-line Python tool that extracts indicators of compromise (IOCs) from plain-text threat reports and writes them to a CSV with a JSON summary. It uses only the Python standard library.

## What it extracts

| Type | How it is matched |
| --- | --- |
| `ipv4` | Dotted quad, validated with `ipaddress`. |
| `ipv6` | Hex/colon tokens with at least two colons (`::` compression allowed, `%zone` stripped), validated with `ipaddress.IPv6Address`. Times such as `12:30:45` are not valid addresses and are dropped. |
| `url` | `http`, `https` and `ftp`. A URL ends at whitespace, quotes or brackets; trailing `.,;:!?*` is stripped. |
| `domain` | Labels of 1-63 letters, digits or hyphens (no leading/trailing hyphen), total length up to 253, alphabetic TLD of 2-24 characters. Tokens ending in a common file extension (`report.txt`, `loader.dll`) are skipped, as are domains that are part of an email address. |
| `email` | `local@domain`, with the domain checked as above. |
| `md5`, `sha1`, `sha256` | Hex tokens of exactly 32, 40 or 64 characters. Other lengths are ignored. |

Before matching, common defanging is reversed: `hxxp`/`hxxps`, `[.]`, `(.)`, `{.}`, `[dot]`, `[:]`, `[://]`, `[@]` and `[at]`.

Each indicator is reported once per file, in first-seen order. The `note` column records:

- `scope=` for IPs: `global`, `private`, `loopback`, `link-local`, `multicast`, `reserved`, `unspecified` or `non-global`, from Python's `ipaddress` properties. Nothing is filtered on scope. `private` follows `ipaddress.is_private`, so documentation ranges such as `192.0.2.0/24`, `203.0.113.0/24` and `2001:db8::/32` count as private.
- `from url` for a domain taken from a URL host. The URL's host is also reported as a domain.
- `refanged` when the indicator was defanged in the source text.

## Running it

Python 3.11 or later (tested on 3.11, 3.12 and 3.13). There is nothing to install for the tool itself.

```bash
mkdir -p raw_reports && cp your_report.txt raw_reports/
python main.py                        # reads raw_reports/, writes output/
python main.py --input reports --output results --max-size-mb 50 --log-file run.log
```

Only `.txt` files are read. Other files, and files over `--max-size-mb` (default 100), are skipped with a warning and listed in the summary. A file with no indicators is fine. A missing input directory exits with code 2. Files are decoded as UTF-8, falling back to cp1252 and then latin-1.

With Docker:

```bash
docker build -t analyzer .
docker run --rm -v "$PWD/raw_reports:/app/raw_reports" -v "$PWD/output:/app/output" analyzer
```

## Output

`output/extracted_iocs.csv`:

```
file,ioc_type,ioc,note
campaign.txt,ipv4,185.220.101.47,scope=global; refanged
campaign.txt,ipv6,2001:db8:4f::1d,scope=private
campaign.txt,url,https://cdn-update.freight-billing.com/api/v2/check?id=7f3a,refanged
campaign.txt,domain,cdn-update.freight-billing.com,from url; refanged
campaign.txt,sha256,9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08,
```

`output/summary.json` has files processed and skipped (with reasons), files with indicators, the total, and counts per type.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
flake8 ioc_analyzer tests main.py
```

The tests cover each indicator type with positive and negative cases, plus an end-to-end CLI run. GitHub Actions runs flake8 and pytest on every push and pull request (`.github/workflows/tests.yml`).

## Limitations

This is regex extraction with structural checks. It finds strings that look like indicators, not strings that are malicious.

- No reputation or context. Vendor sites, reference links (for example `attack.mitre.org`), the report author's own email and documentation IPs are all reported. Expect to review the output.
- False positives from text that only resembles an indicator: code identifiers (`os.path`), file names whose extension is not on the skip list (`config.yaml`), abbreviations such as `Mr.Smith`, and any hex string of the right length (a git commit ID is reported as `sha1`).
- No TLD list. Any alphabetic TLD of 2-24 characters is accepted, so `foo.bar` is a domain and so is `example.zip` if it appears inside a URL. Outside a URL, a name whose last label is a common file extension is dropped, which also drops real domains under TLDs like `.zip`.
- Domains inside URLs are reported twice by design: once in the URL and once as a domain.
- False negatives from defanging styles not listed above (`evil(dot)com`, `evil . com`), indicators split across lines, internationalized domain names in Unicode, and punycode TLDs (`xn--p1ai`). Punycode in other labels is matched.
- Only `.txt` files are read. Indicators in PDF, HTML, Word or email files are not extracted unless converted to text first.
- IPs written in other forms (decimal integers, hex, CIDR suffixes) are not normalized; `10.0.0.0/8` is reported as `10.0.0.0`.

## Layout

- `ioc_analyzer/extract.py`: extraction and validation
- `ioc_analyzer/cli.py`: arguments, file handling, CSV and JSON output
- `main.py`: entry point
- `tests/`: pytest suite
- `experimental/`: unused model code (see below)

## History

I started this in January 2025 as a practice project for turning unstructured threat-intel text into structured data. The repo once included an LSTM classifier and Hugging Face transformer loading that were never wired into the pipeline, because there was no labeled data to train them on. That code now lives in `experimental/lstm_classifier.py`, with its dependencies in `requirements-ml.txt`. It is not part of the tool. I later rewrote the extraction, which had several broken patterns (IPv6 addresses and ordinary URLs never matched, file names were reported as domains), and added the CLI options and tests.

## License

MIT License — see [LICENSE](LICENSE).
