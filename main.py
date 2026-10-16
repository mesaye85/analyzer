"""Thin entry point so `python main.py` keeps working. See ioc_analyzer/cli.py."""

import sys

from ioc_analyzer.cli import main

if __name__ == "__main__":
    sys.exit(main())
