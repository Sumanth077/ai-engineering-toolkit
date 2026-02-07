"""CLI entry point for the intraday trading analyzer."""

from __future__ import annotations

import json
import sys

from intraday_analyzer.analyzer import IntradayAnalyzer
from intraday_analyzer.models import parse_input


def main() -> None:
    """Read JSON input from stdin or a file argument, print the trade signal."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path) as f:
            raw = json.load(f)
    else:
        raw = json.load(sys.stdin)

    inp = parse_input(raw)
    analyzer = IntradayAnalyzer()
    print(analyzer.analyze_json(inp))


if __name__ == "__main__":
    main()
