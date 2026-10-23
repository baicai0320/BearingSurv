#!/usr/bin/env python3
"""Replay the bundled frozen FPT score source."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "XJTU-SY",
                        help="root directory containing the original XJTU-SY CSV files")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new directory for the generated FPT labels and manifest")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from src import config as cfg
    from fpt_detection.generation import replay_labels

    replay_labels(args.output_dir or cfg.FPT_LABELS_DIR, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
