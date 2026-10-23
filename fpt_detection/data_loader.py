"""Shared raw CSV parsing and record inventory for the XJTU-SY workflow."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src import config as cfg


def _numeric_csv_files(path: Path) -> list[Path]:
    return sorted(
        [file for file in path.glob("*.csv") if re.fullmatch(r"\d+\.csv", file.name)],
        key=lambda file: int(file.stem),
    )


def resolve_bearing_directory(
    data_dir: Path | str, condition_id: int, bearing_id: int
) -> Path:
    """Resolve the canonical XJTU bearing directory without assuming its rig ID.

    The three released operating conditions conventionally use ``Bearing1_*``,
    ``Bearing2_*``, and ``Bearing3_*`` respectively.  The FPT protocol indexes
    the five bearings within a condition as 1..5, so hard-coding ``Bearing1``
    for every condition silently rejects the original C2 and C3 directories.
    Exactly one rig-prefixed directory is required for a protocol bearing.
    """
    if int(condition_id) not in cfg.CONDITIONS or int(bearing_id) not in cfg.BEARING_IDS:
        raise ValueError("condition_id and bearing_id must be part of the XJTU protocol")
    root = Path(data_dir)
    condition = cfg.CONDITION_NAMES[int(condition_id)]
    condition_path = root / condition
    if not condition_path.is_dir():
        raise FileNotFoundError(f"missing XJTU condition directory: {condition_path}")
    # XJTU-SY names the physical rig by condition (C1/Bearing1, C2/Bearing2,
    # C3/Bearing3). Some server extracts also retain duplicate alternate rig
    # directories; the canonical condition-specific directory must win rather
    # than making an otherwise valid source spuriously ambiguous.
    canonical = condition_path / f"Bearing{int(condition_id) + 1}_{int(bearing_id)}"
    if canonical.is_dir():
        return canonical
    pattern = re.compile(rf"Bearing\d+_{int(bearing_id)}$")
    candidates = sorted(
        path for path in condition_path.iterdir()
        if path.is_dir() and pattern.fullmatch(path.name)
    )
    if not candidates:
        raise FileNotFoundError(
            f"no XJTU bearing directory for condition {condition} bearing {bearing_id} "
            f"below {condition_path}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"ambiguous XJTU bearing directories for condition {condition} "
            f"bearing {bearing_id}: {[path.name for path in candidates]}"
        )
    return candidates[0]


def _read_signal(path: Path, *, dtype=np.float64) -> np.ndarray:
    for header in ("infer", None):
        try:
            frame = pd.read_csv(path, header=header)
        except (OSError, ValueError) as exc:
            if header is None:
                raise ValueError(f"cannot read raw signal {path}: {exc}") from exc
            continue
        if frame.shape[1] < 2:
            continue
        values = frame.iloc[:, :2].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=dtype
        )
        if values.shape == (cfg.FPT_SIGNAL_LENGTH, 2) and np.isfinite(values).all():
            return values
    raise ValueError(
        f"XJTU input {path} must be a finite [{cfg.FPT_SIGNAL_LENGTH}, 2] CSV record"
    )


class XJTUDataLoader:
    """Locate the contiguous raw records used by FPT and survival workflows."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else cfg.DATA_DIR

    def bearing_path(self, condition_id: int, bearing_id: int) -> Path:
        """Return the unique raw bearing directory for a protocol cell."""
        return resolve_bearing_directory(self.data_dir, condition_id, bearing_id)

    def raw_record_paths(
        self, condition_id: int, bearing_id: int
    ) -> tuple[Path, list[Path]]:
        """Return validated, numerically ordered raw CSV paths for one bearing."""
        bearing_path = self.bearing_path(condition_id, bearing_id)
        files = _numeric_csv_files(bearing_path)
        if not files:
            raise FileNotFoundError(f"no numeric raw records found in {bearing_path}")
        names = [int(file.stem) for file in files]
        if names != list(range(1, len(files) + 1)):
            raise ValueError(f"raw XJTU filenames must be contiguous 1..N in {bearing_path}")
        return bearing_path, files
