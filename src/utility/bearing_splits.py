"""Deterministic, bearing-disjoint outer cross-validation splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BearingFold:
    """One outer fold whose test rows all belong to one physical bearing."""

    fold_index: int
    held_out_bearing: Hashable
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(frozen=True)
class NestedBearingFold:
    """Outer-test and cyclic inner-validation groups for one formal fold."""

    fold_index: int
    held_out_bearing: Hashable
    validation_bearing: Hashable
    train_bearings: tuple[Hashable, ...]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def nested_bearing_role_record(fold: NestedBearingFold) -> dict[str, str]:
    """Return stable, CSV-friendly bearing-set provenance for a nested fold."""

    return {
        "TrainBearings": "+".join(map(str, fold.train_bearings)),
        "ValidationBearings": str(fold.validation_bearing),
        "TestBearings": str(fold.held_out_bearing),
    }


def select_held_out_bearings(
    bearing_ids: Iterable[Hashable],
    n_folds: int | None = None,
) -> tuple[Hashable, ...]:
    """Return the deterministic bearing order used by the outer protocol.

    ``n_folds`` exists for explicitly requested development runs. Formal XJTU
    runs should leave it unset so all five configured bearing IDs are used.
    """

    ordered = tuple(bearing_ids)
    if not ordered:
        raise ValueError("at least one bearing ID is required")
    if len(set(ordered)) != len(ordered):
        raise ValueError("bearing IDs must be unique")

    if n_folds is None:
        return ordered
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_folds > len(ordered):
        raise ValueError(
            f"n_folds={n_folds} exceeds the {len(ordered)} available bearings"
        )
    return ordered[:n_folds]


def build_bearing_disjoint_folds(
    data: pd.DataFrame,
    held_out_bearings: Sequence[Hashable] | None = None,
    *,
    bearing_column: str = "bearing_id",
) -> tuple[BearingFold, ...]:
    """Build leave-one-bearing-out row indices without minute-row leakage."""

    if bearing_column not in data.columns:
        raise ValueError(f"missing bearing column: {bearing_column}")
    if data.empty:
        raise ValueError("cannot split an empty dataset")

    groups = data[bearing_column]
    if groups.isna().any():
        raise ValueError(f"{bearing_column} contains missing values")

    available = tuple(pd.unique(groups))
    requested = available if held_out_bearings is None else tuple(held_out_bearings)
    if not requested:
        raise ValueError("at least one held-out bearing is required")
    if len(set(requested)) != len(requested):
        raise ValueError("held-out bearing IDs must be unique")

    missing = [bearing_id for bearing_id in requested if bearing_id not in available]
    if missing:
        raise ValueError(f"held-out bearings are absent from the data: {missing}")

    all_positions = np.arange(len(data), dtype=np.int64)
    group_values = groups.to_numpy(copy=False)
    folds: list[BearingFold] = []
    fold_number_by_bearing = {
        bearing_id: index + 1 for index, bearing_id in enumerate(available)
    }
    for held_out in requested:
        test_mask = group_values == held_out
        train_indices = all_positions[~test_mask]
        test_indices = all_positions[test_mask]
        if train_indices.size == 0:
            raise ValueError("an outer fold must contain at least one training bearing")
        if test_indices.size == 0:
            raise RuntimeError(f"outer fold for bearing {held_out!r} is empty")
        if np.intersect1d(train_indices, test_indices).size:
            raise RuntimeError("bearing split produced overlapping row indices")

        train_groups = set(group_values[train_indices])
        test_groups = set(group_values[test_indices])
        if test_groups != {held_out} or held_out in train_groups:
            raise RuntimeError("bearing split violated group isolation")

        folds.append(
            BearingFold(
                fold_index=fold_number_by_bearing[held_out],
                held_out_bearing=held_out,
                train_indices=train_indices,
                test_indices=test_indices,
            )
        )
    return tuple(folds)


def build_nested_bearing_folds(
    data: pd.DataFrame,
    bearing_ids: Sequence[Hashable],
    held_out_bearings: Sequence[Hashable] | None = None,
    *,
    bearing_column: str = "bearing_id",
) -> tuple[NestedBearingFold, ...]:
    """Build deterministic outer-test/inner-validation bearing partitions.

    For ordered bearings ``(B1, ..., Bn)``, outer fold ``k`` tests on ``Bk``
    and validates on the next bearing cyclically. All other bearings form the
    training set. With the formal five-bearing XJTU protocol this yields three
    training bearings in every fold.
    """

    if bearing_column not in data.columns:
        raise ValueError(f"missing bearing column: {bearing_column}")
    if data.empty:
        raise ValueError("cannot split an empty dataset")

    ordered = tuple(bearing_ids)
    if len(ordered) < 3:
        raise ValueError("nested bearing folds require at least three bearings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("bearing IDs must be unique")

    groups = data[bearing_column]
    if groups.isna().any():
        raise ValueError(f"{bearing_column} contains missing values")
    if "time_step" in data.columns:
        time_steps = pd.to_numeric(data["time_step"], errors="coerce")
        if time_steps.isna().any() or not np.isfinite(
            time_steps.to_numpy(dtype=float)
        ).all():
            raise ValueError("time_step must contain finite values")
        if (time_steps < 0).any() or (time_steps != np.floor(time_steps)).any():
            raise ValueError("time_step must contain non-negative integer indices")
        minute_keys = pd.DataFrame(
            {
                "bearing_id": groups.to_numpy(copy=False),
                "time_step": time_steps.astype(np.int64).to_numpy(copy=False),
            }
        )
        if minute_keys.duplicated().any():
            raise ValueError("bearing_id/time_step minute keys must be unique")
    available = tuple(pd.unique(groups))
    if set(available) != set(ordered):
        missing = [bearing_id for bearing_id in ordered if bearing_id not in available]
        unexpected = [bearing_id for bearing_id in available if bearing_id not in ordered]
        raise ValueError(
            "data bearing groups do not match the configured protocol: "
            f"missing={missing}, unexpected={unexpected}"
        )

    requested = ordered if held_out_bearings is None else tuple(held_out_bearings)
    if not requested:
        raise ValueError("at least one held-out bearing is required")
    if len(set(requested)) != len(requested):
        raise ValueError("held-out bearing IDs must be unique")
    missing_requested = [value for value in requested if value not in ordered]
    if missing_requested:
        raise ValueError(
            f"held-out bearings are absent from the configured order: {missing_requested}"
        )

    positions = np.arange(len(data), dtype=np.int64)
    group_values = groups.to_numpy(copy=False)
    all_position_set = set(positions.tolist())
    folds: list[NestedBearingFold] = []

    for held_out in requested:
        outer_position = ordered.index(held_out)
        validation_bearing = ordered[(outer_position + 1) % len(ordered)]
        train_bearings = tuple(
            value
            for value in ordered
            if value not in {held_out, validation_bearing}
        )

        test_indices = positions[group_values == held_out]
        validation_indices = positions[group_values == validation_bearing]
        train_mask = np.isin(group_values, train_bearings)
        train_indices = positions[train_mask]

        split_sets = (
            set(train_indices.tolist()),
            set(validation_indices.tolist()),
            set(test_indices.tolist()),
        )
        if any(not values for values in split_sets):
            raise RuntimeError("nested bearing split produced an empty partition")
        if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("nested bearing partitions overlap")
        if set().union(*split_sets) != all_position_set:
            raise RuntimeError("nested bearing partitions do not cover every row")

        actual_train = set(group_values[train_indices])
        actual_validation = set(group_values[validation_indices])
        actual_test = set(group_values[test_indices])
        if actual_train != set(train_bearings):
            raise RuntimeError("training rows do not match the configured bearings")
        if actual_validation != {validation_bearing}:
            raise RuntimeError("validation rows cross bearing boundaries")
        if actual_test != {held_out}:
            raise RuntimeError("test rows cross bearing boundaries")

        folds.append(
            NestedBearingFold(
                fold_index=outer_position + 1,
                held_out_bearing=held_out,
                validation_bearing=validation_bearing,
                train_bearings=train_bearings,
                train_indices=train_indices,
                validation_indices=validation_indices,
                test_indices=test_indices,
            )
        )

    return tuple(folds)
