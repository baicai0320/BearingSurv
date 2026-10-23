"""Shared task catalog for the public main-table protocol."""

from itertools import product


MODELS = ("otsurv", "rawcnn", "coxph", "coxboost", "rsf", "mtlr", "dsm", "bnnsurv")


def is_dual(model):
    return model == "otsurv"


def cpu_threads(model):
    return 37 if model == "otsurv" else 18


def task_gpu_ids(model, config):
    if config["device"] == "cpu" or model in {"coxph", "coxboost", "rsf", "dsm"}:
        return []
    ids = config["gpu_ids"]
    return list(ids) if is_dual(model) else [ids[0]]


def tasks_for(config):
    if config.get("seed", 0) != 0:
        raise ValueError("non-paper seed")
    choices = {"models": MODELS, "conditions": (0, 1, 2),
               "outer_folds": (1, 2, 3, 4, 5), "censoring_levels": (0.25, 0.5, 0.75)}
    for key, allowed in choices.items():
        values = config[key]
        if not values or len(values) != len(set(values)) or not set(values).issubset(allowed):
            raise ValueError(f"invalid or duplicated {key}")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    ids = config["gpu_ids"]
    if not ids or len(ids) != len(set(ids)) or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("GPU indices must be unique non-negative integers")
    if config["device"] == "cuda" and any(is_dual(m) for m in config["models"]) and len(ids) != 2:
        raise ValueError("dual-endpoint CUDA tasks require exactly two physical GPU indices")
    return [(f"C{c + 1}_q{q:g}_fold{f}_{m}", c, q, f, m)
            for c, q, f, m in product(config["conditions"], config["censoring_levels"],
                                      config["outer_folds"], config["models"])]
