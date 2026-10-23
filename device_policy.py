"""Physical device policy, independent of numerical model configuration."""

import socket


# This shared server must never fall back to its other users' devices.
SERVER_GPU_ALLOWLIST = {"z133-8": (4, 5)}


def allowed_gpu_ids():
    return SERVER_GPU_ALLOWLIST.get(socket.gethostname().split(".")[0])


def default_gpu_ids():
    return list(allowed_gpu_ids() or (0, 1))


def validate_gpu_ids(indices):
    allowed = allowed_gpu_ids()
    if (not indices or any(type(index) is not int or index < 0 for index in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("only physical GPUs with unique non-negative integer indices are accepted")
    if allowed is not None and not set(indices).issubset(allowed):
        raise ValueError("only physical GPUs 4 and 5 are authorized on this server")
    return list(indices)
