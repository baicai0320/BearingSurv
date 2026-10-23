"""Public model factory. Numerical implementations live in ``paper.training``.

The former simplified adapters are retired. OTSurv accepts raw signals and
explicit FPT supervision; tabular baselines accept the verified fold frame.
"""


def make_regressor(name, condition=1, *, device="cpu", time_device=None, cpu_workers=18):
    from paper.models import OTSurv, PaperBaseline, MODEL_NAMES

    name = {"r14f": "otsurv", "raw-cnn": "rawcnn"}.get(name, name)
    if name not in MODEL_NAMES:
        raise ValueError(f"unsupported public model: {name}")
    if name == "otsurv":
        return OTSurv(condition, device=device, time_device=time_device)
    return PaperBaseline(name, condition, device=device, cpu_workers=cpu_workers)
