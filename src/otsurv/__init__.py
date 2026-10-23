"""Raw-waveform models published by BearingSurv."""

from otsurv.model_r14f_otsurv import R14FOTSurv
from otsurv.model_raw_cnn_cox import RawCNNCox


def __getattr__(name):
    if name == "OTSurv":
        from paper.models import OTSurv
        return OTSurv
    raise AttributeError(name)


__all__ = ["OTSurv"]
