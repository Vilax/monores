#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import sys

import torch

PI = math.pi
DBL_MIN = sys.float_info.min          # 2.2250738585072014e-308, as C's DBL_MIN


# ---------------------------------------------------------------------------
# xmippCore scalar helpers
# ---------------------------------------------------------------------------

def icdf_gauss(p: float) -> float:
    """Inverse Gaussian CDF, literal port of ``icdf_gauss`` in
    ``xmippCore/core/xmipp_funcs.cpp``"""
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    if p < 0.5:
        # F^-1(p) = -G^-1(p)
        t = math.sqrt(-2.0 * math.log(p))
        z = t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
        return -z
    # F^-1(p) = G^-1(1-p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    z = t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
    return z


def _c_round(x: float) -> int:
    """C's ``round()``: half away from zero (Python's ``round`` is half-to-even)."""
    if math.isnan(x) or math.isinf(x):
        raise ValueError("cannot round %r to an FFT index" % x)
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))


def _c_div(a: float, b: float) -> float:
    """IEEE-754 division as C does it: ``x / 0.0`` is +/-inf (or NaN for 0/0)."""
    if b == 0.0:
        if a == 0.0:
            return float("nan")
        return math.copysign(float("inf"), a) * math.copysign(1.0, b)
    return a / b


def _c_sqrt(x: float) -> float:
    """C's ``sqrt``: NaN for negative arguments (see deviation #6 in monores.py)."""
    if x < 0.0:
        return float("nan")
    return math.sqrt(x)


def digfreq2fft_idx(freq: float, size: int) -> int:
    """``DIGFREQ2FFT_IDX`` from ``xmippCore/core/xmipp_fft.h``."""
    idx = _c_round(size * freq)
    if idx < 0:
        idx += int(size)
    return idx


def fft_idx2digfreq(idx: int, size: int) -> float:
    """``FFT_IDX2DIGFREQ`` from ``xmippCore/core/xmipp_fft.h`` (scalar form).
    """
    if size <= 1:
        return 0.0
    idx = int(idx)
    size = int(size)
    num = idx if idx <= (size >> 1) else (-size + idx)
    return num / float(size)


def fft_idx2digfreq_t(idx: torch.Tensor, size: int) -> torch.Tensor:
    """Vectorised ``FFT_IDX2DIGFREQ`` over a tensor of integer FFT indices."""
    if size <= 1:
        return torch.zeros_like(idx, dtype=torch.float64)
    half = int(size) >> 1
    num = torch.where(idx <= half, idx, idx - int(size))
    return num.to(torch.float64) / float(size)
