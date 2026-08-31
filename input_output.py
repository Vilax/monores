#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:
    raise SystemExit("monores requires PyTorch: pip install torch") from exc

try:
    import mrcfile
except ImportError as exc:
    raise SystemExit("monores requires mrcfile: pip install mrcfile") from exc


def read_mrc(filename: str) -> np.ndarray:
    with mrcfile.open(filename, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float64)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise RuntimeError("%s is not a 3D volume (shape %s)" % (filename, data.shape))
    return np.ascontiguousarray(data)


def write_mrc(filename: str, volume, sampling: float) -> None:
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().to("cpu").numpy()
    data = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = float(sampling)   # see monores.py deviation #3
