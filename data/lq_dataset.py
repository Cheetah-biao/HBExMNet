from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


def _scan_tiff_files(path_like) -> list[str]:
    path = Path(path_like)
    if path.is_file():
        return [str(path)]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    tif_files = sorted(
        [*path.glob("*.tif"), *path.glob("*.tiff")],
        key=lambda item: item.name.lower(),
    )
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files were found in: {path}")
    return [str(item) for item in tif_files]


def _squeeze_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume after squeezing, but got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


class LQDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.paths_LQ = _scan_tiff_files(opt["dataroot_LQ"])
        val_opt = opt.get("val", {}) or {}
        self.low_p = float(val_opt.get("low_p", 0.2))
        self.high_p = float(val_opt.get("high_p", 99.99))

    def __len__(self):
        return len(self.paths_LQ)

    def __getitem__(self, index):
        path = self.paths_LQ[index]
        volume = _squeeze_volume(tifffile.imread(path))

        mi = np.percentile(volume, self.low_p)
        ma = np.percentile(volume, self.high_p)
        if ma <= mi:
            ma = float(volume.max())
            mi = float(volume.min())
        if ma <= mi:
            ma = mi + 1.0

        normalized = np.clip((volume - mi) / (ma - mi), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(normalized), np.array([mi], dtype=np.float32), np.array([ma], dtype=np.float32)
