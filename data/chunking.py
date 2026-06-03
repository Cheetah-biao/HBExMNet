from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom


def _scan_tiff_files(path_like) -> list[Path]:
    path = Path(path_like)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    tif_files = sorted([*path.glob("*.tif"), *path.glob("*.tiff")], key=lambda item: item.name.lower())
    if not tif_files:
        raise FileNotFoundError(f"No TIFF files were found in: {path}")
    return tif_files


def _read_volume(path: Path) -> np.ndarray:
    volume = np.squeeze(np.asarray(tifffile.imread(str(path))))
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, but got shape {volume.shape} from {path}.")
    return volume.astype(np.float32, copy=False)


def _normalize(volume: np.ndarray) -> np.ndarray:
    min_val = float(volume.min())
    max_val = float(volume.max())
    if max_val <= min_val:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - min_val) / (max_val - min_val)).astype(np.float32)


def _resize_like(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    if tuple(volume.shape) == tuple(target_shape):
        return volume.astype(np.float32, copy=False)
    factors = [t / s for s, t in zip(volume.shape, target_shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


def _pair_files(primary_files: list[Path], secondary_files: list[Path]) -> list[tuple[Path, Path]]:
    primary_map = {path.stem: path for path in primary_files}
    secondary_map = {path.stem: path for path in secondary_files}
    common = sorted(set(primary_map) & set(secondary_map))
    if common:
        return [(primary_map[name], secondary_map[name]) for name in common]

    if len(primary_files) != len(secondary_files):
        raise ValueError(
            "Could not match files by stem, and the file counts are different. "
            f"Primary: {len(primary_files)}, Secondary: {len(secondary_files)}."
        )
    return list(zip(sorted(primary_files), sorted(secondary_files)))


def _pair_triplets(gt_files: list[Path], lr_files: list[Path], mr_files: list[Path]) -> list[tuple[Path, Path, Path]]:
    gt_map = {path.stem: path for path in gt_files}
    lr_map = {path.stem: path for path in lr_files}
    mr_map = {path.stem: path for path in mr_files}
    common = sorted(set(gt_map) & set(lr_map) & set(mr_map))
    if common:
        return [(gt_map[name], lr_map[name], mr_map[name]) for name in common]

    if not (len(gt_files) == len(lr_files) == len(mr_files)):
        raise ValueError(
            "Could not match GT/LR/MR files by stem, and the file counts are different. "
            f"GT: {len(gt_files)}, LR: {len(lr_files)}, MR: {len(mr_files)}."
        )
    return list(zip(sorted(gt_files), sorted(lr_files), sorted(mr_files)))


def _extract_patches_with_mid(
    gt_volume: np.ndarray,
    lr_volume: np.ndarray,
    mr_volume: np.ndarray,
    patch_size,
    overlap: float,
):
    patch_d, patch_h, patch_w = [int(v) for v in patch_size]
    patch_d = max(1, patch_d)
    patch_h = max(1, patch_h)
    patch_w = max(1, patch_w)

    lr_shape = lr_volume.shape
    gt_shape = gt_volume.shape
    if mr_volume.shape != lr_shape:
        mr_volume = _resize_like(mr_volume, lr_shape)
    scale = [g / l for g, l in zip(gt_shape, lr_shape)]

    patches_gt = []
    patches_lr = []
    patches_mid = []

    max_patches = _estimate_patch_budget(lr_shape, gt_shape, (patch_d, patch_h, patch_w), overlap)

    for start_d, start_h, start_w in _iter_patch_origins(lr_shape, (patch_d, patch_h, patch_w), overlap, max_patches):
        lr_patch = lr_volume[start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]
        mr_patch = mr_volume[start_d:start_d + patch_d, start_h:start_h + patch_h, start_w:start_w + patch_w]

        gt_start = [int(round(start * factor)) for start, factor in zip((start_d, start_h, start_w), scale)]
        gt_size = [max(1, int(round(size * factor))) for size, factor in zip((patch_d, patch_h, patch_w), scale)]
        gt_end = [min(length, start + size) for start, size, length in zip(gt_start, gt_size, gt_shape)]
        gt_start = [max(0, end - size) for end, size in zip(gt_end, gt_size)]
        gt_patch = gt_volume[gt_start[0]:gt_end[0], gt_start[1]:gt_end[1], gt_start[2]:gt_end[2]]
        if gt_patch.shape != tuple(gt_size):
            gt_patch = _resize_like(gt_patch, tuple(gt_size))

        patches_gt.append(gt_patch[np.newaxis, ...])
        patches_lr.append(lr_patch[np.newaxis, ...])
        patches_mid.append(mr_patch[np.newaxis, ...])

    return patches_gt, patches_lr, patches_mid


def _iter_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch >= length:
        return [0]

    starts = list(range(0, max(length - patch, 0) + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iter_patch_origins(shape, patch_size, overlap: float, max_patches=None):
    patch_d, patch_h, patch_w = [max(1, int(v)) for v in patch_size]
    stride = [max(1, int(round(size * (1.0 - overlap)))) for size in (patch_d, patch_h, patch_w)]

    starts_d = _iter_starts(shape[0], patch_d, stride[0])
    starts_h = _iter_starts(shape[1], patch_h, stride[1])
    starts_w = _iter_starts(shape[2], patch_w, stride[2])

    all_origins = [(d, h, w) for d in starts_d for h in starts_h for w in starts_w]
    if max_patches is not None and len(all_origins) > max_patches:
        rng = np.random.default_rng(0)
        sampled_indices = sorted(rng.choice(len(all_origins), size=max_patches, replace=False).tolist())
        all_origins = [all_origins[index] for index in sampled_indices]
    return all_origins


def _estimate_patch_budget(lr_shape, gt_shape, patch_size, overlap: float, target_bytes=2_000_000_000):
    lr_patch = [max(1, int(v)) for v in patch_size]
    scale = [g / l for g, l in zip(gt_shape, lr_shape)]
    gt_patch = [max(1, int(round(size * factor))) for size, factor in zip(lr_patch, scale)]
    bytes_per_patch = (
        np.prod(gt_patch) * 4
        + np.prod(lr_patch) * 4
        + np.prod(lr_patch) * 4
    )
    if bytes_per_patch <= 0:
        return None
    max_patches = max(32, int(target_bytes // bytes_per_patch))
    return max_patches


def _extract_patches(gt_volume: np.ndarray, lr_volume: np.ndarray, patch_size, overlap: float):
    patch_d, patch_h, patch_w = [int(v) for v in patch_size]
    patch_d = max(1, patch_d)
    patch_h = max(1, patch_h)
    patch_w = max(1, patch_w)

    lr_shape = lr_volume.shape
    gt_shape = gt_volume.shape
    scale = [g / l for g, l in zip(gt_shape, lr_shape)]
    mid_volume = _resize_like(gt_volume, lr_shape)

    patches_gt = []
    patches_lr = []
    patches_mid = []

    max_patches = _estimate_patch_budget(lr_shape, gt_shape, (patch_d, patch_h, patch_w), overlap)

    for start_d, start_h, start_w in _iter_patch_origins(lr_shape, (patch_d, patch_h, patch_w), overlap, max_patches):
        lr_patch = lr_volume[
            start_d:start_d + patch_d,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
        ]
        mid_patch = mid_volume[
            start_d:start_d + patch_d,
            start_h:start_h + patch_h,
            start_w:start_w + patch_w,
        ]

        gt_start = [int(round(start * factor)) for start, factor in zip((start_d, start_h, start_w), scale)]
        gt_size = [max(1, int(round(size * factor))) for size, factor in zip((patch_d, patch_h, patch_w), scale)]
        gt_end = [min(length, start + size) for start, size, length in zip(gt_start, gt_size, gt_shape)]
        gt_start = [max(0, end - size) for end, size in zip(gt_end, gt_size)]
        gt_patch = gt_volume[
            gt_start[0]:gt_end[0],
            gt_start[1]:gt_end[1],
            gt_start[2]:gt_end[2],
        ]

        expected_gt_shape = tuple(gt_size)
        if gt_patch.shape != expected_gt_shape:
            gt_patch = _resize_like(gt_patch, expected_gt_shape)

        patches_gt.append(gt_patch[np.newaxis, ...])
        patches_lr.append(lr_patch[np.newaxis, ...])
        patches_mid.append(mid_patch[np.newaxis, ...])

    return patches_gt, patches_lr, patches_mid


def generate_training_data(
    hr_path,
    lr_path,
    output_file,
    patch_size,
    factor=1,
    overlap=0.5,
    mr_path=None,
):
    del factor  # The paired volumes already define the true scale relationship.

    hr_files = _scan_tiff_files(hr_path)
    lr_files = _scan_tiff_files(lr_path)

    gt_patches = []
    lr_patches = []
    mid_patches = []

    if mr_path:
        mr_files = _scan_tiff_files(mr_path)
        file_triplets = _pair_triplets(hr_files, lr_files, mr_files)
        for hr_file, lr_file, mr_file in file_triplets:
            gt_volume = _normalize(_read_volume(hr_file))
            lr_volume = _normalize(_read_volume(lr_file))
            mr_volume = _normalize(_read_volume(mr_file))
            current_gt, current_lr, current_mid = _extract_patches_with_mid(gt_volume, lr_volume, mr_volume, patch_size, overlap)
            gt_patches.extend(current_gt)
            lr_patches.extend(current_lr)
            mid_patches.extend(current_mid)
    else:
        file_pairs = _pair_files(hr_files, lr_files)
        for hr_file, lr_file in file_pairs:
            gt_volume = _normalize(_read_volume(hr_file))
            lr_volume = _normalize(_read_volume(lr_file))
            current_gt, current_lr, current_mid = _extract_patches(gt_volume, lr_volume, patch_size, overlap)
            gt_patches.extend(current_gt)
            lr_patches.extend(current_lr)
            mid_patches.extend(current_mid)

    if not gt_patches:
        raise ValueError("No training patches were generated. Check the input paths and patch size.")

    x = np.stack(gt_patches).astype(np.float32)
    y = np.stack(lr_patches).astype(np.float32)
    z = np.stack(mid_patches).astype(np.float32)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=x, Y=y, Z=z)
    return x, y, z
