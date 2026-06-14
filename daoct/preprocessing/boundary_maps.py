import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation, disk


def compute_boundary_distance_maps(seg_mask: np.ndarray, num_classes: int = 10):
    dist_maps = np.zeros((num_classes, *seg_mask.shape), dtype=np.float32)
    for c in range(num_classes):
        binary = (seg_mask == c).astype(np.uint8)
        dist_maps[c] = distance_transform_edt(binary) + distance_transform_edt(1 - binary)
        dist_maps[c] /= dist_maps[c].max() + 1e-8
    return dist_maps


def interlayer_boundary_map(seg_mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(seg_mask, dtype=np.float32)
    for c in range(1, 10):
        layer_c = (seg_mask == c).astype(np.uint8)
        layer_cm1 = (seg_mask == c - 1).astype(np.uint8)
        interface = dilation(layer_c, disk(1)) & layer_cm1
        interface |= dilation(layer_cm1, disk(1)) & layer_c
        boundaries += interface
    return np.clip(boundaries, 0, 1)
