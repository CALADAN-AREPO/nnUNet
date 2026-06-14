import numpy as np
from scipy.ndimage import distance_transform_edt

SEEN_VENDORS = ["Maestro2", "Spectralis", "Cirrus"]
UNSEEN_VENDORS = ["Triton"]

ALPHA = 0.3
LAMBDA_PENALTY = 1.5
TAU = 0.02
BETA_MACULA = 0.5
BETA_WIDEFIELD = 0.5
NUM_CLASSES = 10


def dice_coefficient(pred_mask, gt_mask, smooth=1e-6):
    intersection = (pred_mask & gt_mask).sum()
    return (2.0 * intersection + smooth) / (pred_mask.sum() + gt_mask.sum() + smooth)


def extract_boundary(mask):
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_distance(pred_mask, gt_mask):
    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 100.0
    gt_dist = distance_transform_edt(~gt_boundary)
    pred_dist = distance_transform_edt(~pred_boundary)
    pred_to_gt = gt_dist[pred_boundary]
    gt_to_pred = pred_dist[gt_boundary]
    return (pred_to_gt.mean() + gt_to_pred.mean()) / 2.0


def masd_per_class(pred, gt, H):
    masd = []
    for c in range(NUM_CLASSES):
        d = surface_distance(pred == c, gt == c)
        if np.isnan(d) or d >= 100:
            masd.append(0.0)
        else:
            masd.append(d / H)
    return np.array(masd)


def image_score(pred, gt):
    H = pred.shape[0]
    dice_scores = np.array([dice_coefficient(pred == c, gt == c) for c in range(NUM_CLASSES)])
    masd = masd_per_class(pred, gt, H)
    masd_score = np.exp(-masd / TAU)
    layer_score = 0.5 * (dice_scores + masd_score)
    return layer_score.mean()


def compute_final_score(csv_data):
    """
    csv_data: list of dicts with keys: device, status, anatomy, image_score
    anatomy: 'Macula' | 'WideField'
    status: 'healthy' | 'diseased'
    device: vendor name
    """
    import pandas as pd
    from collections import defaultdict

    df = pd.DataFrame(csv_data)

    cohort_scores = df.groupby(["anatomy", "device", "status"])["image_score"].mean()

    vendor_scores = {}
    for (anatomy, device), group in cohort_scores.groupby(["anatomy", "device"]):
        healthy = group.get("healthy", 0.0)
        diseased = group.get("diseased", 0.0)
        vendor_scores[(anatomy, device)] = ALPHA * healthy + (1 - ALPHA) * diseased

    anatomy_scores = {}
    for anatomy in ["Macula", "WideField"]:
        anat_vendor_scores = {v: s for (a, v), s in vendor_scores.items() if a == anatomy}
        if not anat_vendor_scores:
            anatomy_scores[anatomy] = 0.0
            continue
        overall = np.mean(list(anat_vendor_scores.values()))
        seen_vals = [anat_vendor_scores[v] for v in SEEN_VENDORS if v in anat_vendor_scores]
        seen_mean = np.mean(seen_vals) if seen_vals else 0.0
        penalties = [max(0.0, seen_mean - anat_vendor_scores[v]) for v in UNSEEN_VENDORS if v in anat_vendor_scores]
        penalty = np.mean(penalties) if penalties else 0.0
        anatomy_scores[anatomy] = max(0.0, overall - LAMBDA_PENALTY * penalty)

    final_score = BETA_MACULA * anatomy_scores.get("Macula", 0.0) + BETA_WIDEFIELD * anatomy_scores.get("WideField", 0.0)
    return final_score, anatomy_scores
