import torch
import numpy as np


def compute_iou(pred: torch.Tensor, target: torch.Tensor,
                num_classes: int, ignore_index: int = -1) -> np.ndarray:
    """Per-class IoU (Intersection over Union).

    Args:
        pred:   (B, H, W) predicted class indices.
        target: (B, H, W) ground truth class indices.
        num_classes: total number of classes.
        ignore_index: class to ignore.

    Returns:
        (num_classes,) array of per-class IoU values.
    """
    ious = np.zeros(num_classes, dtype=np.float64)
    valid = target != ignore_index

    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious[c] = intersection / union if union > 0 else float("nan")

    return ious


def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                 num_classes: int, ignore_index: int = -1) -> np.ndarray:
    """Per-class Dice coefficient."""
    dices = np.zeros(num_classes, dtype=np.float64)
    valid = target != ignore_index

    for c in range(num_classes):
        pred_c = (pred == c) & valid
        target_c = (target == c) & valid
        intersection = (pred_c & target_c).sum().item()
        total = pred_c.sum().item() + target_c.sum().item()
        dices[c] = (2.0 * intersection) / total if total > 0 else float("nan")

    return dices
