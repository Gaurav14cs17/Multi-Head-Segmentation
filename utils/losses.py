import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0
        one_hot = F.one_hot(targets_clean, num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid.unsqueeze(1).float()
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Cross-Entropy + Dice loss for segmentation.

    Args:
        dice_weight: weight for Dice loss component.
        ce_weight:   weight for CE loss component.
        ignore_index: class index to ignore.
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5,
                 ignore_index: int = -1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


class MultiTaskLoss(nn.Module):
    """Learnable multi-task loss weighting (Kendall et al., 2018).

    Learns a log-variance parameter per task so that noisier/harder tasks
    are automatically down-weighted during training.

    total_loss = sum_t [ (1 / 2*sigma_t^2) * L_t + log(sigma_t) ]
    """

    def __init__(self, task_ids: list[int], ignore_index: int = -1):
        super().__init__()
        self.task_ids = task_ids
        self.log_vars = nn.ParameterDict({
            str(tid): nn.Parameter(torch.zeros(1)) for tid in task_ids
        })
        self.seg_loss = CombinedSegLoss(ignore_index=ignore_index)

    def forward(
        self,
        predictions: dict[int, torch.Tensor],
        targets: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, float]]:
        """
        Args:
            predictions: dict task_id → (N_task, C, H, W) logits.
            targets:     (B, H, W) full batch ground truth.
            dataset_ids: (B,) task id per sample.

        Returns:
            total_loss, per_task_losses dict.
        """
        total_loss = torch.tensor(0.0, device=targets.device)
        per_task = {}

        for tid in predictions:
            mask = dataset_ids == tid
            task_targets = targets[mask]
            task_logits = predictions[tid]

            raw_loss = self.seg_loss(task_logits, task_targets)
            log_var = self.log_vars[str(tid)]
            weighted = torch.exp(-log_var) * raw_loss + log_var

            total_loss = total_loss + weighted
            per_task[tid] = raw_loss.item()

        return total_loss, per_task
