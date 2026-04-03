"""Live visualization queue for training monitoring.

Maintains a FIFO queue of the last N samples showing:
  [Input Image | GT Mask | Head 0 Pred | Head 1 Pred | Head 2 Pred | Head 3 Pred]

All 4 head predictions are shown for every image so you can compare
how each head interprets the same input.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _generate_colormap(max_classes: int = 256) -> np.ndarray:
    """Deterministic colormap with high contrast between adjacent classes."""
    cmap = np.zeros((max_classes, 3), dtype=np.uint8)
    for i in range(max_classes):
        r, g, b = 0, 0, 0
        idx = i
        for j in range(8):
            r |= ((idx >> 0) & 1) << (7 - j)
            g |= ((idx >> 1) & 1) << (7 - j)
            b |= ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        cmap[i] = [r, g, b]
    cmap[0] = [0, 0, 0]
    return cmap


COLORMAP = _generate_colormap()


def colorize_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert single-channel class-id mask to RGB using the colormap."""
    h, w = mask.shape
    indices = mask.astype(np.int32).flatten() % 256
    rgb = COLORMAP[indices].reshape(h, w, 3)
    return rgb


def _tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """(C, H, W) normalized tensor → (H, W, 3) uint8 BGR for OpenCV.

    Undoes ImageNet normalization before converting.
    """
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


class VisualQueue:
    """FIFO visualization queue showing ALL head predictions per image.

    Each row shows one sample with:
      [Task Tag | Input | GT Mask | Head0 Pred | Head1 Pred | Head2 Pred | Head3 Pred]

    So you can see how every head interprets the same image.

    Args:
        max_size:      how many recent samples to keep (default 5).
        save_dir:      directory to write the live grid image.
        task_names:    optional dict task_id → display name.
        num_classes:   dict task_id → num_classes (for mask coloring).
        cell_height:   pixel height of each row in the grid.
    """

    def __init__(
        self,
        max_size: int = 5,
        save_dir: str = "vis",
        task_names: dict[int, str] | None = None,
        num_classes: dict[int, int] | None = None,
        cell_height: int = 192,
    ):
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_names = task_names or {}
        self.num_classes = num_classes or {}
        self.task_ids = sorted(self.num_classes.keys())
        self.cell_height = cell_height
        self.queue: deque[dict] = deque(maxlen=max_size)
        self._step = 0

    def push(
        self,
        image: torch.Tensor,
        gt_mask: torch.Tensor,
        all_head_logits: dict[int, torch.Tensor],
        gt_task_id: int,
        epoch: int,
        batch_idx: int,
    ):
        """Add one sample with ALL head predictions to the queue.

        Args:
            image:           (C, H, W) float tensor (ImageNet-normalized).
            gt_mask:         (H, W) long tensor of class ids.
            all_head_logits: dict task_id → (num_classes, H, W) logits from each head.
            gt_task_id:      which task this image actually belongs to.
            epoch:           current epoch number.
            batch_idx:       current batch index.
        """
        img_bgr = _tensor_to_numpy_image(image)
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        gt_nc = self.num_classes.get(gt_task_id, 256)

        head_preds = {}
        for tid in self.task_ids:
            if tid in all_head_logits:
                pred_cls = all_head_logits[tid].argmax(dim=0).cpu().numpy().astype(np.uint8)
                nc = self.num_classes.get(tid, 256)
                head_preds[tid] = colorize_mask(pred_cls, nc)
            else:
                h, w = gt_np.shape
                head_preds[tid] = np.zeros((h, w, 3), dtype=np.uint8)

        self.queue.append({
            "image": img_bgr,
            "gt_color": colorize_mask(gt_np, gt_nc),
            "head_preds": head_preds,
            "gt_task_id": gt_task_id,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "step": self._step,
        })
        self._step += 1

    def render(self) -> np.ndarray:
        """Compose the current queue into a single grid image.

        Layout per row:
            [Tag | Input | GT Mask | Head0 | Head1 | Head2 | Head3]
        """
        if not self.queue:
            placeholder = np.zeros((self.cell_height, 800, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for predictions...",
                        (10, self.cell_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            return placeholder

        cell_h = self.cell_height
        sep_w = 2
        tag_w = 140
        n_heads = len(self.task_ids)

        rows = []
        for entry in self.queue:
            img = cv2.resize(entry["image"], (cell_h, cell_h))
            gt = cv2.resize(entry["gt_color"], (cell_h, cell_h),
                            interpolation=cv2.INTER_NEAREST)

            sep = np.full((cell_h, sep_w, 3), 80, dtype=np.uint8)
            parts = [img, sep, gt]

            for tid in self.task_ids:
                pred_color = entry["head_preds"].get(tid,
                    np.zeros((cell_h, cell_h, 3), dtype=np.uint8))
                pred_resized = cv2.resize(pred_color, (cell_h, cell_h),
                                          interpolation=cv2.INTER_NEAREST)
                if tid == entry["gt_task_id"]:
                    border = np.full((cell_h, cell_h, 3), 0, dtype=np.uint8)
                    b = 3
                    border[b:-b, b:-b] = pred_resized[b:-b, b:-b]
                    border[:b, :, :] = [0, 255, 0]
                    border[-b:, :, :] = [0, 255, 0]
                    border[:, :b, :] = [0, 255, 0]
                    border[:, -b:, :] = [0, 255, 0]
                    pred_resized = border
                parts.extend([sep.copy(), pred_resized])

            row = np.concatenate(parts, axis=1)

            tag = np.zeros((cell_h, tag_w, 3), dtype=np.uint8)
            gt_name = self.task_names.get(entry["gt_task_id"], f"Task {entry['gt_task_id']}")
            cv2.putText(tag, "GT:", (6, cell_h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.putText(tag, gt_name, (6, cell_h // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(tag, f"E{entry['epoch']+1} B{entry['batch_idx']}",
                        (6, cell_h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            row = np.concatenate([tag, row], axis=1)
            rows.append(row)

        row_sep = np.full((2, rows[0].shape[1], 3), 40, dtype=np.uint8)
        parts = []
        for i, row in enumerate(rows):
            if i > 0:
                parts.append(row_sep)
            parts.append(row)

        header_h = 40
        total_w = rows[0].shape[1]
        header = np.zeros((header_h, total_w, 3), dtype=np.uint8)

        x = tag_w
        for lbl in ["Input", "GT Mask"]:
            cv2.putText(header, lbl, (x + 5, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x += cell_h + sep_w

        for tid in self.task_ids:
            name = self.task_names.get(tid, f"Head {tid}")
            nc = self.num_classes.get(tid, "?")
            label = f"{name} ({nc}c)"
            cv2.putText(header, label, (x + 5, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
            x += cell_h + sep_w

        cv2.putText(header, "Source", (6, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        legend_h = 24
        legend = np.zeros((legend_h, total_w, 3), dtype=np.uint8)
        cv2.putText(legend, "Green border = correct head for this image",
                    (tag_w + 5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        grid = np.concatenate([header] + parts + [legend], axis=0)
        return grid

    def save(self, writer=None, global_step: int | None = None):
        """Render and save the grid to disk (and optionally to TensorBoard)."""
        grid = self.render()
        cv2.imwrite(str(self.save_dir / "live_queue.png"), grid)

        if writer is not None and global_step is not None:
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
            grid_tensor = torch.from_numpy(grid_rgb).permute(2, 0, 1)
            writer.add_image("train/visual_queue", grid_tensor, global_step)

    def push_batch_last(
        self,
        model,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: dict[int, torch.Tensor],
        dataset_ids: torch.Tensor,
        epoch: int,
        batch_idx: int,
    ):
        """Pick one image per active task, run ALL heads on it, push to queue.

        For each task present in the batch, takes the last image,
        runs it through ALL heads so we can visualize all predictions.
        """
        for tid in predictions:
            tid_mask = dataset_ids == tid
            indices = tid_mask.nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                continue

            last_idx = indices[-1].item()
            single_img = images[last_idx:last_idx + 1]

            all_head_logits = {}
            for head_tid in self.task_ids:
                head_logits = model.forward_single_task(single_img, head_tid)
                all_head_logits[head_tid] = head_logits.squeeze(0).detach()

            self.push(
                image=images[last_idx],
                gt_mask=masks[last_idx],
                all_head_logits=all_head_logits,
                gt_task_id=tid,
                epoch=epoch,
                batch_idx=batch_idx,
            )
