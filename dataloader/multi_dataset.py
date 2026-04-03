from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SegmentationDataset(Dataset):
    """Single-task segmentation dataset.

    Expects:
        root/images/  — input images (png, jpg, etc.)
        root/masks/   — label masks (single-channel, pixel value = class id)

    Images and masks are paired by matching filename stems (ignoring
    extensions), so `images/001.jpg` pairs with `masks/001.png`.

    Args:
        root:        path to dataset directory.
        task_id:     integer identifier for this task/head.
        img_size:    (H, W) to resize all images and masks.
        transform:   optional albumentations transform.
    """

    def __init__(
        self,
        root: str | Path,
        task_id: int,
        img_size: tuple[int, int] = (512, 512),
        transform=None,
    ):
        self.root = Path(root)
        self.task_id = task_id
        self.img_size = img_size
        self.transform = transform

        img_dir = self.root / "images"
        mask_dir = self.root / "masks"

        img_by_stem = {p.stem: p for p in sorted(img_dir.iterdir()) if p.is_file()}
        mask_by_stem = {p.stem: p for p in sorted(mask_dir.iterdir()) if p.is_file()}

        common_stems = sorted(set(img_by_stem) & set(mask_by_stem))
        assert len(common_stems) > 0, (
            f"No matching image/mask pairs found in {root}. "
            f"Images: {len(img_by_stem)}, Masks: {len(mask_by_stem)}"
        )

        orphan_imgs = set(img_by_stem) - set(mask_by_stem)
        orphan_masks = set(mask_by_stem) - set(img_by_stem)
        if orphan_imgs or orphan_masks:
            print(f"  WARNING [{root}]: {len(orphan_imgs)} images without masks, "
                  f"{len(orphan_masks)} masks without images (skipped)")

        self.images = [img_by_stem[s] for s in common_stems]
        self.masks = [mask_by_stem[s] for s in common_stems]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size[1], self.img_size[0]),
                         interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        mask = torch.from_numpy(mask).long()

        return img, mask, self.task_id


class BalancedMultiTaskSampler(Sampler):
    """Yields indices that guarantee equal samples from every task per batch.

    With batch_size=B and T tasks, each batch gets B//T samples per task.
    Smaller datasets are cycled; larger datasets are subsampled per epoch.
    """

    def __init__(self, dataset_sizes: list[int], samples_per_task: int,
                 num_batches: int, seed: int = 42):
        self.dataset_sizes = dataset_sizes
        self.num_tasks = len(dataset_sizes)
        self.samples_per_task = samples_per_task
        self.num_batches = num_batches
        self.base_seed = seed
        self.epoch = 0
        self.offsets = []
        offset = 0
        for s in dataset_sizes:
            self.offsets.append(offset)
            offset += s

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.base_seed + self.epoch)
        per_task_pools = []
        for i, size in enumerate(self.dataset_sizes):
            pool = np.arange(size)
            rng.shuffle(pool)
            total_needed = self.samples_per_task * self.num_batches
            repeats = (total_needed // size) + 1
            pool = np.tile(pool, repeats)[:total_needed]
            pool += self.offsets[i]
            per_task_pools.append(pool)

        for b in range(self.num_batches):
            batch_indices = []
            for i in range(self.num_tasks):
                start = b * self.samples_per_task
                end = start + self.samples_per_task
                batch_indices.extend(per_task_pools[i][start:end].tolist())
            rng.shuffle(batch_indices)
            yield from batch_indices

    def __len__(self) -> int:
        return self.num_batches * self.samples_per_task * self.num_tasks


class MultiDatasetLoader:
    """Convenience wrapper that creates a balanced DataLoader across tasks.

    Args:
        datasets:         list of SegmentationDataset (one per task).
        batch_size:       total batch size (split equally across tasks).
        num_workers:      DataLoader workers.
        num_batches_per_epoch: how many batches constitute one epoch.
        seed:             random seed for reproducibility.
    """

    def __init__(
        self,
        datasets: list[SegmentationDataset],
        batch_size: int = 16,
        num_workers: int = 4,
        num_batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        self.combined = torch.utils.data.ConcatDataset(datasets)
        self.num_tasks = len(datasets)

        if batch_size % self.num_tasks != 0:
            old_bs = batch_size
            batch_size = (batch_size // self.num_tasks) * self.num_tasks
            print(f"  WARNING: batch_size={old_bs} not divisible by {self.num_tasks} tasks. "
                  f"Rounded down to {batch_size} for balanced sampling.")

        samples_per_task = batch_size // self.num_tasks
        assert samples_per_task >= 1, (
            f"batch_size ({batch_size}) must be >= num_tasks ({self.num_tasks})"
        )

        sizes = [len(d) for d in datasets]
        if num_batches_per_epoch is None:
            num_batches_per_epoch = max(sizes) // samples_per_task

        self.sampler = BalancedMultiTaskSampler(
            sizes, samples_per_task, num_batches_per_epoch, seed=seed,
        )

        self.loader = DataLoader(
            self.combined,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def update_epoch(self, epoch: int):
        """Call at the start of each epoch to reshuffle."""
        self.sampler.set_epoch(epoch)


def train_val_split(
    dataset: SegmentationDataset,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Split a single SegmentationDataset into train and val subsets.

    Both subsets preserve the original task_id.  Uses a fixed random
    seed so splits are reproducible across runs.
    """
    n = len(dataset)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    val_size = max(1, int(n * val_ratio))
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class SimpleValLoader:
    """Simple shuffled DataLoader for validation (no balanced sampling needed)."""

    def __init__(self, datasets: list, batch_size: int = 16, num_workers: int = 4):
        self.combined = torch.utils.data.ConcatDataset(datasets)
        self.loader = DataLoader(
            self.combined,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
