#!/usr/bin/env python3
"""Download real segmentation datasets and prepare 10-image samples.

Uses:
  - Pascal VOC 2012 (21 classes) → dataset_01, dataset_02, dataset_03
  - Oxford-IIIT Pets (3-class trimap) → dataset_04

dataset_01: VOC full 21-class segmentation
dataset_02: VOC binarized (background=0, foreground=1)
dataset_03: VOC remapped to 8 super-categories
dataset_04: Oxford Pets 3-class trimap (pet / background / border)
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torchvision

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path("data")
CACHE_DIR = Path("data_cache")
N_SAMPLES = 10

VOC_SUPERCLASS_MAP = np.zeros(256, dtype=np.uint8)
VOC_SUPERCLASS_MAP[0] = 0    # background
VOC_SUPERCLASS_MAP[15] = 1   # person
VOC_SUPERCLASS_MAP[1] = 2    # aeroplane → vehicle
VOC_SUPERCLASS_MAP[2] = 2    # bicycle → vehicle
VOC_SUPERCLASS_MAP[4] = 2    # boat → vehicle
VOC_SUPERCLASS_MAP[6] = 2    # bus → vehicle
VOC_SUPERCLASS_MAP[7] = 2    # car → vehicle
VOC_SUPERCLASS_MAP[14] = 2   # motorbike → vehicle
VOC_SUPERCLASS_MAP[19] = 2   # train → vehicle
VOC_SUPERCLASS_MAP[3] = 3    # bird → animal
VOC_SUPERCLASS_MAP[8] = 3    # cat → animal
VOC_SUPERCLASS_MAP[10] = 3   # cow → animal
VOC_SUPERCLASS_MAP[12] = 3   # dog → animal
VOC_SUPERCLASS_MAP[13] = 3   # horse → animal
VOC_SUPERCLASS_MAP[17] = 3   # sheep → animal
VOC_SUPERCLASS_MAP[5] = 4    # bottle → object
VOC_SUPERCLASS_MAP[9] = 5    # chair → furniture
VOC_SUPERCLASS_MAP[11] = 5   # dining table → furniture
VOC_SUPERCLASS_MAP[18] = 5   # sofa → furniture
VOC_SUPERCLASS_MAP[16] = 6   # potted plant → plant
VOC_SUPERCLASS_MAP[20] = 7   # tv/monitor → electronics
VOC_SUPERCLASS_MAP[255] = 0  # void → background


def save_pair(img_pil: Image.Image, mask_np: np.ndarray,
              images_dir: Path, masks_dir: Path, name: str):
    img_pil.convert("RGB").save(images_dir / f"{name}.png")
    Image.fromarray(mask_np.astype(np.uint8)).save(masks_dir / f"{name}.png")


def prepare_voc21(ds, indices, out_dir: Path):
    """VOC 21-class segmentation (classes 0-20)."""
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        img, mask = ds[idx]
        mask_np = np.array(mask)
        mask_np[mask_np == 255] = 0
        save_pair(img, mask_np, images_dir, masks_dir, f"{i:04d}")

    print(f"  dataset_01 ({out_dir}): {len(indices)} images, 21 classes [VOC full]")


def prepare_voc_binary(ds, indices, out_dir: Path):
    """VOC binarized: 0=background, 1=any foreground object."""
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        img, mask = ds[idx]
        mask_np = np.array(mask)
        mask_np = ((mask_np > 0) & (mask_np < 255)).astype(np.uint8)
        save_pair(img, mask_np, images_dir, masks_dir, f"{i:04d}")

    print(f"  dataset_02 ({out_dir}): {len(indices)} images, 2 classes [VOC binary]")


def prepare_voc8(ds, indices, out_dir: Path):
    """VOC remapped to 8 super-categories."""
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(indices):
        img, mask = ds[idx]
        mask_np = VOC_SUPERCLASS_MAP[np.array(mask)]
        save_pair(img, mask_np, images_dir, masks_dir, f"{i:04d}")

    print(f"  dataset_03 ({out_dir}): {len(indices)} images, 8 classes [VOC super-categories]")


def prepare_pets(out_dir: Path):
    """Oxford-IIIT Pets 3-class trimap: 0=pet, 1=background, 2=border."""
    print("Downloading Oxford-IIIT Pets...")
    ds = torchvision.datasets.OxfordIIITPet(
        root=str(CACHE_DIR), split="trainval", download=True,
        target_types="segmentation",
    )

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(ds)), N_SAMPLES)
    for i, idx in enumerate(indices):
        img, mask = ds[idx]
        mask_np = np.array(mask) - 1  # 1-indexed → 0-indexed
        mask_np = np.clip(mask_np, 0, 2)
        save_pair(img, mask_np, images_dir, masks_dir, f"{i:04d}")

    print(f"  dataset_04 ({out_dir}): {N_SAMPLES} images, 3 classes [Oxford Pets trimap]")


def main():
    print("=" * 60)
    print("Downloading and preparing real segmentation datasets")
    print("=" * 60)

    print("\nDownloading Pascal VOC 2012...")
    voc = torchvision.datasets.VOCSegmentation(
        root=str(CACHE_DIR), year="2012", image_set="train", download=True,
    )
    print(f"  VOC 2012 train: {len(voc)} images total")

    all_indices = list(range(len(voc)))
    random.shuffle(all_indices)
    voc_indices_1 = all_indices[0:N_SAMPLES]
    voc_indices_2 = all_indices[N_SAMPLES:2 * N_SAMPLES]
    voc_indices_3 = all_indices[2 * N_SAMPLES:3 * N_SAMPLES]

    print("\nPreparing datasets:")
    prepare_voc21(voc, voc_indices_1, DATA_DIR / "dataset_01")
    prepare_voc_binary(voc, voc_indices_2, DATA_DIR / "dataset_02")
    prepare_voc8(voc, voc_indices_3, DATA_DIR / "dataset_03")
    prepare_pets(DATA_DIR / "dataset_04")

    print("\n" + "=" * 60)
    print("Done! Data structure:")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "images").exists():
            n_img = len(list((d / "images").iterdir()))
            n_mask = len(list((d / "masks").iterdir()))
            print(f"  {d}/  →  {n_img} images, {n_mask} masks")
    print("=" * 60)


if __name__ == "__main__":
    main()
