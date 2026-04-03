"""Multi-head segmentation training script.

Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --resume checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import MultiHeadSegModel
from dataloader import SegmentationDataset, MultiDatasetLoader, SimpleValLoader, train_val_split
from utils.losses import MultiTaskLoss
from utils.metrics import compute_iou
from utils.visualize import VisualQueue


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: tuple[int, int], is_train: bool):
    """Build albumentations transforms. Returns None if albumentations is missing."""
    try:
        import albumentations as A
    except ImportError:
        return None

    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
        ])
    return None


def build_datasets(cfg: dict):
    """Build per-task datasets with train/val splits.

    Returns:
        train_datasets: list of Subset (one per task, for training).
        val_datasets:   list of Subset (one per task, for validation).
        num_classes_map: dict task_id → num_classes.
    """
    train_datasets = []
    val_datasets = []
    num_classes_map = {}
    img_size = tuple(cfg["training"]["img_size"])
    train_transform = build_transforms(img_size, is_train=True)
    val_ratio = cfg["training"].get("val_ratio", 0.15)
    seed = cfg["training"]["seed"]

    for ds_cfg in cfg["datasets"]:
        full_ds = SegmentationDataset(
            root=ds_cfg["root"],
            task_id=ds_cfg["task_id"],
            img_size=img_size,
            transform=train_transform,
        )
        train_sub, val_sub = train_val_split(full_ds, val_ratio=val_ratio, seed=seed)
        train_datasets.append(train_sub)
        val_datasets.append(val_sub)
        num_classes_map[ds_cfg["task_id"]] = ds_cfg["num_classes"]
        print(f"  Task {ds_cfg['task_id']} ({ds_cfg['name']}): "
              f"{len(train_sub)} train / {len(val_sub)} val, "
              f"{ds_cfg['num_classes']} classes")

    return train_datasets, val_datasets, num_classes_map


def validate(model, val_loader, criterion, num_classes_map, device):
    model.eval()
    task_ious: dict[int, list] = {tid: [] for tid in num_classes_map}
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks, task_ids in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            task_ids = task_ids.to(device)

            preds = model(images, task_ids)
            loss, _ = criterion(preds, masks, task_ids)
            val_loss += loss.item()
            num_batches += 1

            for tid, logits in preds.items():
                pred_cls = logits.argmax(dim=1)
                tid_mask = task_ids == tid
                iou = compute_iou(pred_cls.cpu(), masks[tid_mask].cpu(),
                                  num_classes_map[tid])
                task_ious[tid].append(iou)

    val_loss /= max(num_batches, 1)
    mean_ious = {}
    for tid, iou_list in task_ious.items():
        if iou_list:
            all_ious = np.stack(iou_list)
            mean_ious[tid] = float(np.nanmean(all_ious))
        else:
            mean_ious[tid] = 0.0

    return val_loss, mean_ious


def train(cfg: dict, resume_path: str | None = None):
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(tcfg["seed"])
    print(f"Device: {device}")

    # --- Datasets ---
    print("Building datasets...")
    train_datasets, val_datasets, num_classes_map = build_datasets(cfg)
    train_loader = MultiDatasetLoader(
        train_datasets,
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        seed=tcfg["seed"],
    )
    val_loader = SimpleValLoader(
        val_datasets,
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
    )
    print(f"  Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model ---
    mcfg = cfg["model"]
    model = MultiHeadSegModel(
        backbone_name=mcfg["backbone"],
        num_classes=num_classes_map,
        decoder_channels=mcfg["decoder_channels"],
        head_hidden=mcfg["head_hidden"],
        pretrained=mcfg["pretrained"],
        shared_decoder=mcfg["shared_decoder"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Parameters: {total_params:.1f}M total, {trainable:.1f}M trainable")

    # --- Loss & Optimizer ---
    criterion = MultiTaskLoss(
        task_ids=sorted(num_classes_map.keys()),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    warmup_epochs = tcfg.get("warmup_epochs", 5)
    total_epochs = tcfg["epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        if tcfg["lr_scheduler"] == "cosine":
            return 0.5 * (1 + np.cos(np.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_miou = 0.0

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Logging ---
    save_dir = Path(tcfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tcfg["log_dir"])

    # --- Visual Queue ---
    vis_cfg = tcfg.get("visualization", {})
    task_names = {d["task_id"]: d["name"] for d in cfg["datasets"]}
    vis_queue = VisualQueue(
        max_size=vis_cfg.get("queue_size", 5),
        save_dir=vis_cfg.get("save_dir", "vis"),
        task_names=task_names,
        num_classes=num_classes_map,
        cell_height=vis_cfg.get("cell_height", 192),
    )
    vis_interval = vis_cfg.get("interval_batches", 50)

    # --- Training Loop ---
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    global_step = start_epoch * max(len(train_loader), 1)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        criterion.train()
        train_loader.update_epoch(epoch)

        epoch_loss = 0.0
        task_losses: dict[int, float] = {tid: 0.0 for tid in num_classes_map}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for images, masks, task_ids in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            task_ids = task_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                predictions = model(images, task_ids)
                loss, per_task = criterion(predictions, masks, task_ids)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            for tid, l in per_task.items():
                task_losses[tid] += l
            num_batches += 1
            global_step += 1

            if num_batches % vis_interval == 0:
                model.eval()
                with torch.no_grad():
                    vis_queue.push_batch_last(
                        model, images, masks, predictions, task_ids,
                        epoch=epoch, batch_idx=num_batches,
                    )
                    vis_queue.save(writer=writer, global_step=global_step)
                model.train()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # --- Epoch Logging ---
        avg_loss = epoch_loss / max(num_batches, 1)
        writer.add_scalar("train/total_loss", avg_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        log_parts = [f"Epoch {epoch+1} | loss={avg_loss:.4f}"]
        for tid in sorted(task_losses):
            tl = task_losses[tid] / max(num_batches, 1)
            log_var = criterion.log_vars[str(tid)].item()
            writer.add_scalar(f"train/loss_task_{tid}", tl, epoch)
            writer.add_scalar(f"train/log_var_task_{tid}", log_var, epoch)
            log_parts.append(f"T{tid}={tl:.4f}(w={np.exp(-log_var):.2f})")
        print(" | ".join(log_parts))

        # --- Validation ---
        if (epoch + 1) % tcfg.get("val_interval", 5) == 0:
            val_loss, mean_ious = validate(
                model, val_loader, criterion, num_classes_map, device,
            )
            writer.add_scalar("val/loss", val_loss, epoch)
            overall_miou = np.mean(list(mean_ious.values()))
            writer.add_scalar("val/mIoU", overall_miou, epoch)
            for tid, miou in mean_ious.items():
                writer.add_scalar(f"val/mIoU_task_{tid}", miou, epoch)

            print(f"  Val loss={val_loss:.4f} | "
                  + " | ".join(f"T{t} mIoU={v:.4f}" for t, v in mean_ious.items())
                  + f" | Overall mIoU={overall_miou:.4f}")

            if tcfg.get("save_best", True) and overall_miou > best_miou:
                best_miou = overall_miou
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "config": cfg,
                }, save_dir / "best.pt")
                print(f"  Saved best model (mIoU={best_miou:.4f})")

        # --- Save Last ---
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": cfg,
        }, save_dir / "last.pt")

    writer.close()
    print(f"Training complete. Best mIoU: {best_miou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Head Segmentation Training")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
