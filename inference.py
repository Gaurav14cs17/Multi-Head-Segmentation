"""Multi-head segmentation inference script.

Usage:
    # AUTO-DETECT: model decides which head to use (no label needed)
    python inference.py --checkpoint checkpoints/best.pt --task auto --input image.png --output out/

    # AUTO-DETECT on a whole directory
    python inference.py --checkpoint checkpoints/best.pt --task auto --input images/ --output out/ --overlay

    # Manual: specify which task head to use
    python inference.py --checkpoint checkpoints/best.pt --task 0 --input image.png --output out/

    # Entire directory of images (manual)
    python inference.py --checkpoint checkpoints/best.pt --task 2 --input data/dataset_03/images/ --output out/

    # With overlay blending
    python inference.py --checkpoint checkpoints/best.pt --task 1 --input image.png --output out/ --overlay --alpha 0.5

    # Export to ONNX
    python inference.py --checkpoint checkpoints/best.pt --task 0 --export-onnx model_task0.onnx
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import MultiHeadSegModel
from utils.visualize import colorize_mask


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[MultiHeadSegModel, dict]:
    """Load model from checkpoint and return (model, config)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    mcfg = cfg["model"]
    num_classes = {d["task_id"]: d["num_classes"] for d in cfg["datasets"]}

    model = MultiHeadSegModel(
        backbone_name=mcfg["backbone"],
        num_classes=num_classes,
        decoder_channels=mcfg["decoder_channels"],
        head_hidden=mcfg["head_hidden"],
        pretrained=False,
        shared_decoder=mcfg["shared_decoder"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt['epoch']+1}, "
          f"mIoU={ckpt.get('best_miou', 0):.4f})")
    print(f"Available tasks: {list(num_classes.keys())} → classes {num_classes}")

    return model, cfg


def preprocess(
    image_path: str,
    img_size: tuple[int, int],
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int]]:
    """Read and preprocess a single image.

    Returns:
        tensor:       (1, 3, H, W) float32 normalized input.
        original_bgr: original image in BGR for overlay.
        orig_size:    (H, W) of the original image.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_size = (img_bgr.shape[0], img_bgr.shape[1])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size[1], img_size[0]),
                             interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, img_bgr, orig_size


@torch.no_grad()
def predict(
    model: MultiHeadSegModel,
    tensor: torch.Tensor,
    task_id: int | None,
    device: torch.device,
    orig_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, dict[int, float] | None]:
    """Run inference and return the class-id mask at original resolution.

    Args:
        task_id: head to use.  None = auto-detect (run all heads, pick best).

    Returns:
        pred:         (H, W) uint8 array of predicted class indices.
        used_task_id: the task head that was actually used.
        confidences:  per-head confidence scores (only when auto-detect).
    """
    tensor = tensor.to(device)

    if task_id is not None:
        logits = model.forward_single_task(tensor, task_id)
        used_task_id = task_id
        confidences = None
    else:
        logits, used_task_id, confidences = model.forward_auto_detect(tensor)

    if orig_size is not None and (tuple(logits.shape[2:]) != orig_size):
        logits = F.interpolate(
            logits, size=orig_size, mode="bilinear", align_corners=False,
        )

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, used_task_id, confidences


def save_results(
    pred_mask: np.ndarray,
    original_bgr: np.ndarray,
    num_classes: int,
    output_path: str,
    overlay: bool = False,
    alpha: float = 0.5,
):
    """Save colored mask (and optional overlay) to disk."""
    color_mask = colorize_mask(pred_mask, num_classes)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    stem = Path(output_path).stem
    parent = Path(output_path).parent

    cv2.imwrite(str(parent / f"{stem}_mask.png"), color_mask_bgr)
    cv2.imwrite(str(parent / f"{stem}_class_ids.png"), pred_mask)

    if overlay:
        orig_resized = cv2.resize(original_bgr,
                                  (pred_mask.shape[1], pred_mask.shape[0]))
        blended = cv2.addWeighted(orig_resized, 1 - alpha, color_mask_bgr, alpha, 0)
        cv2.imwrite(str(parent / f"{stem}_overlay.png"), blended)


def run_single(
    model: MultiHeadSegModel,
    image_path: str,
    task_id: int | None,
    num_classes_map: dict[int, int],
    img_size: tuple[int, int],
    output_dir: str,
    device: torch.device,
    task_names: dict[int, str] | None = None,
    overlay: bool = False,
    alpha: float = 0.5,
):
    """Run inference on a single image (with optional auto-detect)."""
    task_names = task_names or {}
    tensor, original_bgr, orig_size = preprocess(image_path, img_size)

    t0 = time.time()
    pred_mask, used_tid, confidences = predict(
        model, tensor, task_id, device, orig_size,
    )
    dt = (time.time() - t0) * 1000

    nc = num_classes_map[used_tid]
    out_name = Path(image_path).stem
    save_results(pred_mask, original_bgr, nc,
                 os.path.join(output_dir, out_name),
                 overlay=overlay, alpha=alpha)

    unique_cls = np.unique(pred_mask)
    name = task_names.get(used_tid, f"Task {used_tid}")

    if confidences is not None:
        conf_str = "  ".join(
            f"{task_names.get(t, f'T{t}')}: {c:.3f}" for t, c in sorted(confidences.items())
        )
        print(f"  {Path(image_path).name} → {dt:.1f}ms | "
              f"AUTO-DETECTED: {name} (task {used_tid}) | "
              f"classes: {unique_cls.tolist()}")
        print(f"    confidence scores: {conf_str}")
    else:
        print(f"  {Path(image_path).name} → {dt:.1f}ms | "
              f"head: {name} | classes: {unique_cls.tolist()}")


def run_directory(
    model: MultiHeadSegModel,
    input_dir: str,
    task_id: int | None,
    num_classes_map: dict[int, int],
    img_size: tuple[int, int],
    output_dir: str,
    device: torch.device,
    task_names: dict[int, str] | None = None,
    overlay: bool = False,
    alpha: float = 0.5,
    batch_size: int = 4,
):
    """Run inference on all images in a directory.

    When task_id is None (auto-detect), processes images one by one
    since each image may be routed to a different head.
    """
    task_names = task_names or {}
    image_paths = sorted([
        str(p) for p in Path(input_dir).iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS
    ])
    print(f"Found {len(image_paths)} images in {input_dir}")

    if task_id is None:
        print("Mode: AUTO-DETECT (each image routed to best head)")
        task_counts: dict[int, int] = {}
        total_time = 0.0
        for idx, p in enumerate(image_paths):
            tensor, original_bgr, orig_size = preprocess(p, img_size)

            t0 = time.time()
            pred_mask, used_tid, confidences = predict(
                model, tensor, None, device, orig_size,
            )
            dt = time.time() - t0
            total_time += dt

            nc = num_classes_map[used_tid]
            out_name = Path(p).stem
            save_results(pred_mask, original_bgr, nc,
                         os.path.join(output_dir, out_name),
                         overlay=overlay, alpha=alpha)

            task_counts[used_tid] = task_counts.get(used_tid, 0) + 1
            name = task_names.get(used_tid, f"T{used_tid}")
            print(f"  [{idx+1}/{len(image_paths)}] {Path(p).name} → "
                  f"{name} ({dt*1000:.1f}ms)")

        fps = len(image_paths) / total_time if total_time > 0 else 0
        print(f"\nDone: {len(image_paths)} images in {total_time:.2f}s ({fps:.1f} FPS)")
        print("Task distribution:")
        for tid, count in sorted(task_counts.items()):
            name = task_names.get(tid, f"Task {tid}")
            print(f"  {name}: {count} images")
        return

    nc = num_classes_map[task_id]
    total_time = 0.0
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        tensors, originals, orig_sizes = [], [], []

        for p in batch_paths:
            t, orig, sz = preprocess(p, img_size)
            tensors.append(t)
            originals.append(orig)
            orig_sizes.append(sz)

        batch_tensor = torch.cat(tensors, dim=0).to(device)

        t0 = time.time()
        logits = model.forward_single_task(batch_tensor, task_id)
        dt = time.time() - t0
        total_time += dt

        for j, p in enumerate(batch_paths):
            single_logits = logits[j : j + 1]
            if tuple(single_logits.shape[2:]) != orig_sizes[j]:
                single_logits = F.interpolate(
                    single_logits, size=orig_sizes[j],
                    mode="bilinear", align_corners=False,
                )
            pred = single_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            out_name = Path(p).stem
            save_results(pred, originals[j], nc,
                         os.path.join(output_dir, out_name),
                         overlay=overlay, alpha=alpha)

        processed = min(i + batch_size, len(image_paths))
        print(f"  [{processed}/{len(image_paths)}] batch {dt*1000:.1f}ms")

    fps = len(image_paths) / total_time if total_time > 0 else 0
    print(f"Done: {len(image_paths)} images in {total_time:.2f}s ({fps:.1f} FPS)")


def export_onnx(
    model: MultiHeadSegModel,
    task_id: int,
    img_size: tuple[int, int],
    output_path: str,
    device: torch.device,
):
    """Export a single task head to ONNX format."""

    class SingleTaskWrapper(torch.nn.Module):
        def __init__(self, model, task_id):
            super().__init__()
            self.model = model
            self.task_id = task_id

        def forward(self, x):
            return self.model.forward_single_task(x, self.task_id)

    wrapper = SingleTaskWrapper(model, task_id).to(device)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    torch.onnx.export(
        wrapper, dummy, output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX: {output_path} (task {task_id}, "
          f"input shape: [B, 3, {img_size[0]}, {img_size[1]}])")


def main():
    parser = argparse.ArgumentParser(description="Multi-Head Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt)")
    parser.add_argument("--task", type=str, required=True,
                        help="Task/head ID (0,1,2,3) or 'auto' to auto-detect")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to single image or directory of images")
    parser.add_argument("--output", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--img-size", type=int, nargs=2, default=None,
                        help="Input size [H W], defaults to training config")
    parser.add_argument("--overlay", action="store_true",
                        help="Also save blended overlay image")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay blend alpha (0=image, 1=mask)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for directory inference (fixed-task only)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda / cpu), auto-detected if omitted")
    parser.add_argument("--export-onnx", type=str, default=None,
                        help="Export model to ONNX at this path and exit")

    args = parser.parse_args()

    auto_mode = args.task.lower() == "auto"
    task_id: int | None = None if auto_mode else int(args.task)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_model(args.checkpoint, device)
    num_classes_map = {d["task_id"]: d["num_classes"] for d in cfg["datasets"]}
    task_names = {d["task_id"]: d["name"] for d in cfg["datasets"]}

    if not auto_mode and task_id not in num_classes_map:
        print(f"Error: task {task_id} not found. "
              f"Available: {list(num_classes_map.keys())} or 'auto'")
        return

    img_size = tuple(args.img_size) if args.img_size else tuple(cfg["training"]["img_size"])

    if auto_mode:
        print(f"Mode: AUTO-DETECT | input size {img_size}")
        print(f"Will run all heads and pick the most confident one per image")
    else:
        print(f"Task {task_id}: {num_classes_map[task_id]} classes, input size {img_size}")

    if args.export_onnx:
        if auto_mode:
            parser.error("--export-onnx requires a specific --task (not 'auto')")
        export_onnx(model, task_id, img_size, args.export_onnx, device)
        return

    if args.input is None:
        parser.error("--input is required for inference (use --export-onnx for export only)")

    os.makedirs(args.output, exist_ok=True)
    input_path = Path(args.input)

    if input_path.is_dir():
        run_directory(
            model, str(input_path), task_id, num_classes_map,
            img_size, args.output, device,
            task_names=task_names,
            overlay=args.overlay, alpha=args.alpha,
            batch_size=args.batch_size,
        )
    elif input_path.is_file():
        run_single(
            model, str(input_path), task_id, num_classes_map,
            img_size, args.output, device,
            task_names=task_names,
            overlay=args.overlay, alpha=args.alpha,
        )
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
