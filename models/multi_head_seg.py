import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import SharedBackbone
from .heads import FPNDecoder, SegmentationHead


class MultiHeadSegModel(nn.Module):
    """Multi-head segmentation model with shared backbone and decoder.

    Architecture:
        Shared Backbone (timm) → Shared FPN Decoder → Per-task Seg Head

    Each task gets its own lightweight classification head while sharing
    the heavy feature extraction and decoding layers.

    Args:
        backbone_name:  timm model name (e.g. "resnet50", "efficientnet_b3").
        num_classes:    dict mapping task_id (int) → number of classes.
        decoder_channels: width of the FPN decoder.
        head_hidden:    hidden channels inside each classification head.
        pretrained:     load ImageNet pretrained backbone weights.
        shared_decoder: if True, all tasks share one FPN decoder;
                        if False, each task gets its own decoder (more params).
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: dict[int, int] | None = None,
        decoder_channels: int = 256,
        head_hidden: int = 128,
        pretrained: bool = True,
        shared_decoder: bool = True,
    ):
        super().__init__()
        if num_classes is None:
            num_classes = {0: 32, 1: 2, 2: 8, 3: 3}

        self.task_ids = sorted(num_classes.keys())
        self.shared_decoder_mode = shared_decoder

        self.backbone = SharedBackbone(backbone_name, pretrained=pretrained)
        enc_ch = self.backbone.channels

        if shared_decoder:
            self.decoder = FPNDecoder(enc_ch, decoder_channels)
            dec_out = self.decoder.out_channels
        else:
            self.decoders = nn.ModuleDict({
                str(tid): FPNDecoder(enc_ch, decoder_channels)
                for tid in self.task_ids
            })
            dec_out = decoder_channels

        self.heads = nn.ModuleDict({
            str(tid): SegmentationHead(dec_out, nc, head_hidden)
            for tid, nc in num_classes.items()
        })

    def forward(
        self,
        images: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Forward pass routing each sample to its task head.

        Args:
            images:      (B, 3, H, W) input batch.
            dataset_ids: (B,) integer tensor, one task id per sample.

        Returns:
            dict mapping task_id → (N_task, C_task, H, W) logits
            where N_task is the count of samples belonging to that task.
        """
        features = self.backbone(images)
        input_size = images.shape[2:]
        outputs: dict[int, torch.Tensor] = {}

        for tid in dataset_ids.unique().tolist():
            mask = dataset_ids == tid
            task_feats = [f[mask] for f in features]

            if self.shared_decoder_mode:
                decoded = self.decoder(task_feats)
            else:
                decoded = self.decoders[str(tid)](task_feats)

            outputs[tid] = self.heads[str(tid)](decoded, input_size)

        return outputs

    def forward_single_task(
        self,
        images: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        """Convenience method for inference on a single task."""
        features = self.backbone(images)
        if self.shared_decoder_mode:
            decoded = self.decoder(features)
        else:
            decoded = self.decoders[str(task_id)](features)
        return self.heads[str(task_id)](decoded, images.shape[2:])

    def forward_auto_detect(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, int, dict[int, float]]:
        """Run all heads and pick the most confident one.

        Uses **normalized entropy** so heads with different class counts
        are compared fairly (raw softmax confidence is biased toward
        heads with fewer classes).

        Normalized confidence = 1 - H(p) / log(num_classes)
        where H(p) is the per-pixel entropy of the softmax output.
        This gives a 0-1 score regardless of class count.

        Args:
            images: (B, 3, H, W) input batch.

        Returns:
            logits:       (B, C, H, W) from the winning head.
            best_task_id: int, which head was selected.
            confidences:  dict task_id → normalized confidence score.
        """
        features = self.backbone(images)
        input_size = images.shape[2:]

        if self.shared_decoder_mode:
            decoded = self.decoder(features)

        all_logits: dict[int, torch.Tensor] = {}
        confidences: dict[int, float] = {}

        for tid in self.task_ids:
            if not self.shared_decoder_mode:
                decoded = self.decoders[str(tid)](features)

            logits = self.heads[str(tid)](decoded, input_size)
            all_logits[tid] = logits

            num_classes = logits.shape[1]
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1)
            max_entropy = torch.log(torch.tensor(float(num_classes)))
            normalized_conf = 1.0 - (entropy.mean() / max_entropy).item()
            confidences[tid] = normalized_conf

        best_task_id = max(confidences, key=confidences.get)
        return all_logits[best_task_id], best_task_id, confidences
