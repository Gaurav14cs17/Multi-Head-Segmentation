import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder.
    
    Takes multi-scale encoder features and produces a fused feature map
    at the finest resolution via top-down lateral connections.
    """

    def __init__(self, encoder_channels: list[int], decoder_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for ch in encoder_channels:
            self.lateral_convs.append(nn.Conv2d(ch, decoder_channels, 1))
            self.output_convs.append(nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_channels),
                nn.ReLU(inplace=True),
            ))

        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(encoder_channels), decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = decoder_channels

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:],
                mode="bilinear", align_corners=False,
            )

        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]

        target_size = outputs[0].shape[2:]
        for i in range(1, len(outputs)):
            outputs[i] = F.interpolate(
                outputs[i], size=target_size, mode="bilinear", align_corners=False,
            )

        return self.fuse(torch.cat(outputs, dim=1))


class SegmentationHead(nn.Module):
    """Lightweight classification head that maps decoder features to class logits."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
