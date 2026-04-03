import torch
import torch.nn as nn
import timm


class SharedBackbone(nn.Module):
    """Shared feature extractor using timm models.
    
    Returns multi-scale features (C2, C3, C4, C5) for FPN-style decoding.
    Supports any timm backbone that has `features_only` mode.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True,
                 out_indices: tuple = (1, 2, 3, 4)):
        super().__init__()
        self.encoder = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.channels = self.encoder.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(x)
