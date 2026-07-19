"""Frozen CNN encoder (ResNet50) used as a feature extractor.

Two outputs are possible:
  - global vector (2048,)  -> phase 1 (plain LSTM decoder)
  - grid  (49, 2048)       -> phase 2 (attention decoder)
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# transform every image the way the pretrained CNN expects
image_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class EncoderCNN(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            # testing-only path: seed so every script builds the SAME random
            # encoder (otherwise train/inference would see different features)
            weights = None
            torch.manual_seed(1234)
        resnet = models.resnet50(weights=weights)
        # everything except avgpool+fc -> output (B, 2048, 7, 7)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():          # frozen: pure feature extractor
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images, return_grid: bool = False):
        """images: (B, 3, 224, 224) normalized.
        Returns (B, 2048) or, if return_grid, (B, 49, 2048)."""
        fmap = self.backbone(images)                       # (B, 2048, 7, 7)
        if return_grid:
            return fmap.flatten(2).transpose(1, 2)         # (B, 49, 2048)
        return self.pool(fmap).flatten(1)                  # (B, 2048)
