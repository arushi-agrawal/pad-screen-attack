"""The network and the tensor conversion, shared by training, analysis, and inference.

MobileNetV3-small with ImageNet weights, first seven feature blocks frozen, single-output head. Patches are 128 px,
normalised with the ImageNet mean and std.
"""
import numpy as np
import torch, torch.nn as nn
import torchvision

PATCH = 128
FACENORM_PX = 865          # bona fide median face-region size; used by the shrink test and the face-normalised experiment
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(p):
    """PIL RGB patch -> float tensor (3, H, W), ImageNet-normalised."""
    a = (np.asarray(p, np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def make_model(pretrained=True):
    """MobileNetV3-small, single logit, first 7 feature blocks frozen. pretrained=False when loading saved weights."""
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    for p in m.features[:7].parameters():
        p.requires_grad = False
    return m


def load_model(weights_path):
    m = make_model(pretrained=False)
    m.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return m.eval()
