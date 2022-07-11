
import torch
import logging
import torchvision
from torch import nn
import torch.nn.functional as F

from model.layers import Flatten, L2Norm, GeM


EPS = 1e-06
STRIDE = 1

CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "vgg16": 512,
    }


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim, init_p=3):
        super().__init__()

        self.backbone, features_dim = get_backbone(backbone)
        self.projection = nn.Linear(features_dim, fc_output_dim)
        self.p = nn.Parameter(torch.ones(1)*init_p)

    def embedd(self, x):
        return self.backbone(x)

    def global_pool(self, x):
        x = F.normalize(x, p=2, dim=1)
        x = x.clamp(min=EPS).pow(self.p)
        x = x.mean([-2, -1])
        x = x.pow(1/self.p)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    def local_pool(self, x, patch_sizes_list=(3,)):
        x = F.normalize(x, p=2, dim=1)

        global_descriptors = x.clamp(min=EPS).pow(self.p)
        global_descriptors = global_descriptors.mean([-2, -1])
        global_descriptors = global_descriptors.pow(1/self.p)
        global_descriptors = self.projection(global_descriptors)
        global_descriptors = F.normalize(global_descriptors, p=2, dim=1)

        integral_feat = get_integral_feature(x)
        local_descriptors = []
        for patch_size in patch_sizes_list:
            patch_descriptors = get_square_regions_from_integral(integral_feat, patch_size, STRIDE)
            patch_descriptors = patch_descriptors.view(x.shape[0], x.shape[1], -1)
            patch_descriptors = patch_descriptors.permute([0, 2, 1])
            patch_descriptors = self.projection(patch_descriptors)
            patch_descriptors = F.normalize(patch_descriptors, p=2, dim=2)
            patch_descriptors = patch_descriptors.permute([0, 2, 1])

            local_descriptors.append(patch_descriptors)

        return global_descriptors, local_descriptors


def get_integral_feature(feat_in):
    """
    Input/Output as [N,D,H,W] where N is batch size and D is descriptor dimensions
    """

    feat_out = torch.cumsum(feat_in, dim=-1)
    feat_out = torch.cumsum(feat_out, dim=-2)

    return feat_out

def get_square_regions_from_integral(feat_integral, patch_size, patch_stride):
    """
    Input as [N,D,H,W] integral fetures.
    Output as [N, D, H, W] unstandardized patch descriptors. Note that standardization
    isn't needde if the output is intended to be normalized on the D dimension.
    """

    offset = patch_size // 2

    feat_integral = torch.nn.functional.pad(feat_integral, (1 + offset, 0, 1 + offset, 0), "constant", 0)
    feat_integral = torch.nn.functional.pad(feat_integral, (0, offset, 0, offset), "replicate", 0)

    N, D, H, W = feat_integral.shape

    if feat_integral.get_device() == -1:
        conv_weight = torch.ones(D, 1, 2, 2)
    else:
        conv_weight = torch.ones(D, 1, 2, 2, device=feat_integral.get_device())
    conv_weight[:, :, 0, -1] = -1
    conv_weight[:, :, -1, 0] = -1
    feat_regions = torch.nn.functional.conv2d(feat_integral, conv_weight, stride=patch_stride, groups=D, dilation=patch_size)

    return feat_regions


def get_backbone(backbone_name):
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)

        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]

    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")

    backbone = torch.nn.Sequential(*layers)
    print("backbone test")
    print(backbone(torch.zeros([2, 3, 224, 224])).shape)

    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]

    return backbone, features_dim
