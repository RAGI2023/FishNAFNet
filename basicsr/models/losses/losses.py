# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.models as tv_models

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class VGGPerceptualLoss(nn.Module):
    """VGG19 perceptual loss using multi-layer features.

    Extracts features from relu1_2, relu2_2, relu3_4 of pretrained VGG19
    and computes L1 loss between predicted and target feature maps.

    forward() returns (l_percep, l_style) to match image_restoration_model.py
    interface. l_style is None unless style_weight > 0.

    Args:
        loss_weight (float): Weight for perceptual feature loss. Default: 1.0.
        style_weight (float): Weight for Gram-matrix style loss. Default: 0.0.
        use_input_norm (bool): Normalize inputs to ImageNet stats. Default: True.
    """

    # VGG19 layer indices for relu1_2, relu2_2, relu3_4
    _SLICE_ENDS = [4, 9, 18]

    def __init__(self, loss_weight=1.0, style_weight=0.0, use_input_norm=True):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        features = list(vgg.features)
        self.slices = nn.ModuleList([
            nn.Sequential(*features[start:end])
            for start, end in zip([0] + self._SLICE_ENDS[:-1], self._SLICE_ENDS)
        ])
        for p in self.parameters():
            p.requires_grad = False

        self.loss_weight = loss_weight
        self.style_weight = style_weight

        if use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        else:
            self.mean = self.std = None

    @staticmethod
    def _gram(feat):
        b, c, h, w = feat.shape
        f = feat.view(b, c, -1)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, pred, target):
        if self.mean is not None:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std

        l_percep = pred.new_zeros(())
        l_style = pred.new_zeros(()) if self.style_weight > 0 else None

        x, y = pred, target
        for slice_ in self.slices:
            x = slice_(x)
            with torch.no_grad():
                y = slice_(y)
            l_percep = l_percep + F.l1_loss(x, y)
            if l_style is not None:
                l_style = l_style + F.l1_loss(self._gram(x), self._gram(y))

        l_percep = self.loss_weight * l_percep
        if l_style is not None:
            l_style = self.style_weight * l_style

        return l_percep, l_style
