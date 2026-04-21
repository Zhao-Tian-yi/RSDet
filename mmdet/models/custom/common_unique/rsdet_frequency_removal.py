import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS


class TinyFrequencyMaskPredictor(nn.Module):

    def __init__(self, imgshape, patch_num=20):
        super().__init__()
        width, height = imgshape
        self.imgshape = imgshape
        self.patch_num = patch_num
        self.conv1 = nn.Conv2d(3, 16, 5, 2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.flatten = nn.Flatten()
        # conv1(stride=2)+pool, conv2+pool, conv3+pool => spatial size / 16
        self.trans2list = nn.Sequential(
            nn.Linear(
                in_features=int(64 * height / 16 * width / 16),
                out_features=np.power(self.patch_num, 2),
                bias=False),
            nn.Sigmoid())
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        return self.trans2list(self.flatten(x))


class FrequencyMaskBinarizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mask_logits):
        mask_topk, _ = mask_logits.topk(320)
        mask_threshold = torch.min(mask_topk, dim=1).values.unsqueeze(-1)
        binary_mask = torch.where(
            torch.ge(mask_logits, mask_threshold),
            torch.ones_like(mask_logits),
            torch.zeros_like(mask_logits))
        ctx.save_for_backward(mask_logits, mask_threshold)
        return binary_mask

    @staticmethod
    def backward(ctx, grad_output):
        mask_logits, mask_threshold = ctx.saved_tensors
        grad_mask_logits = grad_output.clone()
        grad_mask_logits[mask_logits < mask_threshold] = 0
        return grad_mask_logits


@MODELS.register_module()
class FrequencyRemovalModule(BaseModule):
    """Paper-aligned frequency-domain removal module used by RSDet."""

    def __init__(self, imgshape, keep_low, patch_num=20):
        super().__init__()
        self.patch_num = patch_num
        self.keep_low = keep_low
        self.imgshape = imgshape
        self.rgb_mask_predictor = TinyFrequencyMaskPredictor(imgshape, patch_num)
        self.ir_mask_predictor = TinyFrequencyMaskPredictor(imgshape, patch_num)

    @staticmethod
    def _complex_magnitude(freq_tensor, eps=1e-12):
        return torch.sqrt(
            freq_tensor.real.square() + freq_tensor.imag.square() + eps)

    def _keep_low_frequency(self, mask):
        center = int(self.patch_num / 2)
        low_slice = slice(center - 1, center + 1)
        mask[:, :, low_slice, low_slice] = 1 if self.keep_low else 0
        return mask

    def forward(self, img_vis, img_lwir):
        vis_fft = torch.fft.fft2(img_vis)
        lwir_fft = torch.fft.fft2(img_lwir)
        vis_fft = torch.fft.fftshift(self._complex_magnitude(vis_fft))
        lwir_fft = torch.fft.fftshift(self._complex_magnitude(lwir_fft))

        mask_vis_logits = self.rgb_mask_predictor(vis_fft)
        mask_lwir_logits = self.ir_mask_predictor(lwir_fft)

        mask_vis = FrequencyMaskBinarizer.apply(mask_vis_logits).reshape(
            (-1, 1, self.patch_num, self.patch_num))
        mask_lwir = FrequencyMaskBinarizer.apply(mask_lwir_logits).reshape(
            (-1, 1, self.patch_num, self.patch_num))

        mask_vis = self._keep_low_frequency(mask_vis)
        mask_lwir = self._keep_low_frequency(mask_lwir)

        scale_factor = [
            self.imgshape[0] / self.patch_num,
            self.imgshape[1] / self.patch_num
        ]
        mask_vis = F.interpolate(mask_vis, scale_factor=scale_factor, mode='nearest')
        mask_lwir = F.interpolate(mask_lwir, scale_factor=scale_factor, mode='nearest')
        return mask_vis, mask_lwir
