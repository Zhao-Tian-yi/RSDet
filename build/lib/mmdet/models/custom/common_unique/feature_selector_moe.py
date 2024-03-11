import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import ResLayer


@MODELS.register_module()
class FeatureSelectorMOE(BaseModule):
    """


    """

    def __init__(self, ):
        pass

    def forward(self, x):

        return tuple(outs)
