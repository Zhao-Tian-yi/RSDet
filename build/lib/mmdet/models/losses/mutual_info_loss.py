# source: https://github.com/manman1995/Mutual-Information-driven-Pan-sharpening/blob/main/models/GPPNN.py
from torch.distributions import Normal, Independent, kl
from torch.autograd import Variable
import torch
import torch.nn as nn
CE = torch.nn.BCELoss(reduction='sum')

from typing import Optional, Tuple, Union

import numpy as np

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss

@weighted_loss
def kl_divergence_loss( posterior_latent_space, prior_latent_space):
    kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
    return kl_div

@MODELS.register_module()
class MutualInfoLoss(nn.Module):
    def __init__(self, input_channels=1024, channels=64, latent_size = 64):
        super(MutualInfoLoss, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)
        self.layer3 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        self.layer4 = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

        self.channel = channels

        # self.fc1_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_rgb1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc1_lwir1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        # self.fc2_lwir1 = nn.Linear(channels * 1 * 16 * 16, latent_size)
        #
        # self.fc1_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_rgb2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc1_lwir2 = nn.Linear(channels * 1 * 22 * 22, latent_size)
        # self.fc2_lwir2 = nn.Linear(channels * 1 * 22 * 22, latent_size)

        self.fc1_rgb3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_rgb3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc1_lwir3 = nn.Linear(channels * 1 * 4 * 4, latent_size)
        self.fc2_lwir3 = nn.Linear(channels * 1 * 4 * 4, latent_size)

        self.leakyrelu = nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, rgb_feat, lwir_feat):
        rgb_feat = self.layer3(self.leakyrelu(self.layer1(rgb_feat)))
        lwir_feat = self.layer4(self.leakyrelu(self.layer2(lwir_feat)))
        rgb_feat = rgb_feat.view(-1, self.channel  * 4 * 4)
        lwir_feat = lwir_feat.view(-1, self.channel  * 4 * 4)
        mu_rgb = self.fc1_rgb3(rgb_feat)
        logvar_rgb = self.fc2_rgb3(rgb_feat)
        mu_lwir = self.fc1_lwir3(lwir_feat)
        logvar_lwir = self.fc2_lwir3(lwir_feat)

        mu_lwir = self.tanh(mu_lwir)
        mu_rgb = self.tanh(mu_rgb)
        logvar_lwir = self.tanh(logvar_lwir)
        logvar_rgb = self.tanh(logvar_rgb)
        z_rgb = self.reparametrize(mu_rgb, logvar_rgb)
        dist_rgb = Independent(Normal(loc=mu_rgb+10-8, scale=torch.exp(logvar_rgb+10-8)), 1)
        z_lwir = self.reparametrize(mu_lwir, logvar_lwir)
        dist_lwir = Independent(Normal(loc=mu_lwir+10-8, scale=torch.exp(logvar_lwir+10-8)), 1)
        bi_di_kld = torch.mean(kl_divergence_loss(dist_rgb, dist_lwir)) + torch.mean(kl_divergence_loss(dist_lwir, dist_rgb))
        z_rgb_norm = torch.sigmoid(z_rgb)
        z_lwir_norm = torch.sigmoid(z_lwir)
        ce_rgb_lwir = CE(z_rgb_norm,z_lwir_norm.detach())
        ce_lwir_rgb = CE(z_lwir_norm, z_rgb_norm.detach())
        latent_loss = ce_rgb_lwir+ce_lwir_rgb-bi_di_kld
        # latent_loss = torch.abs(cos_sim(z_rgb,z_lwir)).sum()

        return latent_loss