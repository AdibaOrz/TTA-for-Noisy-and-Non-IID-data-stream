import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy

def convert_iabn(module, alpha=4):
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = IABN2d if isinstance(module, nn.BatchNorm2d) else IABN1d
        module_output = IABN(
            num_channels=module.num_features,
            alpha=alpha,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
        )

        module_output.bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(child, alpha=alpha)
        )
    del module
    return module_output

class IABN2d(nn.Module):
    def __init__(self, num_channels, alpha, eps, momentum, affine=True):
        super(IABN2d, self).__init__()

        self.num_channels = num_channels
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.bn = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum, affine=affine)

    def _soft_shrinkage(self, x: torch.Tensor, lbd: torch.Tensor):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        sigma2, mu = torch.var_mean(x, dim=(2, 3), keepdim=True, unbiased=True)

        if self.training:
            _ = self.bn(x)  # Update running stats

            sigma2_b, mu_b = torch.var_mean(x, dim=(0, 2, 3), keepdim=True, unbiased=True)
        else:
            if self.bn.track_running_stats:
                sigma2_b, mu_b = self.bn.running_var.view(1, -1, 1, 1), self.bn.running_mean.view(1, -1, 1, 1)
            else:
                sigma2_b, mu_b = torch.var_mean(x, dim=(0, 2, 3), keepdim=True, unbiased=True)

        # Skip to avoid division by zero
        if L <= 1:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / L)
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (L-1))

            mu_adj = mu_b + self._soft_shrinkage(mu - mu_b, self.alpha * s_mu)
            sigma2_adj = sigma2_b + self._soft_shrinkage(sigma2 - sigma2_b, self.alpha * s_sigma2)
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self.bn.weight.view(C, 1, 1)
            bias = self.bn.bias.view(C, 1, 1)
            x_n = x_n * weight + bias
        return x_n

class IABN1d(nn.Module):
    def __init__(self, num_channels, alpha, eps, momentum, affine=True):
        super(IABN1d, self).__init__()

        self.num_channels = num_channels
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.bn = nn.BatchNorm1d(num_channels, eps=eps, momentum=momentum, affine=affine)

    def _soft_shrinkage(self, x: torch.Tensor, lbd: torch.Tensor):
        x_p = F.relu(x - lbd, inplace=True)
        x_n = F.relu(-(x + lbd), inplace=True)
        y = x_p - x_n
        return y

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape

        mu = torch.mean(x, dim=(2), keepdim=True)  # (B, C, 1)
        sigma2 = torch.var(x, dim=(2), keepdim=True)  # (B, C, 1)

        if self.training:
            self.bn(x)  # Update running stats

            sigma2_b, mu_b = torch.var_mean(x, dim=(0, 2), keepdim=True, unbiased=True)
        else:
            if self.bn.track_running_stats:
                sigma2_b, mu_b = self.bn.running_var.view(1, -1, 1), self.bn.running_mean.view(1, -1, 1)
            else:
                sigma2_b, mu_b = torch.var_mean(x, dim=(0, 2), keepdim=True, unbiased=True)

        # Originally there's a threshold conditional, skipped here
        s_mu = torch.sqrt((sigma2_b + self.eps) / L)
        s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (L - 1))

        mu_adj = mu_b + self._soft_shrinkage(mu - mu_b, self.alpha * s_mu)
        sigma2_adj = sigma2_b + self._soft_shrinkage(sigma2 - sigma2_b, self.alpha * s_sigma2)
        sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)
        if self.affine:
            weight = self.bn.weight.view(C, 1)
            bias = self.bn.bias.view(C, 1)
            x_n = x_n * weight + bias
        return x_n