import torch
from torch import nn
import torch.nn.functional as F

import math

# https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
# but there is a bug in the original code: it sums up the entropy over a batch. so I take mean instead of sum
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0, weighted=False, e_margin=0.4*math.log(10)):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor
        self.weighted = weighted
        self.e_margin = e_margin

    def forward(self, x):

        softmax = F.softmax(x/self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax+1e-6)

        if self.weighted:
            coeff = 1 / (torch.exp(entropy.clone().detach() - self.e_margin))
            entropy = entropy.mul(coeff)

        return entropy.mean()