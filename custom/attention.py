import math
import torch
from torch import nn

class Attention(nn.Module):
    def __init__(self,
                 attn_size: int,
                 reduction: int=16):
        super(Attention, self).__init__()
        assert not attn_size % reduction

        self.conv_hw = nn.Conv2d(in_channels=attn_size, out_channels=attn_size//reduction,
                                 kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.conv_h = nn.Conv2d(in_channels=attn_size//reduction, out_channels=attn_size,
                                kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_w = nn.Conv2d(in_channels=attn_size//reduction, out_channels=attn_size,
                                kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.batch_norm = nn.BatchNorm2d(num_features=attn_size//reduction)

    def forward(self, x):

        *_, height, width = x.size()

        feats_h = torch.mean(x, dim=-1, keepdim=True)
        feats_w = torch.mean(x, dim=2, keepdim=True).transpose(2, 3)

        feats = torch.cat([feats_h, feats_w], dim=2)
        feats = self.conv_hw(feats)
        feats = self.batch_norm(feats)
        feats = torch.relu(feats)

        feats_h, feats_w = feats.split(split_size=[height, width], dim=2)

        feats_h = self.conv_h(feats_h)
        feats_w = self.conv_w(feats_w).transpose(2, 3)

        feats = x * feats_w * feats_h

        return feats

    def init_params(self):

        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if 'batch_norm' in name.split('.'):
                    continue
                elif name.split('.')[-1] == 'weight':
                    stddev = math.sqrt(2 / sum(param.size()[:2]))
                    torch.nn.init.normal_(param, std=stddev)
                else:
                    torch.nn.init.zeros_(param)