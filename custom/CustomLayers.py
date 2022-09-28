import math
import torch
from torch import nn
from torch.nn import functional as F

class ShuffleUnit(nn.Module):
    def __init__(self,
                 groups: int):
        super(ShuffleUnit, self).__init__()
        self.groups = groups

    def forward(self, feats):
        
        batch_size, _, height, width = feats.size()
        
        feats = feats.reshape(batch_size, self.groups, -1, height, width)
        feats = feats.transpose(1, 2)
        feats = feats.reshape(batch_size, -1, height, width)
        
        return feats

class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple=(3, 3),
                 stride: tuple=(1, 1),
                 padding: tuple=(1, 1),
                 groups: int = 1,
                 padding_mode: str = 'reflect'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups, bias=False, padding_mode=padding_mode)

        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):

        x = self.conv(x)

        x = self.batch_norm(x)

        x = F.leaky_relu(x, negative_slope=.1)

        return x

class Upsample(nn.Module):
    def __init__(self,
                 in_channels: int=128,
                 out_channels: int=256):
        super(Upsample, self).__init__()

        self.block = nn.Sequential(ConvBlock(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=(1, 1),
                                             padding=(0, 0)),
                                   nn.Upsample(scale_factor=2,
                                               mode='bicubic',
                                               align_corners=True))
        self.init_params()

    def forward(self, x):

        x = self.block(x)

        return x

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

class YoloHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels_list: list):
        super(YoloHead, self).__init__()
        self.block = nn.Sequential(ConvBlock(in_channels=in_channels,
                                             out_channels=channels_list[0]),
                                   nn.Conv2d(in_channels=channels_list[0],
                                             out_channels=channels_list[-1],
                                             kernel_size=(1, 1),
                                             padding=(0, 0)))
        self.init_params()

    def forward(self, x):

        x = self.block(x)

        return x

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
    
class ResBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shuffle: bool=True):
        super(ResBlock, self).__init__()
        assert out_channels == 2 * in_channels
        assert not (out_channels % 2 or in_channels % 2)

        self.init_conv = ConvBlock(in_channels=in_channels,
                                   out_channels=out_channels//2)

        self.former_middle_conv = ConvBlock(in_channels=out_channels//2,
                                            out_channels=out_channels//4,
                                            groups=2)

        self.latter_middle_conv = ConvBlock(in_channels=out_channels//4,
                                            out_channels=out_channels//4)

        self.final_conv = ConvBlock(in_channels=out_channels//2,
                                    out_channels=out_channels//2)

        self.shuffle = shuffle
        if shuffle:
            self.shuffle_unit = ShuffleUnit(groups=2)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2),
                                     stride=(2, 2))

    def forward(self, x):

        x = self.init_conv(x)

        main_residue = x

        x = self.former_middle_conv(x)

        if self.shuffle:

            x = self.shuffle_unit(x)

        branch_residue = x

        x = self.latter_middle_conv(x)

        x = torch.cat([x, branch_residue], dim=1)

        x = self.final_conv(x)

        feat = x

        x = torch.cat([main_residue, x], dim=1)

        x = self.max_pool(x)

        return x, feat

class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()

        self.init_conv = ConvBlock(in_channels=3, out_channels=32, stride=(2, 2))
        self.middle_conv = ConvBlock(in_channels=32, out_channels=64, stride=(2, 2))
        self.final_conv = ConvBlock(in_channels=512, out_channels=512)

        self.init_resblock = ResBlock(in_channels=64, out_channels=128)
        self.middle_resblock = ResBlock(in_channels=128, out_channels=256)
        self.final_resblock = ResBlock(in_channels=256, out_channels=512)

        self.init_params()

    def forward(self, x):

        x = self.init_conv(x)
        x = self.middle_conv(x)

        x, _ = self.init_resblock(x)
        x, _ = self.middle_resblock(x)
        x, feat2 = self.final_resblock(x)

        feat1 = self.final_conv(x)

        return feat1, feat2

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