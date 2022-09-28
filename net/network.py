import torch
from torch import nn
from custom.CustomLayers import BackBone, Upsample, YoloHead, ConvBlock
from custom.attention import Attention

class YoloBody(nn.Module):
    def __init__(self,
                 class_num: int,
                 anchor_num: int):
        super(YoloBody, self).__init__()

        self.class_num = class_num
        self.anchor_num = anchor_num

        self.backbone = BackBone()

        self.conv_block_P5 = ConvBlock(in_channels=512,
                                       out_channels=256,
                                       kernel_size=(1, 1),
                                       padding=(0, 0))

        self.up_sample = Upsample(in_channels=256,
                                  out_channels=128)

        self.yolo_head_P5 = YoloHead(in_channels=256,
                                     channels_list=[512, anchor_num * (class_num + 5)])

        self.yolo_head_P4 = YoloHead(in_channels=384,
                                     channels_list=[256, anchor_num * (class_num + 5)])

        self.attention_P5 = Attention(attn_size=512)
        self.attention_P4 = Attention(attn_size=256)
        self.attention = Attention(attn_size=128)

    def forward(self, x):

        feat1, feat2 = self.backbone(x)

        feat1 = self.attention_P5(feat1)
        feat2 = self.attention_P4(feat2)

        P5 = self.conv_block_P5(feat1)
        P5_output = self.yolo_head_P5(P5)

        P5_upsamp = self.up_sample(P5)
        P5_atten = self.attention(P5_upsamp)

        P4 = torch.cat([P5_atten, feat2], dim=1)
        P4_output = self.yolo_head_P4(P4)

        P5_output = P5_output.permute(0, 2, 3, 1)
        P4_output = P4_output.permute(0, 2, 3, 1)

        h_p5, w_p5 = P5_output.size()[1:3]
        h_p4, w_p4 = P4_output.size()[1:3]

        P5_output = P5_output.reshape(-1, h_p5, w_p5, self.anchor_num, self.class_num+5)
        P4_output = P4_output.reshape(-1, h_p4, w_p4, self.anchor_num, self.class_num+5)

        return P5_output, P4_output

    def get_weights(self):

        weights = []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)

        return weights
