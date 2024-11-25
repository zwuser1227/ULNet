import torch
from torch import nn
import torch.nn.functional as F
import utils
from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
import numpy as np
import torch.fft as fft
class sgma(nn.Module):
    def __init__(self, in_channels):
        super(sgma, self).__init__()
        self.attention_out = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0)
    def forward(self, x):
        attention = self.attention_out(x)
        attention_weight = attention.sigmoid()
        sg = torch.max(attention_weight)
        return sg
class CVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
        # self.sg = sgma(input_dim//4)
        self.gao_fil = GaoFilter2s()
     
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        # print(x_norm.shape)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + x1 
        x_mamba2 = self.mamba(x2+x_mamba1)  + self.gao_fil(x1) 
        x_mamba3 = self.mamba(x3+x_mamba2)  + self.gao_fil(x2) 
        x_mamba4 = self.mamba(x4+x_mamba3)  + self.gao_fil(x3)

        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out
class conv_d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_d, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        self.ChannelAttention = ChannelAttention(out_channels)
        self.SpatialAttention = SpatialAttention()
        self.ch = nn.Conv2d(3*out_channels, out_channels, kernel_size=1, padding=0)
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x)
        x_o = x
        x_channelAttention = self.ChannelAttention(x)
        x= x*x_channelAttention
        x_os = x
        x_spatialAttention = self.SpatialAttention(x_o+x)
        x_out= x_o*x_spatialAttention
        # x_cat = torch.cat([x_out, x_os,x_o], dim=1)
        sum= x_out+x_os+x_o
        # return self.ch(x_cat)
        return sum
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            # nn.Conv2d(in_planes, in_planes, 1, bias=False),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, kernel_size=1, padding=0, dilation=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class wave_attention(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(wave_attention, self).__init__()
        self.pre_conv1 = conv_d(in_channel[0], out_channel)
        self.pre_conv2 = conv_d(in_channel[1], out_channel)


        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
   
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        # x_o = x1
        if x1.size(2)==135:
            pad_h = (2 - x1.size(2) % 2) % 2
            pad_w = (2 - x1.size(3) % 2) % 2
            x1 = F.pad(x1, (0, pad_w, 0, pad_h), mode='reflect')
            x2 = self.pre_conv2(input[1])
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                    output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
       )
            fusion = x1 + x2
            return fusion
        else:
            x2 = self.pre_conv2(input[1])
            x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                    output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
           
            fusion = x1 + x2
           
            return fusion
class ULNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,8,8,16,32,64],
                split_att='fc', bridge=False):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, dilation=1, padding=1),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, dilation=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            CVMLayer(input_dim=c_list[2], output_dim=c_list[3]),
           
        )
        self.encoder5 = nn.Sequential(
            CVMLayer(input_dim=c_list[3], output_dim=c_list[4]),
            # nn.Conv2d(c_list[4], c_list[4], 3, stride=1, dilation=1, padding=1),
        )
        # self.encoder5 = nn.Sequential(
        #     nn.Conv2d(c_list[3], c_list[4], 3, stride=1, dilation=1, padding=1)
        # )
        self.encoder6 = nn.Sequential(
            CVMLayer(input_dim=c_list[4], output_dim=c_list[5]),
            # nn.Conv2d(c_list[4], c_list[4], 3, stride=1, dilation=1, padding=1),
        )

        self.channl_change = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.Gao0 = GaoFilter2(c_list[0], c_list[1])#input_channel, output_channel, kernel_size=30, sigma=5
        self.Gao1 = GaoFilter2(c_list[1], c_list[2])
        self.Gao2 = GaoFilter2(c_list[2], c_list[3])
        self.Gao3 = GaoFilter2(c_list[3], c_list[4])
        self.Gao4 = GaoFilter2(c_list[4], c_list[5])
        
        self.sg1 = sgma(c_list[4])
        # self.skip_scale= nn.Parameter(torch.ones(1))
        self.decoder1 = nn.Sequential(
            CVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            CVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            CVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )   
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1, dilation=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1, dilation=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.level1_1 = wave_attention((c_list[3], c_list[4]), c_list[3], 2)
        self.level1_2 = wave_attention((c_list[2], c_list[3]), c_list[2], 2)
        self.level1_3 = wave_attention((c_list[1], c_list[2]), c_list[1], 2)
        self.level1_4 = wave_attention((c_list[0], c_list[1]), c_list[0], 2)

        self.level2_1 = wave_attention((c_list[2], c_list[3]), c_list[2], 2)
        self.level2_2 = wave_attention((c_list[1], c_list[2]), c_list[1], 2)
        self.level2_3 = wave_attention((c_list[0], c_list[1]), c_list[0], 2)

        self.level3_1 = wave_attention((c_list[1], c_list[2]), c_list[1], 2)
        self.level3_2 = wave_attention((c_list[0], c_list[1]), c_list[0], 2)

        self.level4_1 = wave_attention((c_list[0], c_list[1]), c_list[0], 2)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.final1 = nn.Conv2d(c_list[1], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(c_list[2], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(c_list[3], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(c_list[4], num_classes, kernel_size=1)
        # self.final5 = nn.Conv2d(c_list[1], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x):
        out = F.max_pool2d(F.gelu(self.ebn1(self.encoder1(x))),2,2)#S1-1
        t1 = out # b, c0, H/2, W/2-512
        channl_change = F.max_pool2d(F.gelu(self.channl_change(x)),2,2)

        out = F.max_pool2d(F.gelu(self.ebn2(self.encoder2(out))),2,2)#S1-2
        t2 = out# b, c1, H/4, W/4-256 
        Gao0 = F.max_pool2d(F.gelu(self.Gao0(channl_change)),2,2)
        
        out = F.max_pool2d(F.gelu(self.ebn3(self.encoder3(out+Gao0))),2,2)#S1-3
        # 对输入进行对称填充，使其大小可以被池化层整除
        out_o = out
        pad_h = (2 - out.size(2) % 2) % 2
        pad_w = (2 - out.size(3) % 2) % 2
        out = F.pad(out, (0, pad_w, 0, pad_h), mode='reflect')
        t3 = out# b, c2, H/8, W/8-138
        Gao1 = F.max_pool2d(F.gelu(self.Gao1(Gao0)),2,2)
  
        out = F.max_pool2d(F.gelu(self.ebn4(self.encoder4(out+Gao1))),2,2)#S2
        t4 = out# b, c3, H/16, W/16
        Gao2 = F.max_pool2d(F.gelu(self.Gao2(Gao1)),2,2)

        out = F.max_pool2d(F.gelu(self.ebn5(self.encoder5(out+Gao2))),2,2)#S3
        t5 = out # b, c4, H/32, W/32

        Gao3 = F.max_pool2d(F.gelu(self.Gao3(Gao2)),2,2)

        out = F.gelu(self.encoder6(out+Gao3)) # b, c5, H/32, W/32   S4

        Gao4 = F.gelu(self.Gao4(Gao3))


        out5_act = self.dbn1(self.decoder1(out+Gao4))
        out5_w = torch.sigmoid(out5_act)
        out5a = F.gelu(out5_act) # b, c4, H/32, W/32
        out5 = torch.add(out5a, self.sg1(out5a)*t5) # b, c4, H/32, W/32
        # out5 = torch.add(out5, self.sg1(out5a) * self.Gao1(t5))

        out4_act = F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)
        out4_w = torch.sigmoid(out4_act)
        out4a = F.gelu(out4_act) # b, c3, H/16, W/16
        out4 = torch.add(out4a, self.sg1(out5a)*t4) # b, c3, H/16, W/16
        # out4 = torch.add(out4, self.sg1(out5a) *self.Gao2(t4))
#
        out3_act = F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)
        out3_w = torch.sigmoid(out3_act)
        out3a = F.gelu(out3_act) # b, c2, H/8, W/8
        out3 = torch.add(out3a, self.sg1(out5a)*t3) # b, c2, H/8, W/8
        # out3 = torch.add(out3, self.sg1(out5a) *self.Gao3(t3))
        #裁剪恢复
        out3 = out3[:, :, :out_o.size(2), :out_o.size(3)]
        
        out2_act = F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)
        out2_w = torch.sigmoid(out2_act)
        out2a = F.gelu(out2_act) # b, c1, H/4, W/4
        out2 = torch.add(out2a, self.sg1(out5a)*t2) # b, c1, H/4, W/4 
        # out2 = torch.add(out2, self.sg1(out5a) *self.Gao4(t2))
        
        out1_act = F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)
        out1_w = torch.sigmoid(out1_act)
        out1a = F.gelu(out1_act) # b, c0, H/2, W/2
        out1 = torch.add(out1a, self.sg1(out5a)*t1) # b, c0, H/2, W/2
        
        #out1s = self.final4(out5)
        #out2s = self.final3(out4)
        #out3s = self.final2(out3)
        #out4s = self.final1(out2)
        #out5s = self.final(out1)

        #Gao4s = F.interpolate(self.final4(Gao4),scale_factor=(32,32),mode ='bilinear',align_corners=True) # b, num_class, H, W
        #Gao3s = F.interpolate(self.final4(Gao3),scale_factor=(32,32),mode ='bilinear',align_corners=True) # b, num_class, H, W
        #Gao2s = F.interpolate(self.final3(Gao2),scale_factor=(16,16),mode ='bilinear',align_corners=True) # b, num_class, H, W
        #Gao1s = F.interpolate(self.final2(Gao1),scale_factor=(8,8),mode ='bilinear',align_corners=True) # b, num_class, H, W
        #Gao0s = F.interpolate(self.final1(Gao0),scale_factor=(4,4),mode ='bilinear',align_corners=True) # b, num_class, H, W


        level1_1 = self.level1_1(out4, out5)
        level1_2 = self.level1_2(out3, out4)
        level1_3 = self.level1_3(out2, out3)
        level1_4 = self.level1_4(out1, out2)
    
        level2_1 = self.level2_1(level1_2, level1_1)
        level2_2 = self.level2_2(level1_3, level1_2)
        level2_3 = self.level2_3(level1_4, level1_3)

        level3_1 = self.level3_1(level2_2, level2_1)
        level3_2 = self.level3_2(level2_3, level2_2)

        level4_1 = self.level4_1(level3_2, level3_1)
        out0 = F.interpolate(self.final(level4_1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        # print(out0.shape)
        # print(Gao0s.shape)
        return torch.sigmoid(out0)
        #return torch.sigmoid(out0),torch.sigmoid(out1s),torch.sigmoid(out2s),torch.sigmoid(out3s),torch.sigmoid(out4s),torch.sigmoid(out5s),Gao0s,Gao1s,torch.sigmoid(Gao2s),torch.sigmoid(Gao3s),torch.sigmoid(Gao4s)

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
