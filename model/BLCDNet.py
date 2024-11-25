import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils
# import transformer
import torch.nn.functional as F
# import pvt1
# import pvt2
from functools import partial
class VGG_together(nn.Module):
    def __init__(self):
        super(VGG_together, self).__init__()
        # 感受器细胞
        self.conv_1 = self.make_layers(3, [16])
        self.conv_2 = self.make_layers(16, [16])
    def forward(self, x):
        conv_1 = self.conv_1(x)  # 3-16
        conv_2 = self.conv_2(conv_1)  # 16-16
        r = x[:,0:1,:,:]   #R通道
        g = x[:,1:2,:,:]   #G通道
        b = x[:,2:3,:,:]   #B通道
        y = (r + g)/2      #Y通道
        result = b - y
        return conv_2, result
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

class VGG_channel1(nn.Module):
    def __init__(self):
        super(VGG_channel1, self).__init__()

        self.sj1 = self.make_layers_1(16, [16], rate=5)
        self.sj2 = self.make_layers_1(16, [16], rate=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lg1 = self.make_layers_1(16, [64], rate=5)
        self.lg2 = self.make_layers_1(64, [64], rate=5)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sp1 = self.make_layers_1(64, [128], rate=5)
        self.sp2 = self.make_layers_1(128, [128], rate=5)

        # self.channe_conv1 = self.make_layers_2(128, [128])

    def forward(self, x):
        sj1 = self.sj1(x)
        sj2 = self.sj2(sj1)
        pool1 = self.pool1(sj2)

        lg1 = self.lg1(pool1)
        lg2 = self.lg2(lg1)
        pool2 = self.pool2(lg2)

        sp1 = self.sp1(pool2)
        sp2 = self.sp2(sp1)
        # channe_conv1 = self.channe_conv1(sp2)
        return sj2, lg2, sp2
    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=5, padding=2, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=rate, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_2(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class VGG_channel2(nn.Module):
    def __init__(self):
        super(VGG_channel2, self).__init__()
        #数字卷积核个数，M代表池化层
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        #VGG16的13个卷积层
        self.stage1 = self.make_layers(16, [16, 16])
        self.stage2 = self.make_layers(16, ['M', 64, 64])
        self.stage3 = self.make_layers(64, ['M', 128, 128])

        # self.channel2_conv1 = self.make_layers1(128, [128])

    #正向传播过程
    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        # channel2_conv1 = self.channel2_conv1(stage3)
        return stage1, stage2, stage3

    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class VGG_channel3(nn.Module):
    def __init__(self):
        super(VGG_channel3, self).__init__()
        #数字卷积核个数，M代表池化层
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        #VGG16的13个卷积层
        # self.stage1 = self.make_layers(1, [16, 16])
        # self.stage2 = self.make_layers(16, ['M', 64, 64])
        # self.stage3 = self.make_layers(64, ['M', 128, 128])
        #
        # self.channel2_conv1 = self.make_layers1(128, [128])
        self.sj1 = self.make_layers_1(1, [16], rate=1)
        self.sj2 = self.make_layers_1(16, [16], rate=5)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lg1 = self.make_layers_1(16, [64], rate=1)
        self.lg2 = self.make_layers_1(64, [64], rate=5)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sp1 = self.make_layers_1(64, [128], rate=1)
        self.sp2 = self.make_layers_1(128, [128], rate=5)
    #正向传播过程
    def forward(self, x):
        sj1 = self.sj1(x)
        sj2 = self.sj2(sj1)
        pool1 = self.pool1(sj2)

        lg1 = self.lg1(pool1)
        lg2 = self.lg2(lg1)
        pool2 = self.pool2(lg2)

        sp1 = self.sp1(pool2)
        sp2 = self.sp2(sp1)

        # stage1 = self.stage1(x)
        # stage2 = self.stage2(stage1)
        # stage3 = self.stage3(stage2)
        # channel2_conv1 = self.channel2_conv1(stage3)
        return sj2, lg2, sp2

    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=rate, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
#权重卷积块对应不同的流，其中一个使用sigmod函数激活，再融合
class adap_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(adap_conv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True)])
        # self.conv = ResNeXtBottleNeck(in_channels, out_channels, D, cardinality=groups)
        self.weight = nn.Parameter(torch.Tensor([0.]))
    def forward(self, x):
        x = self.conv(x) * self.weight.sigmoid()
        # x = self.conv(x)
        return x
#特征图融合，细化块把低分辨率特征图上采样高分辨率图
class Refine_block2_1(nn.Module):
    def __init__(self, in_channel, out_channel, factor, require_grad=False):
        super(Refine_block2_1, self).__init__()
        self.pre_conv1 = adap_conv(in_channel[0], out_channel,)
        self.pre_conv2 = adap_conv(in_channel[1], out_channel,)
        self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(factor, out_channel), requires_grad=require_grad)
        self.factor = factor
    def forward(self, *input):
        x1 = self.pre_conv1(input[0])
        x2 = self.pre_conv2(input[1])
        x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
                                output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
        return x1 + x2
class super_pixels(nn.Module):
    def __init__(self, inplanes, factor):
        super(super_pixels, self).__init__()
        self.superpixels = nn.PixelShuffle(factor) #伸缩
        planes = int(inplanes/(factor*2))
        self.down_sample = nn.Conv2d(planes, 1, kernel_size=2, stride=2, padding=0)
    def forward(self, x):
        x = self.superpixels(x)
        x = self.down_sample(x)
        return x
class VGG_fusion(nn.Module):
    def __init__(self):
        super(VGG_fusion, self).__init__()
        self.encode0 = VGG_together()
        self.encode1 = VGG_channel1()
        self.encode2 = VGG_channel2()
        self.encode3 = VGG_channel3()
    def forward(self, x):
        end_points0 = self.encode0(x)
        end_points = self.encode1(end_points0[0])
        end_points1 = self.encode2(end_points0[0])
        end_points2 = self.encode3(end_points0[1])
        # end_point_out = end_points[1] + end_points1[1] + end_points2[1]
        end_point_out1 = end_points[0] + end_points1[0] + end_points2[0]
        end_point_out2 = end_points[1] + end_points1[1] + end_points2[1]
        end_point_out3 = end_points[2] + end_points1[2] + end_points2[2]
        return end_point_out1, end_point_out2, end_point_out3
    @staticmethod
    #定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
#VGG的5个输出，每个输出三个，分别为不变，下采样，上采样结合3*3不变 1*1升维
class decode(nn.Module):
    def __init__(self):
        super(decode, self).__init__()
        #     # 解码网络可修改的地方，弄懂参数，D，groups是残差网络模块中的参数对应intermediate,cardinality。
        self.conv_1_1 = nn.Conv2d(16, 16, kernel_size=1, padding=0)
        self.conv_1_2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv_1_3 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        # self.level1 = Refine_block2_1((16, 32), 16, 2)
        # self.level2 = Refine_block2_1((32, 64), 32, 2)#2上采样1不变
        # self.level3 = Refine_block2_1((16, 32), 16, 2)
        self.level1 = Refine_block2_1((16, 32), 16, 2)
        self.level2 = Refine_block2_1((16, 64), 16, 4)
        # self.level2 = Refine_block2_1((64, 128), 64, 2)  # 2上采样1不变
        # self.level3 = Refine_block2_1((16, 64), 16, 2)
        self.conv_1_4 = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):
        conv_1_1 = self.conv_1_1(input[0])
        conv_1_2 = self.conv_1_2(input[1])
        conv_1_3 = self.conv_1_3(input[2])
        level1 = self.level1(conv_1_1, conv_1_2)
        level2 = self.level2(conv_1_1, conv_1_3)
        # level1 = self.level1(conv_1_1, conv_1_2)
        # level2 = self.level2(conv_1_2, conv_1_3)
        # level3 = self.level3(level1, level2)
        # level1 = self.level1(conv_1_1,  conv_1_2)
        # level2 = self.level2(conv_1_2,  conv_1_3)
        # level3 = self.level3(level1, level2)
        sum = level1 + level2
        conv_1_4 = self.conv_1_4(sum)
        return conv_1_4
        # return level5
    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=0, stride=stride)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    # 定义的实现卷积层的模块化函数
    def make_layers_1(in_channels, cfg, stride=1, rate=1):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=stride, dilation=rate)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
#VGG的5个输出，每个输出三个，分别为不变，下采样，上采样结合3*3不变 -3*3升维
class DRNet(nn.Module):
    def __init__(self):
        super(DRNet, self).__init__()
        self.encode = VGG_fusion()
        self.decode = decode()
    def forward(self, x):
        end_points = self.encode(x)  #输出特征
        x = self.decode(*end_points).sigmoid()  # 把输出的特征进行反卷积解码
        return x,x
class DRNet1(nn.Module):
    def __init__(self, cfgs):
        super(DRNet1, self).__init__()
        self.encode = VGG13_qz1(cfgs)
        # self.encode = VGG1(cfgs)
        self.double_decode5 = double_decode5()

    def forward(self, x):
        end_points = self.encode(x)  # 输出特征
        x = self.double_decode5(*end_points).sigmoid()  # 把输出的特征进行反卷积解码
        return x

class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]))
        self.weight2 = nn.Parameter(torch.Tensor([1.]))
    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        # total_loss = cross_entropy_per_image(pred, labels)
        # # total_loss = dice_loss_per_image(pred, labels)
        total_loss = 1.00 * cross_entropy_per_image(pred, labels) + \
                     0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg
class Cross_Entropy1(nn.Module):
    def __init__(self):
        super(Cross_Entropy1, self).__init__()
        self.weight1 = nn.Parameter(torch.Tensor([1.]))
        self.weight2 = nn.Parameter(torch.Tensor([1.]))
    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        # total_loss = cross_entropy_per_image1(pred, labels)
        # # total_loss = dice_loss_per_image(pred, labels)
        total_loss = 1.00 * cross_entropy_per_image1(pred, labels) + \
                     0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg
def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = dice.pow(-1)
    return dice_loss

def dice_loss_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_per_image(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_orignal(_logit, _label)
    return total_loss / len(logits)
def cross_entropy_per_image1(logits, labels):
    total_loss = 0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_orignal(_logit, _label)
    return total_loss / len(logits)
def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.0].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.17, 5.0)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                            (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy
def cross_entropy_with_weight_original(logits, labels, threshold=0.2, weight=1.0):
    logits = logits.view(-1)
    labels = labels.view(-1)  #view(-1)变成一个行向量
    eps = 1e-6
    pred_pos = logits[labels > threshold].clamp(eps, 1.0-eps)    #钳位
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    weight_pos = len(pred_neg)/(len(pred_neg)+len(pred_pos))
    weight_neg = len(pred_pos)/(len(pred_neg)+len(pred_pos))
    cross_entropy = (-weight_pos * pred_pos.log()).sum() + (-(1.4 * weight_neg) * (1.0 - pred_neg).log()).sum()

    return cross_entropy
def cross_entropy_with_weight_original1(logits, labels, threshold=0.2, weight=1.0):
    logits = logits.view(-1)
    labels = labels.view(-1)  #view(-1)变成一个行向量
    eps = 1e-6
    pred_pos = logits[labels > threshold].clamp(eps, 1.0-eps)    #钳位
    pred_neg = logits[labels == 0].clamp(eps, 1.0-eps)
    weight_pos = len(pred_neg)/(len(pred_neg)+len(pred_pos))
    weight_neg = len(pred_pos)/(len(pred_neg)+len(pred_pos))
    cross_entropy = (-weight_pos * pred_pos.log()).sum() + (-(1.4 * weight_neg) * (1.0 - pred_neg).log()).sum()

    return cross_entropy
def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps,1.0-eps)
    pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
    w_anotation = labels[labels > 0]
    # weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)
    cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    # cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
    #                     (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy

def get_weight(src, mask, threshold, weight):
    count_pos = src[mask >= threshold].size()[0]
    count_neg = src[mask == 0.0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg

def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))