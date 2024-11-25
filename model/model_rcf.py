import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import math
import utils


class VGG13(nn.Module):
    def __init__(self):
        super(VGG13, self).__init__()

        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.stage1 = self.make_layers(3, [64])
        self.stage2 = self.make_layers(64, [64])
        self.stage3 = self.make_layers(64, ['M', 128])
        self.stage4 = self.make_layers(128, [128])
        self.stage5 = self.make_layers(128, ['M', 256])
        self.stage6 = self.make_layers(256, [256])
        self.stage7 = self.make_layers(256, [256])
        self.stage8 = self.make_layers(256, ['M', 512])
        self.stage9 = self.make_layers(512, [512])
        self.stage10 = self.make_layers(512, [512])
        self.stage11 = self.make_layers(512, ['M', 512])
        self.stage12 = self.make_layers(512, [512])
        self.stage13 = self.make_layers(512, [512])

        # self._initialize_weights(pertrain)

    def forward(self, x):

        # stage1 = self.stage1(x)
        # stage2 = self.stage2(stage1)
        # stage3 = self.stage3(stage2)
        # stage4 = self.stage4(stage3)
        # stage5 = self.stage5(stage4)
        # stage6 = self.stage6(stage5)
        # stage7 = self.stage7(stage6)
        # stage8 = self.stage8(stage7)
        # stage9 = self.stage9(stage8)
        # stage10 = self.stage10(stage9)
        # stage11 = self.stage11(stage10)
        # stage12 = self.stage12(stage11)
        # stage13 = self.stage13(stage12)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2 + stage1)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4 + stage3)
        stage6 = self.stage6(stage5)
        stage7 = self.stage7(stage6 + stage5)
        stage8 = self.stage8(stage7 + stage5 + stage6)
        stage9 = self.stage9(stage8)
        stage10 = self.stage10(stage9 + stage8)
        stage11 = self.stage11(stage10 + stage9 + stage8)
        stage12 = self.stage12(stage11)
        stage13 = self.stage13(stage12 + stage11)

        return stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8, stage9, stage10, stage11, stage12, stage13

    @staticmethod
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

    def _initialize_weights(self, dict_path):
        model_paramters = torch.load(dict_path)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = (model_paramters.popitem(last=False)[-1])
                m.bias.data = model_paramters.popitem(last=False)[-1]


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
        return x


# class sum1_1(nn.Module):
#     def __init__(self, in_channel, out_channel,  require_grad=False):
#         super(sum1_1, self).__init__()
#         # self.pre_conv1 = adap_conv(in_channel[0], out_channel)
#         # self.pre_conv2 = adap_conv(in_channel[1], out_channel)
#         # self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(2, (21,21)), requires_grad=require_grad)
#         # self.factor = factor
#     def forward(self, *input):
#         # x1 = input[0]
#         # x2 = input[1]
#
#         x1 = self.pre_conv1(input[0])
#         x2 = self.pre_conv2(input[1])
#         # print(x1.size(2), x2.size(2) * self.factor)
#         x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
#                                 output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
#         return x1 + x2

# class sum3_1(nn.Module):
#     def __init__(self, in_channel, out_channel,require_grad=False):
#         super(sum3_1, self).__init__()
#         self.pre_conv1 = adap_conv(in_channel[0], out_channel)
#         self.pre_conv2 = adap_conv(in_channel[1], out_channel)
#         self.pre_conv3 = adap_conv(in_channel[2], out_channel)
#         self.deconv_weight1 = nn.Parameter(utils.bilinear_upsample_weights(2, (21,21)), requires_grad=require_grad)
#         self.deconv_weight2 = nn.Parameter(utils.bilinear_upsample_weights(4, (21,21)), requires_grad=require_grad)

# def forward(self, *input):
#     # x1 = input[0]
#     # x2 = input[1]
#     # x3 = input[2]
#
#     x1 = self.pre_conv1(input[0])
#     x2 = self.pre_conv2(input[1])
#     x3 = self.pre_conv3(input[2])
#     x2 = F.conv_transpose2d(x2, self.deconv_weight1, stride=2, padding=1,
#                             output_padding=(x1.size(2) - x2.size(2)*2, x1.size(3) - x2.size(3)*2))
#     x3 = F.conv_transpose2d(x3, self.deconv_weight2, stride=4, padding=2,
#                             output_padding=(x1.size(2) - x3.size(2) * 4, x1.size(3) - x3.size(3) * 4))
#     return x1 + x2 + x3


# class Refine_block2_1(nn.Module):
#     def __init__(self, in_channel, out_channel, factor,  require_grad=False):
#         super(Refine_block2_1, self).__init__()
#         self.pre_conv1 = adap_conv(in_channel[0], out_channel)
#         self.pre_conv2 = adap_conv(in_channel[1], out_channel)
#         self.deconv_weight = nn.Parameter(utils.bilinear_upsample_weights(2, 21), requires_grad=require_grad)
#         self.factor = factor
#     def forward(self, *input):
#         x1 = self.pre_conv1(input[0])
#         x2 = self.pre_conv2(input[1])
#         x2 = F.conv_transpose2d(x2, self.deconv_weight, stride=self.factor, padding=int(self.factor/2),
#                                 output_padding=(x1.size(2) - x2.size(2)*self.factor, x1.size(3) - x2.size(3)*self.factor))
#         return x1 + x2
#
# class Refine_block3_1(nn.Module):
#     def __init__(self, in_channel, out_channel,  require_grad=False):
#         super(Refine_block3_1, self).__init__()
#         self.pre_conv1 = adap_conv(in_channel[0], out_channel)
#         self.pre_conv2 = adap_conv(in_channel[1], out_channel)
#         self.pre_conv3 = adap_conv(in_channel[2], out_channel)
#         self.deconv_weight1 = nn.Parameter(utils.bilinear_upsample_weights(2, 21), requires_grad=require_grad)
#         self.deconv_weight2 = nn.Parameter(utils.bilinear_upsample_weights(4, 21), requires_grad=require_grad)
#
#     def forward(self, *input):
#         x1 = self.pre_conv1(input[0])
#         x2 = self.pre_conv2(input[1])
#         x3 = self.pre_conv3(input[2])
#
#         x2 = F.conv_transpose2d(x2, self.deconv_weight1, stride=2, padding=1,
#                                 output_padding=(x1.size(2) - x2.size(2)*2, x1.size(3) - x2.size(3)*2))
#         x3 = F.conv_transpose2d(x3, self.deconv_weight2, stride=4, padding=2,
#                                 output_padding=(x1.size(2) - x3.size(2) * 4, x1.size(3) - x3.size(3) * 4))
#         return x1 + x2 + x3


class super_pixels(nn.Module):
    def __init__(self, inplanes, factor):
        super(super_pixels, self).__init__()
        self.superpixels = nn.PixelShuffle(factor)
        planes = int(inplanes / (factor * 2))
        self.down_sample = nn.Conv2d(planes, 1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.superpixels(x)
        x = self.down_sample(x)
        return x


class decode(nn.Module):
    def __init__(self):
        super(decode, self).__init__()

        # 1*1*21卷积
        self.stage1_1 = nn.Conv2d(64, 21, kernel_size=1)
        self.stage2_1 = nn.Conv2d(64, 21, kernel_size=1)
        self.stage3_1 = nn.Conv2d(128, 21, kernel_size=1)
        self.stage4_1 = nn.Conv2d(128, 21, kernel_size=1)
        self.stage5_1 = nn.Conv2d(256, 21, kernel_size=1)
        self.stage6_1 = nn.Conv2d(256, 21, kernel_size=1)
        self.stage7_1 = nn.Conv2d(256, 21, kernel_size=1)
        self.stage8_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.stage9_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.stage10_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.stage11_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.stage12_1 = nn.Conv2d(512, 21, kernel_size=1)
        self.stage13_1 = nn.Conv2d(512, 21, kernel_size=1)

        self.conv1 = nn.Conv2d(21, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(21, 1, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(21, 1, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(21, 1, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(21, 1, kernel_size=1, padding=0)

        self.deconv2 = nn.Parameter(utils.bilinear_upsample_weights(2, (1, 1)), requires_grad=True)
        self.deconv3 = nn.Parameter(utils.bilinear_upsample_weights(4, (1, 1)), requires_grad=True)
        self.deconv4 = nn.Parameter(utils.bilinear_upsample_weights(8, (1, 1)), requires_grad=True)
        self.deconv5 = nn.Parameter(utils.bilinear_upsample_weights(16, (1, 1)), requires_grad=True)

        self.conv6 = nn.Conv2d(5, 1, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, 0, 1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, *input):
        input_0 = self.stage1_1(input[0])
        input_1 = self.stage2_1(input[1])
        input_2 = self.stage3_1(input[2])
        input_3 = self.stage4_1(input[3])
        input_4 = self.stage5_1(input[4])
        input_5 = self.stage6_1(input[5])
        input_6 = self.stage7_1(input[6])
        input_7 = self.stage8_1(input[7])
        input_8 = self.stage9_1(input[8])
        input_9 = self.stage10_1(input[9])
        input_10 = self.stage11_1(input[10])
        input_11 = self.stage12_1(input[11])
        input_12 = self.stage13_1(input[12])

        level1 = input_0 + input_1
        level2 = input_2 + input_3
        level3 = input_4 + input_5 + input_6
        level4 = input_7 + input_8 + input_9
        level5 = input_10 + input_11 + input_12

        s1 = self.conv1(level1)
        s2 = self.conv2(level2)
        s3 = self.conv3(level3)
        s4 = self.conv4(level4)
        s5 = self.conv5(level5)

        # s2 = F.conv_transpose2d(s2, self.deconv2, stride=2, padding=1, output_padding=s1.size(2) - s2.size(2) * 2)
        # s3 = F.conv_transpose2d(s3, self.deconv3, stride=4, padding=2, output_padding=s1.size(2) - s3.size(2) * 4)
        # s4 = F.conv_transpose2d(s4, self.deconv4, stride=8, padding=4, output_padding=s1.size(2) - s4.size(2) * 8)
        # s5 = F.conv_transpose2d(s5, self.deconv5, stride=16, padding=8, output_padding=s1.size(2) - s5.size(2) * 16)
        s2 = F.conv_transpose2d(s2, self.deconv2, stride=2, padding=1, output_padding=(s1.size(2) - s2.size(2) * 2, s1.size(3) - s2.size(3) * 2))
        s3 = F.conv_transpose2d(s3, self.deconv3, stride=4, padding=2, output_padding=(s1.size(2) - s3.size(2) * 4, s1.size(3) - s3.size(3) * 4))
        s4 = F.conv_transpose2d(s4, self.deconv4, stride=8, padding=4, output_padding=(s1.size(2) - s4.size(2) * 8, s1.size(3) - s4.size(3) * 8))
        s5 = F.conv_transpose2d(s5, self.deconv5, stride=16, padding=8, output_padding=(s1.size(2) - s5.size(2) * 16, s1.size(3) - s5.size(3) * 16))

        return self.conv6(torch.cat([s1, s2, s3, s4, s5],
                                    dim=1)).sigmoid(), s1.sigmoid(), s2.sigmoid(), \
            s3.sigmoid(), s4.sigmoid(), s5.sigmoid()


class DRNet(nn.Module):
    def __init__(self):
        super(DRNet, self).__init__()
        # 5层
        self.encode = VGG13()
        self.decode = decode()

    def forward(self, x):
        end_points = self.encode(x)
        x = self.decode(*end_points)
        return x[0], x[1:6]


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
        # total_loss = dice_loss_per_image(pred, labels)
        total_loss = 1.00 * cross_entropy_per_image(pred, labels) + \
                     0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        # return total_loss, (1-pred_pos).abs(), pred_neg
        # total_loss = cross_entropy_per_image(pred, labels) + \
        #              0.2 * cross_entropy_per_image(end_points[0], labels) + \
        #              0.2 * cross_entropy_per_image(end_points[1], labels) + \
        #              0.2 * cross_entropy_per_image(end_points[2], labels) + \
        #              0.2 * cross_entropy_per_image(end_points[3], labels) + \
        #              0.2 * cross_entropy_per_image(end_points[4], labels)
        return total_loss, (1 - pred_pos).abs(), pred_neg


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


def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.5].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.5, 1.5)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                    (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy


def cross_entropy_with_weight(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)
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