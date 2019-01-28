#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from resnet import Resnet18, Resnet18Arm

## TODO: refactoring or simply split the code to two source files


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks = 3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace = True)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FCNOutput(nn.Module):
    def __init__(self, in_chan, n_classes, *args, **kwargs):
        super(FCNOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, 256, ks=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.in_chan = in_chan
        self.conv = nn.Conv2d(in_chan,
                in_chan,
                kernel_size = 1,
                bias = False)
        self.bn = nn.BatchNorm2d(in_chan)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        assert self.in_chan == x.size()[1]
        in_ten = x
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        x = torch.mul(in_ten, x)
        x = x + in_ten
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        #  self.resnet = Resnet18Arm()
        self.conv_cat = ConvBNReLU(768, 512, ks=3, stride=1, padding=1)
        self.arm16 = AttentionRefinementModule(256)
        self.arm32 = AttentionRefinementModule(512)

        self.conv_avg = nn.Conv2d(512,
                512,
                kernel_size = 1,
                bias=True)
        self.bn_avg = nn.BatchNorm2d(512)
        self.sig_avg = nn.Sigmoid()

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        #  avg = F.avg_pool2d(feat32, feat32.size()[2:])
        #  avg_atten = self.conv_avg(avg)
        #  avg_atten = self.bn_avg(avg_atten)
        #  avg_atten = self.sig_avg(avg_atten)
        #  #  feat32 = torch.mul(feat32, avg)
        #  feat32 = feat32 + avg

        #  feat32_arm = self.arm32(feat32)
        #  feat16_arm = self.arm16(feat16)

        feat32_up = F.interpolate(feat32, (H16, W16), mode='nearest')
        feat16_cat = torch.cat([feat32_up, feat16], dim=1)
        feat16_cat = self.conv_cat(feat16_cat)

        feat_out8 = F.interpolate(feat16_cat, (H8, W8), mode='nearest')

        return feat_out8, feat16, feat32
        #  return feat_out8, feat16_arm, feat32_arm

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, ks=3, stride=2, padding=1)
        self.conv2 = ConvBNReLU(32, 128, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(128, 512, ks=3, stride=2, padding=1)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)




class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=3, stride=1)
        self.conv1 = nn.Conv2d(out_chan, out_chan, 1, bias=False)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = F.relu(atten, inplace=True)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.sp = SpatialPath()
        self.ffm = FeatureFusionModule(1024, 1024)
        self.conv_out = FCNOutput(1024, n_classes)
        self.conv_out16 = FCNOutput(256, n_classes)
        self.conv_out32 = FCNOutput(512, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_cp, feat16, feat32 = self.cp(x) # 512
        feat_sp = self.sp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp)
        feat_out = self.conv_out(feat_fuse)
        #  feat_sum = feat_cp + feat_sp
        #  feat_out = self.conv_out(feat_sum)
        feat_out16 = self.conv_out16(feat16)
        feat_out32 = self.conv_out32(feat32)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear')
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear')
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear')
        return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



if __name__ == "__main__":
    net = BiSeNet(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out, out16, out32 = net(in_ten)
    #  with torch.no_grad():
    #      in_ten = torch.randn(1, 3, 1024, 2048).cuda()
    #      out, out16, out32 = net(in_ten)
    print(out.shape)
    #  print(out16.shape)
    #  print(out32.shape)

