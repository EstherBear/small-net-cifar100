import torch.nn as nn
import math
from tools.utils import conv_bn_act
from tools.utils import SameConv
from tools.utils import SE
from tools.utils import drop_connect
from tools.utils import swish


class MBConv(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio=4,
                 is_skip=True, dc_ratio=(1-0.8), bn_momentum=0.90):
        super().__init__()
        mid = expan * inchannels
        self.pointwise1 = conv_bn_act(inchannels, mid, 1) if expan != 1 else nn.Identity()
        self.depthwise = conv_bn_act(mid, mid, kernelsize, stride=stride, groups=mid)
        self.se = SE(mid, int(inchannels/se_ratio))
        self.pointwise2 = nn.Sequential(
            SameConv(mid, outchannels, 1),
            nn.BatchNorm2d(outchannels, 1-bn_momentum)
        )
        self.skip = is_skip and inchannels == outchannels and stride == 1
        # self.dc = drop_connect(1-dc_ratio)
        self.dc = nn.Identity()

    def forward(self, x):
        residual = self.pointwise1(x)
        residual = self.depthwise(residual)
        residual = self.se(residual)
        residual = self.pointwise2(residual)
        if self.skip:
            residual = self.dc(residual)
            out = residual + x
        else:
            out = residual

        return out


class MBblock(nn.Module):
    def __init__(self, inchannels, outchannels, expan, kernelsize, stride, se_ratio, repeat,
                 is_skip, dc_ratio=(1-0.8), bn_momentum=0.90):
        super().__init__()

        layers = []
        layers.append(MBConv(inchannels, outchannels, expan, kernelsize, stride,
                             se_ratio, is_skip, dc_ratio, bn_momentum))
        while repeat-1:
            layers.append(MBConv(outchannels, outchannels, expan, kernelsize, 1,
                                 se_ratio, is_skip, dc_ratio, bn_momentum))
            repeat = repeat - 1

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EfficientNet(nn.Module):
    def __init__(self, width_multipler, depth_multipler, do_ratio, min_width=0, width_divisor=8,
                 se_ratio=4, dc_ratio=(1-0.8), bn_momentum=0.90, num_class=100):
        super().__init__()

        def renew_width(x):
            min = max(min_width, width_divisor)
            x *= width_multipler
            new_x = max(min, int((x + width_divisor/2) // width_divisor * width_divisor))

            if new_x < 0.9 * x:
                new_x += width_divisor
            return int(new_x)

        def renew_depth(x):
            return int(math.ceil(x * depth_multipler))

        self.stage1 = nn.Sequential(
            SameConv(3, renew_width(32), 3),
            nn.BatchNorm2d(renew_width(32), momentum=bn_momentum),
            swish()
        )
        self.stage2 = nn.Sequential(
                    # inchannels     outchannels  expand k  s(mobilenetv2)  repeat      is_skip
            MBblock(renew_width(32), renew_width(16), 1, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum),
            MBblock(renew_width(16), renew_width(24), 6, 3, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum),
            MBblock(renew_width(24), renew_width(40), 6, 5, 2, se_ratio, renew_depth(2), True, dc_ratio, bn_momentum),
            MBblock(renew_width(40), renew_width(80), 6, 3, 2, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum),
            MBblock(renew_width(80), renew_width(112), 6, 5, 1, se_ratio, renew_depth(3), True, dc_ratio, bn_momentum),
            MBblock(renew_width(112), renew_width(192), 6, 5, 1, se_ratio, renew_depth(4), True, dc_ratio, bn_momentum),
            MBblock(renew_width(192), renew_width(320), 6, 3, 1, se_ratio, renew_depth(1), True, dc_ratio, bn_momentum)
        )
        #print("initing stage 3")
        self.stage3 = nn.Sequential(
            SameConv(renew_width(320), renew_width(1280), 1, stride=1),
            nn.BatchNorm2d(renew_width(1280), bn_momentum),
            swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(do_ratio)
        )
        self.FC = nn.Linear(renew_width(1280), num_class)
        #print("initing weights")

        self.init_weights()
        #print("finish initing")

    def init_weights(self):
        # SameConv用kaiming, Linear用1/sqrt(channels)
        for m in self.modules():
            if isinstance(m, SameConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.Linear):
                bound = 1/int(math.sqrt(m.weight.size(1)))
                nn.init.uniform(m.weight, -bound, bound)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        return out


def efficientnet(width_multipler, depth_multipler, num_class=100, bn_momentum=0.90, do_ratio=0.2):
    return EfficientNet(width_multipler, depth_multipler,
                        num_class=num_class, bn_momentum=bn_momentum, do_ratio=do_ratio)




