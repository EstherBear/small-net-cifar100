import torch.nn as nn
import torch
import torch.nn.functional as F


class conv_bn_act(nn.Module):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False, bn_momentum=0.99):
        super().__init__()
        self.block = nn.Sequential(
            SameConv(inchannels, outchannels, kernelsize, stride, dilation, groups, bias=bias),
            nn.BatchNorm2d(outchannels, momentum=1-bn_momentum),
            swish()
        )

    def forward(self, x):
        return self.block(x)


class SameConv(nn.Conv2d):
    def __init__(self, inchannels, outchannels, kernelsize, stride=1, dilation=1, groups=1, bias=False):
        super().__init__(inchannels, outchannels, kernelsize, stride,
                         padding=0, dilation=dilation, groups=groups, bias=bias)

    def how_padding(self, n, kernel, stride, dilation):
        out_size = (n + stride - 1) // stride
        real_kernel = (kernel - 1) * dilation + 1
        padding_needed = max(0, (out_size - 1) * stride + real_kernel - n)
        is_odd = padding_needed % 2
        return padding_needed, is_odd

    def forward(self, x):
        row_padding_needed, row_is_odd = self.how_padding(x.size(2), self.weight.size(2), self.stride[0], self.dilation[0])
        col_padding_needed, col_is_odd = self.how_padding(x.size(3), self.weight.size(3), self.stride[1], self.dilation[1])
        if row_is_odd or col_is_odd:
            x = F.pad(x, [0, col_is_odd, 0, row_is_odd])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        (row_padding_needed//2, col_padding_needed//2), self.dilation, self.groups)

    #def count_your_model(self, x, y):
     #   return y.size(2) * y.size(3) * y.size(1) * self.weight.size(2) * self.weight.size(3) / self.groups


class swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    def __init__(self, inchannels, mid):
        super().__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(inchannels, mid),
            swish(),
            nn.Linear(mid, inchannels)
        )

    def forward(self, x):
        out = self.AvgPool(x)
        out = out.view(x.size(0), -1)
        out = self.SEblock(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return x * torch.sigmoid(out)


class drop_connect(nn.Module):
    def __init__(self, survival=0.8):
        super().__init__()
        self.survival = survival

    def forward(self, x):
        if not self.training:
            return x

        random = torch.rand((x.size(0), 1, 1, 1), device=x.device) # 涉及到x的属性的步骤，直接挪到forward
        random += self.survival
        random.requires_grad = False
        return x / self.survival * torch.floor(random)
