import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, **kwargs):
        super().__init__()
        self.DepthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3,
                      padding=1, stride=stride, groups=inchannels, bias=False, **kwargs),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)

        )
        self.PointwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.DepthwiseConv(x)
        out = self.PointwiseConv(out)
        return out


class FullConv(nn.Module):
    def __init__(self, inchannels, outchannels, stride, **kwargs):
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, bias=False, kernel_size=3,
                      padding=1, stride=stride, **kwargs),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.Conv(x)
        return out


class MobileNet32(nn.Module):
    def __init__(self, alpha=1, num_calss=100):
        super().__init__()
        self.alpha = alpha
        self.num_class = num_calss
        # Conv separated by down sampling
        self.Conv1 = nn.Sequential(
            FullConv(inchannels=3, outchannels=int(alpha*32), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*32), outchannels=int(alpha*64), stride=1)
        )
        self.Conv2 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*64), outchannels=int(alpha*128), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*128), outchannels=int(alpha*128), stride=1)
        )
        self.Conv3 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*128), outchannels=int(alpha*256), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*256), outchannels=int(alpha*256), stride=1)
        )

        self.Conv4 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*256), outchannels=int(alpha*512), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1),
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*512), stride=1)
        )

        self.Conv5 = nn.Sequential(
            DepthwiseSeparableConv(inchannels=int(alpha*512), outchannels=int(alpha*1024), stride=2),
            DepthwiseSeparableConv(inchannels=int(alpha*1024), outchannels=int(alpha*1024), stride=1)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.FC = nn.Linear(int(alpha*1024), num_calss)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Conv2(out)
        out = self.Conv3(out)
        out = self.Conv4(out)
        out = self.Conv5(out)
        out = self.AvgPool(out)
        out = out.view(x.size(0), -1)
        out = self.drop(out)
        out = self.FC(out)
        return out


def mobilenet(alpha=1, num_class=100):
    return MobileNet32(alpha, num_class)