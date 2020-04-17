import torch.nn as nn
import torch
import sys

class PointwiseConv(nn.Module):
    def __init__(self, inchannels, outchannels, **kwargs):
        super().__init__()
        self.pointwise=nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, bias=False, **kwargs),
            nn.BatchNorm2d(outchannels)
        )

    def forward(self, x):
        out = self.pointwise(x)
        return out


class DepthwiseConv(nn.Module):
    def __init__(self, inchannels, stride, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, bias=False, stride=stride,
                      padding=1, groups=inchannels,  **kwargs),
            nn.BatchNorm2d(inchannels)
        )

    def forward(self, x):
        out = self.depthwise(x)
        return out


def ChannelShuffle(input, groups):
    batch_size, C, H, W = input.size()
    output = input.view(batch_size, groups, int(C/groups), H, W)
    output = output.transpose(1, 2).contiguous()
    output = output.view(batch_size, -1, H, W)
    return output


class ShuffleNetUnit(nn.Module):
    def __init__(self, inchannels, outchannels, stride, groups, stage):
        super().__init__()
        if stage == 2:
            self.GConv1 = nn.Sequential(
                PointwiseConv(inchannels, int(outchannels/4), groups=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.GConv1 = nn.Sequential(
                PointwiseConv(inchannels, int(outchannels / 4), groups=groups),
                nn.ReLU(inplace=True)
            )

        self.shuffle = ChannelShuffle

        self.DWConv = DepthwiseConv(int(outchannels / 4), stride)

        if stride != 1 or inchannels != outchannels:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            self.fusion = self._Concat
            self.GConv2 = PointwiseConv(int(outchannels / 4), outchannels-inchannels, groups=groups)
        else:
            self.shortcut = nn.Sequential()
            self.fusion = self._Add
            self.GConv2 = PointwiseConv(int(outchannels / 4), outchannels, groups=groups)

        self.relu = nn.ReLU(inplace=True)

        self.groups = groups

    def _Concat(self, x, y):
        return torch.cat((x, y), dim=1)

    def _Add(self, x, y):
        return x+y

    def forward(self, x):
        out = self.GConv1(x)
        out = self.shuffle(out, self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)
        out = self.fusion(self.shortcut(x), out)
        out = self.relu(out)

        return out


class ShuffleNet(nn.Module):
    def __init__(self, blocks, g, s, num_class):
        super().__init__()
        if g == 1:
            outchannels=[24, 144, 288, 576]
        elif g == 2:
            outchannels = [24, 200, 400, 800]
        elif g == 3:
            outchannels = [24, 240, 480, 960]
        elif g == 4:
            outchannels = [24, 272, 544, 1088]
        elif g == 8:
            outchannels = [24, 384, 768, 1536]
        else:
            print("This g is not supported!")
            sys.exit()

        outchannels = [int(s * outchannel) for outchannel in outchannels]

        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, outchannels[0], 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannels[0]),
            nn.ReLU(inplace=True)
        )
        self.inchannels = outchannels[0]
        self.groups = g
        self.stage2 = self.make_layer(2, 2, blocks[0], outchannels[1])
        self.stage3 = self.make_layer(3, 2, blocks[1], outchannels[2])
        self.stage4 = self.make_layer(4, 2, blocks[2], outchannels[3])
        self.GlobalPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.FC = nn.Linear(outchannels[-1], num_class)

    def forward(self, x):
        output = self.Conv1(x)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.GlobalPool(output)
        output = output.view(output.size(0), -1)
        output = self.drop(output)
        output = self.FC(output)

        return output

    def make_layer(self, stage, stride, repeat, outchannels):
        strides = [stride] + [1] * (repeat - 1)
        layer = []
        for stride in strides:
            layer.append(ShuffleNetUnit(self.inchannels, outchannels, stride, self.groups, stage))
            self.inchannels = outchannels

        return nn.Sequential(*layer)


def shufflenet(blocks, g=3, s=1, num_class=100):
    return ShuffleNet(blocks=blocks, g=g, s=s, num_class=num_class)