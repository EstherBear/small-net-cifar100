import torch.nn as nn
import torch
import sys
def channel_split(x, chunks):
    return torch.split(x, int(x.size(1)/chunks), dim=1)

def channel_shuffle(x, groups):
    batch_size, C, H, W = x.size()
    out = x.view(batch_size, groups, int(C/groups), H, W)
    out = out.transpose(1, 2).contiguous()
    out = out.view(batch_size, C, H, W)

    return out


class ShuffleNetV2Unit(nn.Module):
    def __init__(self, inchannels, outchannels, stride):
        super().__init__()

        self.stride = stride
        self.inchannels = inchannels
        self.outchannels = outchannels

        if inchannels != outchannels or stride != 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inchannels, inchannels, 1, bias=False),
                nn.BatchNorm2d(inchannels),
                nn.ReLU(inplace=True),

                nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1,
                          groups=inchannels, stride=stride, bias=False),
                nn.BatchNorm2d(inchannels),

                nn.Conv2d(inchannels, int(outchannels/2), 1, bias=False),
                nn.BatchNorm2d(int(outchannels/2)),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1,
                          groups=inchannels, stride=stride, bias=False),
                nn.BatchNorm2d(inchannels),

                nn.Conv2d(inchannels, int(outchannels / 2), 1, bias=False),
                nn.BatchNorm2d(int(outchannels / 2)),
                nn.ReLU(inplace=True),
            )
        else:
            inchannels = int(inchannels/2)
            self.branch1 = nn.Sequential(
                nn.Conv2d(inchannels, inchannels, 1, bias=False),
                nn.BatchNorm2d(inchannels),
                nn.ReLU(inplace=True),

                nn.Conv2d(inchannels, inchannels, kernel_size=3, padding=1,
                          groups=inchannels, stride=stride, bias=False),
                nn.BatchNorm2d(inchannels),

                nn.Conv2d(inchannels, inchannels, 1, bias=False),
                nn.BatchNorm2d(inchannels),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential()

    def forward(self, x):
        if self.inchannels != self.outchannels or self.stride != 1:
            shortcut = x
            residual = x
        else:
            shortcut, residual = channel_split(x, 2)

        shortcut = self.branch2(shortcut)
        residual = self.branch1(residual)

        out = torch.cat((shortcut, residual), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_class, scale):
        super().__init__()

        if scale == 0.5:
            outchannels = [48, 96, 192, 1024]
        elif scale == 1:
            outchannels = [116, 232, 464, 1024]
        elif scale == 1.5:
            outchannels = [176, 352, 704, 1024]
        elif scale == 2:
            outchannels = [244, 488, 976, 2048]
        else:
            print("This scale is not supported!")
            sys.exit()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.inchannels = 24
        self.Stage2 = self.make_layer(outchannels[0], 2, 3)
        self.Stage3 = self.make_layer(outchannels[1], 2, 7)
        self.Stage4 = self.make_layer(outchannels[2], 2, 3)
        self.Conv5 = nn.Sequential(
            nn.Conv2d(outchannels[2], outchannels[3], 1, bias=False),
            nn.BatchNorm2d(outchannels[3]),
            nn.ReLU(inplace=True)
        )
        self.GlobalPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.FC = nn.Linear(outchannels[3], num_class)

    def make_layer(self, outchannels, stride, repeat):
        layer = []
        layer.append(ShuffleNetV2Unit(self.inchannels, outchannels, stride))
        self.inchannels = outchannels
        while repeat:
            layer.append(ShuffleNetV2Unit(self.inchannels, outchannels, 1))
            repeat -= 1
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.Stage2(out)
        out = self.Stage3(out)
        out = self.Stage4(out)
        out = self.Conv5(out)
        out = self.GlobalPool(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.FC(out)
        return out


def shufflenetv2(num_class=100, scale=1):
    return ShuffleNetV2(num_class, scale)
