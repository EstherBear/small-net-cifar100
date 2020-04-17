import torch.nn as nn


class bottleneck(nn.Module):
    def __init__(self, inchannels, outchannels, stride, expansion):
        super().__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.stride = stride

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=expansion*inchannels, kernel_size=1),
            nn.BatchNorm2d(expansion*inchannels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=inchannels*expansion, out_channels=inchannels*expansion, kernel_size=3, padding=1,
                      groups=inchannels*expansion, stride=stride),
            nn.BatchNorm2d(expansion*inchannels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=expansion * inchannels, out_channels=outchannels, kernel_size=1),
            nn.BatchNorm2d(outchannels)
        )

    def forward(self, x):
        out = self.residual(x)

        if self.inchannels == self.outchannels and self.stride == 1:
            out += x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, alpha=1, num_class=100):
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(alpha*32), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = bottleneck(int(alpha*32), 16, 1, 1)
        self.stage2 = self.make_layer(int(alpha*16), 6, int(alpha*24), 2, 2)
        self.stage3 = self.make_layer(int(alpha*24), 6, int(alpha*32), 3, 2)
        self.stage4 = self.make_layer(int(alpha*32), 6, int(alpha*64), 4, 2)
        self.stage5 = self.make_layer(int(alpha*64), 6, int(alpha*96), 3, 1)
        self.stage6 = self.make_layer(int(alpha*96), 6, int(alpha*160), 3, 1)
        self.stage7 = self.make_layer(int(alpha*160), 6, int(alpha*320), 1, 1)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(alpha*320), out_channels=1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.2)
        self.Conv3 = nn.Conv2d(in_channels=1280, out_channels=num_class, kernel_size=1)

    def make_layer(self, inchannels, t, outchannels, n, s):
        layer = []
        layer.append(bottleneck(inchannels, outchannels, s, t))
        n = n - 1
        while n:
            layer.append(bottleneck(outchannels, outchannels, 1, t))
            n -= 1
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.Conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = self.stage6(out)
        out = self.stage7(out)
        out = self.Conv2(out)
        out = self.AvgPool(out)
        out = self.drop(out)
        out = self.Conv3(out)
        out = out.view(out.size(0), -1)

        return out


def mobilenetv2(alpha=1, num_class=100):
    return MobileNetV2(alpha, num_class)