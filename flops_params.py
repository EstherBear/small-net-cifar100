from my_thop import profile
from my_thop import clever_format
from models.mobilenet import mobilenet
from models.mobilenetv2 import mobilenetv2
from models.shufflenet import shufflenet
from models.shufflenetv2 import shufflenetv2
from models.efficientnet import efficientnet
from tools.utils import SameConv
import torch
from torchsummary import summary
nets = ["mobilenet(1, 100)", "mobilenetv2(1, 100)", "shufflenet([4, 8, 4], 3, 1, 100)", "shufflenetv2(100, 1)",
        "efficientnet(1, 1, 100, bn_momentum=0.9)"]


def count_your_model(model, x, y):
    return y.size(2) * y.size(3) * y.size(1) * model.weight.size(2) * model.weight.size(3) / model.groups

# custom_ops={SameConv: count_your_model


x = torch.randn(1, 3, 32, 32)
for net_name in nets:
    net = eval(net_name)
    macs, params = profile(model=net, inputs=(x, ))
    flops, params = clever_format([macs, params], "%.3f")
    print(net_name+": ", flops, params)
    # summary(net.cuda(), (3, 32, 32))

