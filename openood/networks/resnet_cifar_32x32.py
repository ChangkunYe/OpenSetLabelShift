# From https://github.com/akamaster/pytorch_resnet_cifar10
'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features, p=1):
        super().__init__()
        # p is the exponent value in the norm formulation
        self.p = p
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.uniform_(-1, 1).renorm_(p=2, dim=0, maxnorm=1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=-1).mm(F.normalize(self.weight.T, p=self.p, dim=0))
        # out = F.normalize(x, p=self.p, dim=-1).mm(F.normalize(self.weight.T, p=self.p, dim=0))
        return out

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, fc_norm: bool = False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if not fc_norm:
            self.linear = nn.Linear(64, num_classes)
        else:
            self.linear = NormedLinear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_list=False):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = F.avg_pool2d(out4, out4.size()[3])
        out6 = out5.view(out5.size(0), -1)

        out = self.linear(out6)
        feature_list = [out1, out2, out3, out4, out5]
        if return_feature:
            return out, out6
        elif return_feature_list:
            return out, feature_list
        else:
            return out

    def forward_threshold(self, x, threshold):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        out = out.clip(max=threshold)
        out = self.linear(out)

        return out

    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        if layer_index == 1:
            return out

        out = self.layer2(out)
        if layer_index == 2:
            return out

        out = self.layer3(out)
        if layer_index == 3:
            return out

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if layer_index == 4:
            return out

        raise ValueError

    def get_fc(self):
        fc = self.linear
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.linear



def resnet20(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, fc_norm=fc_norm)


def resnet32(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, fc_norm=fc_norm)


def resnet44(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, fc_norm=fc_norm)


def resnet56(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, fc_norm=fc_norm)


def resnet110(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, fc_norm=fc_norm)


def resnet1202(num_classes, fc_norm: bool = False):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, fc_norm=fc_norm)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
