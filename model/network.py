from torch import nn
import torchvision
import torch
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, resnet_name):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.__in_features = resnet.fc.in_features

    def forward(self, x):
        return self.feature_layers(x).flatten(1)

    def output_num(self):
        return self.__in_features


def resnet(resnet_name='resnet50'):
    return ResNet(resnet_name)
