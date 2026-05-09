import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, dropout=0.3):
        super(SimpleResNet, self).__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(64, 2, stride=1, use_se=False)
        self.layer2 = self._make_layer(128, 2, stride=2, use_se=False)
        self.layer3 = self._make_layer(256, 2, stride=2, use_se=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride, use_se):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, use_se))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_head(out)
        return out

class SEResNet(SimpleResNet):
    def __init__(self, num_classes=10, in_channels=1, dropout=0.3):
        super(SEResNet, self).__init__(num_classes, in_channels, dropout)
        self.in_planes = 64
        self.layer1 = self._make_layer(64, 2, stride=1, use_se=True)
        self.layer2 = self._make_layer(128, 2, stride=2, use_se=True)
        self.layer3 = self._make_layer(256, 2, stride=2, use_se=True)

class OptimizedSEResNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, dropout=0.2):
        super(OptimizedSEResNet, self).__init__()
        self.in_planes = 32

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(32, 2, stride=1, use_se=True)
        self.layer2 = self._make_layer(64, 2, stride=1, use_se=True)
        self.layer3 = self._make_layer(128, 2, stride=2, use_se=True)
        self.layer4 = self._make_layer(256, 2, stride=1, use_se=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride, use_se):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride, use_se))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_head(out)
        return out

def get_model(name, num_classes, in_channels=1, dropout=None):
    if name.lower() == 'simpleresnet':
        d = dropout if dropout is not None else 0.3
        return SimpleResNet(num_classes, in_channels, dropout=d)
    elif name.lower() == 'seresnet':
        d = dropout if dropout is not None else 0.3
        return SEResNet(num_classes, in_channels, dropout=d)
    elif name.lower() == 'optimizedseresnet':
        d = dropout if dropout is not None else 0.2
        return OptimizedSEResNet(num_classes, in_channels, dropout=d)
    else:
        raise ValueError(f"Unknown model: {name}")
