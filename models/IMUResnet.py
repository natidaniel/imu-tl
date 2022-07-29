"""
The code is based on the original ResNet implementation from torchvision.models.resnet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.conv2 = conv3(out_planes, out_planes, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.conv3 = nn.Conv1d(out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class dummyOutput(nn.Module):
    def __init__(self, xx, yy, **kwargs):
        super(dummyOutput, self).__init__()

    def forward(self, x):
        return x


class FCOutputModule(nn.Module):
    """
    Fully connected output module.
    """
    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        Constructor for a fully connected output layer.

        Args:
          in_planes: number of planes (channels) of the layer immediately proceeding the output module.
          num_outputs: number of output predictions.
          fc_dim: dimension of the fully connected layer.
          dropout: the keep probability of the dropout layer
          trans_planes: (optional) number of planes of the transition convolutional layer.
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get('fc_dim', 1024)
        dropout = kwargs.get('dropout', 0.5)
        in_dim = kwargs.get('in_dim', 7)
        trans_planes = kwargs.get('trans_planes', None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes))
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs))

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y


class GlobAvgOutputModule(nn.Module):
    """
    Global average output module.
    """
    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)

    def get_dropout(self):
        return []

    def forward(self, x):
        x = self.avg()
        x = x.view(x.size(0), -1)
        return self.fc(x)


class IMUResnet(nn.Module):
    def __init__(self, config):

        super(IMUResnet, self).__init__()
        input_dim = config.get("input_dim")
        num_classes = config.get("num_classes")

        config = config.get("imu-resnet")

        fc_dim = config.get("fc_dim")
        dropout = config.get("dropout")
        kernel_size = config.get("kernel_size")
        base_plane = config.get("base_plane")
        self.filter = config.get("filter")
        self.downsampling = config.get("downsampling")
        zero_init_residual = config.get("zero_init_residual")

        block_type = BasicBlock1D
        group_sizes = [2, 2, 2, 2]
        self.inplanes = base_plane

        if self.filter is not None:
            self.weights = torch.zeros(input_dim, input_dim, self.filter)
            for out_ch in range(input_dim):
                self.weights[out_ch][out_ch] = torch.ones([1, self.filter]) / self.filter
            if torch.cuda.is_available():
                self.weights = self.weights.cuda()

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(input_dim, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual groups
        self.planes = [base_plane * (2 ** i) for i in range(len(group_sizes))]
        strides = [1] + [2] * (len(group_sizes) - 1)
        dilations = [1] * len(group_sizes)
        groups = [self._make_residual_group1d(block_type, self.planes[i], kernel_size, group_sizes[i],
                                              strides[i], dilations[i])
                  for i in range(len(group_sizes))]
        self.residual_groups = nn.Sequential(*groups)

        self.classifier = nn.Sequential(
            nn.Linear(fc_dim//4, fc_dim//4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim//4, fc_dim//4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim//4, num_classes))

        self.logsoftmax = nn.LogSoftmax(dim=1)

        # init
        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block_type, planes, kernel_size, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_type.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block_type.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block_type.expansion))
        layers = []
        layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size,
                                 stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block_type.expansion
        for _ in range(1, blocks):
            layers.append(block_type(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.filter:
            padding = self.filter//2
            x = F.pad(x, (padding, padding), mode='replicate')
            x = F.conv1d(x, self.weights)
        if self.downsampling:
            x = F.interpolate(x, scale_factor=0.5, mode='linear')
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.logsoftmax(x)



