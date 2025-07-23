import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Union, List, Optional, Callable


# ---------------------- 辅助卷积函数 ---------------------- #
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 卷积，带填充"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------- SE 注意力模块实现 ---------------------- #
class SqueezeExcitation(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        """
        初始化 SE 模块
        参数：
            channel: 输入特征图的通道数
            reduction: 压缩比率（默认16）
        """
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: 对每个通道进行全局平均池化
        scale = self.avg_pool(x)
        # Excitation: 两层全连接（使用1x1卷积实现），再通过 Sigmoid 得到通道权重
        scale = self.fc(scale)
        # 重标定：将输入与通道权重逐元素相乘
        return x * scale


# ---------------------- Bottleneck 模块（与 ResNet-50 相同） ---------------------- #
class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 集成 SE 注意力模块，作用于 conv3 后的特征图
        self.se = SqueezeExcitation(planes * self.expansion, reduction=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 将 SE 模块应用于输出
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ---------------------- 修改后的 ResNet 类，增加中间特征图返回 ---------------------- #
class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  # Output: /2
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Output: /4
        self.layer1 = self._make_layer(block, 64, layers[0])  # Output: /4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Output: /8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Output: /16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Output: /32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 封装 conv1 到 layer2 为 feature_extractor_stage1
        self.feature_extractor_stage1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        )

        # 定义 feature_extractor_stage2，为 layer3
        self.feature_extractor_stage2 = self.layer3

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block: Type[Union[Bottleneck]], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # 使用 feature_extractor_stage1 提取 feature_64（包含 conv1 到 layer2 的输出）
        feature_64 = self.feature_extractor_stage1(x)
        # 使用 feature_extractor_stage2 提取 feature_32（layer3 的输出）
        feature_32 = self.feature_extractor_stage2(feature_64)
        # 后续层处理 layer4、全局池化和全连接
        x = self.layer4(feature_32)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, feature_64, feature_32


# ---------------------- 修改后的 SiameseResNet50 类 ---------------------- #
class SiameseResNet50(nn.Module):
    def __init__(self) -> None:
        super(SiameseResNet50, self).__init__()
        self.feature_extractor = ResNet(Bottleneck, [3, 4, 6, 3])
        # 可选择加载预训练权重或冻结参数，若需要则取消下列注释：
        # self.feature_extractor.load_state_dict(torch.load('resnet50_pretrained.pth'))
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # 通过共享的特征提取器分别处理两个输入
        _, feat1_64, feat1_32 = self.feature_extractor(x1)
        _, feat2_64, feat2_32 = self.feature_extractor(x2)
        return feat1_64, feat1_32, feat2_64, feat2_32
